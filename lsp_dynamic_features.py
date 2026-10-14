"""Caracteristicas de las senas con movimiento (dinamicas).

Parte de los puntos que MediaPipe Holistic saco de un video y produce un vector de tamano
fijo para el SVM. Solo usa numpy: no abre videos ni MediaPipe, asi que sirve igual para
el entrenamiento, la prueba en vivo y las pruebas automaticas.

Pasos:
 1. Todo se expresa respecto al CUERPO: origen en el centro de los hombros y unidad = ancho de
    hombros. Asi da igual la distancia a la camara (a diferencia del vector de la captura estatica).
 2. La forma de la mano se expresa respecto a su muneca y se escala por el largo de la palma.
 3. La mano que mas se mueve es la principal. Si es la derecha se refleja, de modo que quien
    firma con la derecha y quien firma con la izquierda producen el mismo vector.
 4. La otra mano solo cuenta si ella tambien se mueve (senas de dos manos); si solo descansa,
    se ignora (queda en ceros).
 5. Se recorta el bloque de mayor actividad (se descartan el reposo del inicio y la mano
    quieta del final) y se remuestrea a LONGITUD frames.

Si se cambia algun parametro de abajo, el formato del dataset cambia: hay que volver a
extraer con `extraer_secuencias.py --forzar`. Los parametros se guardan dentro del dataset
para detectar mezclas de formatos.
"""
import numpy as np

# ----- Parametros del formato -----
LONGITUD = 20                  # frames de cada secuencia tras remuestrear
CARACT_POR_MANO = 44           # 2 (posicion de la muneca) + 42 (21 puntos x, y respecto a la muneca)
DIM_VECTOR = LONGITUD * CARACT_POR_MANO * 2
VERSION_FORMATO = 1

# ----- Parametros del recorte -----
MAX_HUECO = 5                  # huecos de deteccion de hasta N frames se rellenan por interpolacion
UMBRAL_VEL = 0.010             # velocidad de la muneca (anchos de hombro por frame) que cuenta como movimiento
UMBRAL_FORMA = 0.030           # cambio medio de la forma de la mano por frame que cuenta como movimiento
FUSIONAR_HUECOS = 6            # bloques activos separados por <= N frames se unen en uno
MARGEN = 3                     # frames de calma que se conservan a cada lado de cada tramo con movimiento
MODO_RECORTE = 'bloque'        # 'bloque' (solo el tramo mas energico) o 'activos' (todo el movimiento, sin las pausas largas)
PESO_TRASLACION = 0.5          # al elegir el bloque, el desplazamiento de la muneca pesa la mitad que el cambio de forma:
                               # asi subir la mano desde abajo no le gana al trazo de la sena
RAZON_AMBIGUO = 1.3            # si el mejor bloque no supera al segundo por este factor, el recorte se marca como ambiguo
MIN_RACHA_ACTIVA = 2           # una racha de movimiento de 1 solo frame se considera ruido
MIN_FRAMES_BLOQUE = 8          # un bloque mas corto no se considera una sena
MAX_FALTANTES = 0.30           # fraccion maxima de frames sin la mano dentro del bloque
UMBRAL_MANO_SECUNDARIA = 0.5   # recorrido (en anchos de hombro) a partir del cual la otra mano "firma"
VIS_HOMBROS = 0.5              # visibilidad minima de los hombros para usarlos de referencia

# ----- Estados que devuelve analizar_movimiento -----
# Los motivos son textos porque salen en el reporte; los estados permiten que la prueba
# en vivo sepa que paso sin tener que interpretar el texto.
ESTADO_OK = 'ok'                    # hay un tramo con movimiento y se puede clasificar
ESTADO_SIN_MOVIMIENTO = 'sin_movimiento'   # la mano esta quieta
ESTADO_CORTO = 'movimiento_corto'   # se mueve, pero todavia en muy pocos frames
ESTADO_SIN_DATOS = 'sin_datos'      # no se puede analizar (sin hombros, sin mano, mano perdida...)
MOTIVO_SIN_MOVIMIENTO = 'no se detecto movimiento de la mano'


def parametros():
    """Parametros del formato (se guardan en el dataset para detectar mezclas)."""
    return {'version': VERSION_FORMATO, 'longitud': LONGITUD, 'caract_por_mano': CARACT_POR_MANO,
            'max_hueco': MAX_HUECO, 'umbral_vel': UMBRAL_VEL, 'umbral_forma': UMBRAL_FORMA,
            'fusionar_huecos': FUSIONAR_HUECOS, 'margen': MARGEN, 'min_frames_bloque': MIN_FRAMES_BLOQUE,
            'umbral_mano_secundaria': UMBRAL_MANO_SECUNDARIA}


def referencia_cuerpo(pose, ancho, alto):
    """Centro y ancho de los hombros (en pixeles), usando la mediana de todo el video.

    `pose` es (T, 33, 4): x, y, z, visibilidad. Devuelve (centro, ancho_hombros) o None.
    """
    xy = pose[:, [11, 12], :2] * np.array([ancho, alto])
    vis = pose[:, [11, 12], 3]
    buenos = (np.nan_to_num(vis, nan=0.0) >= VIS_HOMBROS).all(axis=1) & ~np.isnan(xy).any(axis=(1, 2))
    if buenos.sum() < 3:
        return None
    centro = np.median(xy[buenos].mean(axis=1), axis=0)
    ancho_h = np.median(np.linalg.norm(xy[buenos, 0] - xy[buenos, 1], axis=1))
    if ancho_h < 1e-6:
        return None
    return centro, float(ancho_h)


def caracteristicas_mano(mano, centro, ancho_h, ancho, alto):
    """(T, 21, 3) de MediaPipe -> (presente, posicion_muneca (T,2), forma (T,42)); NaN donde no hay mano."""
    t = len(mano)
    presente = ~np.isnan(mano[:, 0, 0])
    pos = np.full((t, 2), np.nan)
    forma = np.full((t, 42), np.nan)
    if presente.any():
        px = mano[presente, :, :2] * np.array([ancho, alto])                 # (n, 21, 2) en pixeles
        palma = np.linalg.norm(px[:, 0] - px[:, 9], axis=1)
        palma = np.maximum(palma, 1e-6)
        pos[presente] = (px[:, 0] - centro) / ancho_h
        forma[presente] = ((px - px[:, :1]) / palma[:, None, None]).reshape(-1, 42)
    return presente, pos, forma


def rellenar_huecos(serie, presente, max_hueco=MAX_HUECO):
    """Interpola los huecos cortos (<= max_hueco frames) entre dos detecciones. Los largos quedan ausentes."""
    serie, presente = serie.copy(), presente.copy()
    idx = np.flatnonzero(presente)
    for a, b in zip(idx[:-1], idx[1:]):
        hueco = b - a - 1
        if 0 < hueco <= max_hueco:
            w = np.linspace(0.0, 1.0, hueco + 2)[1:-1][:, None]
            serie[a + 1:b] = serie[a] * (1 - w) + serie[b] * w
            presente[a + 1:b] = True
    return serie, presente


def recorrido(pos, presente):
    """Longitud del camino de la muneca (en anchos de hombro) entre frames consecutivos con mano."""
    if presente.sum() < 2:
        return 0.0
    ambos = presente[1:] & presente[:-1]
    pasos = np.linalg.norm(np.diff(np.nan_to_num(pos), axis=0), axis=1)
    return float(pasos[ambos].sum())


def actividad(pos, forma, presente):
    """Devuelve (a, energia) por frame.

    `a` es el puntaje de movimiento (>= 1 significa que la mano se mueve); mezcla muneca y dedos.
    `energia` es lo mismo pero con menos peso al desplazamiento de la muneca; sirve para decidir
    cual de los bloques de movimiento es la sena.
    """
    n = len(presente)
    a, energia = np.zeros(n), np.zeros(n)
    if n < 3:
        return a, energia
    ambos = presente[1:] & presente[:-1]
    vel = np.linalg.norm(np.diff(np.nan_to_num(pos), axis=0), axis=1)
    cambio = np.abs(np.diff(np.nan_to_num(forma), axis=0)).mean(axis=1)
    a[1:][ambos] = np.maximum(vel / UMBRAL_VEL, cambio / UMBRAL_FORMA)[ambos]
    energia[1:][ambos] = np.maximum(PESO_TRASLACION * vel / UMBRAL_VEL, cambio / UMBRAL_FORMA)[ambos]
    suave = np.ones(3) / 3
    return np.convolve(a, suave, mode='same'), np.convolve(energia, suave, mode='same')


def mejor_bloque(a, energia=None):
    """(inicio, fin, razon) del bloque de mayor energia total, con huecos cortos fusionados. None si no hay.

    `razon` es cuantas veces supera el mejor bloque al segundo (infinito si solo hay uno).
    """
    energia = a if energia is None else energia
    idx = np.flatnonzero(a >= 1.0)
    if len(idx) == 0:
        return None
    bloques, ini, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > FUSIONAR_HUECOS:
            bloques.append((ini, prev))
            ini = i
        prev = i
    bloques.append((ini, prev))
    puntajes = [energia[b[0]:b[1] + 1].sum() for b in bloques]
    orden = np.argsort(puntajes)[::-1]
    ini, fin = bloques[orden[0]]
    razon = float('inf') if len(bloques) == 1 or puntajes[orden[1]] <= 0 else puntajes[orden[0]] / puntajes[orden[1]]
    return max(0, ini - MARGEN), min(len(a) - 1, fin + MARGEN), razon


def frames_activos(a):
    """Indices (ordenados) de los frames a conservar: los que tienen movimiento mas un margen alrededor.

    Las pausas largas (la mano quieta antes, entre medias o despues del movimiento) se acortan a
    2 * MARGEN frames, asi que da igual cuanto tiempo se quede quieta la mano: la secuencia resultante
    es la misma. Una racha de movimiento de un solo frame se descarta como ruido.
    """
    activo = a >= 1.0
    # quitar rachas activas demasiado cortas
    i, n = 0, len(activo)
    while i < n:
        if activo[i]:
            j = i
            while j + 1 < n and activo[j + 1]:
                j += 1
            if j - i + 1 < MIN_RACHA_ACTIVA:
                activo[i:j + 1] = False
            i = j + 1
        else:
            i += 1
    if not activo.any():
        return None
    conservar = activo.copy()
    for k in range(1, MARGEN + 1):
        conservar[:-k] |= activo[k:]
        conservar[k:] |= activo[:-k]
    return np.flatnonzero(conservar)


def _reflejar(pos, forma):
    pos = pos * np.array([-1.0, 1.0])
    forma = (forma.reshape(-1, 21, 2) * np.array([-1.0, 1.0])).reshape(-1, 42)
    return pos, forma


def _rellenar_bordes(m):
    """Frames sin mano dentro del bloque: se repite el valor valido mas cercano."""
    m = m.copy()
    validos = ~np.isnan(m).any(axis=1)
    if not validos.any():
        return np.zeros_like(m)
    idx = np.where(validos, np.arange(len(m)), -1)
    np.maximum.accumulate(idx, out=idx)                       # ultimo valido hacia atras
    primero = np.flatnonzero(validos)[0]
    idx[idx < 0] = primero
    return m[idx]


def remuestrear(m, n=LONGITUD):
    x = np.linspace(0.0, 1.0, len(m))
    xn = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(xn, x, m[:, j]) for j in range(m.shape[1])], axis=1)


def analizar_movimiento(pose, izq, der, ancho, alto):
    """Prepara las manos y localiza el tramo con movimiento de UNA muestra.

    Es la primera mitad de extraer_caracteristicas, expuesta aparte para que la prueba en
    vivo corte la seña con exactamente el mismo criterio que la extraccion de videos.

    Devuelve (datos, info).
      * datos = None si no hay tramo que clasificar; entonces info['estado'] dice que paso
        (ESTADO_SIN_MOVIMIENTO, ESTADO_CORTO o ESTADO_SIN_DATOS) e info['motivo'] lo explica.
      * datos = {'idx', 'manos', 'prim', 'sec', 'dos_manos'} si hay tramo usable; se lo
        tiene que pasar a vector_de_tramo() para obtener el vector.
    """
    info = {'estado': ESTADO_SIN_DATOS}
    ref = referencia_cuerpo(pose, ancho, alto)
    if ref is None:
        return None, {**info, 'motivo': 'no se detectaron los hombros'}
    centro, ancho_h = ref

    manos = {}
    for nombre, mano in (('izq', izq), ('der', der)):
        pres, pos, forma = caracteristicas_mano(mano, centro, ancho_h, ancho, alto)
        pos, _ = rellenar_huecos(pos, pres)
        forma, pres = rellenar_huecos(forma, pres)
        manos[nombre] = dict(pres=pres, pos=pos, forma=forma, rec=recorrido(pos, pres))
    info['recorridos'] = {k: round(v['rec'], 3) for k, v in manos.items()}

    prim = max(manos, key=lambda k: (manos[k]['rec'], manos[k]['pres'].sum()))
    sec = 'der' if prim == 'izq' else 'izq'
    if not manos[prim]['pres'].any():
        return None, {**info, 'motivo': 'no se detecto ninguna mano', 'estado': ESTADO_SIN_DATOS}
    dos_manos = manos[sec]['rec'] > UMBRAL_MANO_SECUNDARIA
    info.update(lado_principal=prim, dos_manos=bool(dos_manos))

    # Huecos de deteccion entre la primera y la ultima vez que se ve la mano (entrar o salir del
    # encuadre no cuenta). Si la mano se pierde mucho, el tramo que si se ve no representa la sena.
    vistos = np.flatnonzero(manos[prim]['pres'])
    huecos = 1.0 - manos[prim]['pres'][vistos[0]:vistos[-1] + 1].mean()
    info['huecos_internos'] = round(float(huecos), 3)
    if huecos > MAX_FALTANTES:
        return None, {**info, 'motivo': f'la mano se perdio en el {huecos * 100:.0f}% del video',
                      'estado': ESTADO_SIN_DATOS}

    a, energia = actividad(manos[prim]['pos'], manos[prim]['forma'], manos[prim]['pres'])
    if dos_manos:
        a2, e2 = actividad(manos[sec]['pos'], manos[sec]['forma'], manos[sec]['pres'])
        a, energia = np.maximum(a, a2), np.maximum(energia, e2)
    if MODO_RECORTE == 'bloque':
        bloque = mejor_bloque(a, energia)
        idx = None if bloque is None else np.arange(bloque[0], bloque[1] + 1)
        if bloque is not None:
            info['ambiguo'] = bool(bloque[2] < RAZON_AMBIGUO)
            info['razon_bloques'] = None if np.isinf(bloque[2]) else round(float(bloque[2]), 2)
    else:
        idx = frames_activos(a)
    if idx is None:
        return None, {**info, 'motivo': MOTIVO_SIN_MOVIMIENTO, 'estado': ESTADO_SIN_MOVIMIENTO}
    if len(idx) < MIN_FRAMES_BLOQUE:
        return None, {**info, 'motivo': f'el movimiento dura solo {len(idx)} frames (minimo {MIN_FRAMES_BLOQUE})',
                      'estado': ESTADO_CORTO}

    # El video se acabo (o empezo) con la mano todavia en movimiento: la sena puede estar cortada.
    info['cortada_al_final'] = bool(idx[-1] >= len(a) - 1 and a[-3:].max() >= 1.0)
    info['cortada_al_inicio'] = bool(idx[0] <= 0 and a[:3].max() >= 1.0)

    faltan = 1.0 - manos[prim]['pres'][idx].mean()
    # frames_activos cuenta solo los frames con movimiento de verdad: el bloque trae MARGEN
    # frames de calma a cada lado, asi que su largo no sirve para saber si hubo una seña ni
    # cuando termino. primer_activo/ultimo_activo son los frames con movimiento real.
    con_movimiento = np.flatnonzero(a >= 1.0)
    info.update(bloque=(int(idx[0]), int(idx[-1])), frames_bloque=int(len(idx)),
                frames_activos=int((a[idx] >= 1.0).sum()),
                primer_activo=int(con_movimiento[0]), ultimo_activo=int(con_movimiento[-1]),
                pausas_recortadas=int(idx[-1] - idx[0] + 1 - len(idx)), faltantes=round(float(faltan), 3))
    if faltan > MAX_FALTANTES:
        return None, {**info, 'motivo': f'la mano se perdio en el {faltan * 100:.0f}% del movimiento',
                      'estado': ESTADO_SIN_DATOS}

    info['estado'] = ESTADO_OK
    return {'idx': idx, 'manos': manos, 'prim': prim, 'sec': sec, 'dos_manos': dos_manos}, info


def vector_de_tramo(datos, info):
    """Vector de caracteristicas del tramo que devolvio analizar_movimiento.

    Escribe en info['secuencia'] el tramo recortado sin remuestrear (lo usa DTW, que lo
    remuestrea a su propia longitud) y devuelve el vector fijo que se le da al SVM.
    """
    idx, manos = datos['idx'], datos['manos']
    prim, sec, dos_manos = datos['prim'], datos['sec'], datos['dos_manos']

    partes = []
    for k, activa in ((prim, True), (sec, dos_manos)):
        if not activa:
            partes.append(np.zeros((len(idx), CARACT_POR_MANO)))
            continue
        pos, forma = manos[k]['pos'].copy(), manos[k]['forma'].copy()
        pos[~manos[k]['pres']] = np.nan
        forma[~manos[k]['pres']] = np.nan
        if prim == 'der':
            pos, forma = _reflejar(pos, forma)
        partes.append(_rellenar_bordes(np.hstack([pos, forma]))[idx])

    sin_remuestrear = np.hstack(partes)
    info['secuencia'] = sin_remuestrear.astype(np.float32)
    return remuestrear(sin_remuestrear).astype(np.float32).ravel()


def extraer_caracteristicas(pose, izq, der, ancho, alto):
    """Vector de caracteristicas de UNA muestra.

    pose (T,33,4), izq y der (T,21,3) con NaN donde MediaPipe no detecto la mano.
    Devuelve (vector (DIM_VECTOR,), info). Si la muestra no sirve, vector = None e
    info['motivo'] explica por que (los motivos van sin tildes).
    """
    datos, info = analizar_movimiento(pose, izq, der, ancho, alto)
    if datos is None:
        return None, info
    return vector_de_tramo(datos, info), info
