"""Entrenamiento, evaluacion y prediccion de las senas con movimiento. Sin interfaz.

Lo usa GUI/dynamic_training_widget.py, pero se puede ejecutar solo:

    python lsp_dynamic_train.py                 # entrena con dataset_dinamico.pickle
    python lsp_dynamic_train.py --modelo dtw    # fuerza un tipo de modelo (auto, rbf o dtw)

Se comparan dos modelos, los dos SVM:
  * "rbf": SVM con kernel RBF sobre el vector fijo de 20 frames.
  * "dtw": SVM cuyo kernel se calcula a partir de la distancia DTW entre secuencias. DTW alinea
           dos secuencias aunque una vaya mas rapida o tenga pausas distintas.

La evaluacion es honesta: si hay 2 o mas personas se entrena con todas menos una y se prueba con
la que quedo fuera (repitiendo con cada una). Con una sola persona solo se puede validar con
pliegues, y la exactitud sale optimista porque entrenamiento y prueba comparten condiciones.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import lsp_dynamic_features as feat

LONGITUD_DTW = 30            # frames a los que se remuestrea cada secuencia para comparar con DTW
MIN_MUESTRAS_CLASE = 3       # una clase con menos muestras se deja fuera del entrenamiento
GRID_C = (1.0, 10.0, 100.0)
GRID_GAMMA_RBF = (0.3, 1.0, 3.0)   # multiplos del gamma "natural" (1 / numero de caracteristicas)
GRID_GAMMA_DTW = (1.0, 2.0, 4.0)   # kernel = exp(-g * distancia / distancia_mediana)

RUTA_DATASET = os.path.join(ROOT_DIR, 'dataset_dinamico.pickle')
RUTA_MODELO = os.path.join(ROOT_DIR, 'svm_dinamico_lsp_classifier.p')
RUTA_MATRIZ = os.path.join(ROOT_DIR, 'matriz_confusion_dinamico.png')
RUTA_REPORTE = os.path.join(ROOT_DIR, 'reporte_dinamico.txt')


class DatasetNoValido(Exception):
    """El dataset no se puede usar para entrenar (el mensaje explica por que, en espanol)."""


# --------------------------------------------------------------------------- datos
def cargar_dataset(ruta=RUTA_DATASET):
    if not os.path.exists(ruta):
        raise DatasetNoValido('No existe dataset_dinamico.pickle. Primero procesa los videos '
                              '(python extraer_secuencias.py).')
    with open(ruta, 'rb') as f:
        d = pickle.load(f)
    faltan = [k for k in ('data', 'labels', 'personas', 'secuencias', 'formato') if k not in d]
    if faltan:
        raise DatasetNoValido(f'El dataset no tiene el formato esperado (faltan: {", ".join(faltan)}). '
                              'Vuelve a generarlo con python extraer_secuencias.py --forzar.')
    if d['formato'] != feat.parametros():
        raise DatasetNoValido('El dataset se extrajo con otros parametros de caracteristicas que los actuales. '
                              'Vuelve a generarlo con python extraer_secuencias.py --forzar.')
    return d


def preparar(d):
    """Devuelve (X, S, y, personas, avisos): vectores fijos, secuencias para DTW, etiquetas y personas.

    Las clases con menos de MIN_MUESTRAS_CLASE muestras se dejan fuera y se avisa. Lanza
    DatasetNoValido si no quedan al menos 2 clases.
    """
    y = np.array(d['labels'])
    personas = np.array(d['personas'])
    clases, cuentas = np.unique(y, return_counts=True)
    avisos = []
    pocas = [c for c, n in zip(clases, cuentas) if n < MIN_MUESTRAS_CLASE]
    if pocas:
        avisos.append(f'Se dejan fuera por tener menos de {MIN_MUESTRAS_CLASE} muestras: '
                      + ', '.join(f'{c} ({int(cuentas[list(clases).index(c)])})' for c in pocas))
    quedan = np.isin(y, [c for c in clases if c not in pocas])
    if len(set(y[quedan])) < 2:
        raise DatasetNoValido('Hacen falta al menos 2 clases con 3 o mas muestras cada una para entrenar. '
                              'Graba otra senal (por ejemplo la Z) y vuelve a procesar los videos.')
    X = np.asarray(d['data'], dtype=np.float64)[quedan]
    S = np.stack([feat.remuestrear(np.asarray(s, dtype=np.float64), LONGITUD_DTW)
                  for s, q in zip(d['secuencias'], quedan) if q])
    return X, S, y[quedan], personas[quedan], avisos


# --------------------------------------------------------------------------- DTW
def dtw_distancias(A, B, simetrica=False, lote=3000):
    """Distancia DTW normalizada entre cada secuencia de A (nA, L, d) y cada una de B (nB, L, d).

    Calcula muchos pares a la vez con numpy (el bucle de Python recorre solo la rejilla L x L).
    Si `simetrica` es True, A y B son lo mismo y solo se calcula la mitad.
    """
    nA, L, _ = A.shape
    nB = B.shape[0]
    if simetrica:
        ia, ib = np.triu_indices(nA, k=1)
    else:
        ia, ib = np.divmod(np.arange(nA * nB), nB)
    valores = np.empty(len(ia))
    for ini in range(0, len(ia), lote):
        a, b = A[ia[ini:ini + lote]], B[ib[ini:ini + lote]]
        p = len(a)
        costo = np.sqrt(np.maximum((a ** 2).sum(-1)[:, :, None] + (b ** 2).sum(-1)[:, None, :]
                                   - 2.0 * np.matmul(a, b.transpose(0, 2, 1)), 0.0))
        acum = np.full((p, L + 1, L + 1), np.inf)
        acum[:, 0, 0] = 0.0
        for i in range(1, L + 1):
            for j in range(1, L + 1):
                acum[:, i, j] = costo[:, i - 1, j - 1] + np.minimum(
                    np.minimum(acum[:, i - 1, j], acum[:, i, j - 1]), acum[:, i - 1, j - 1])
        valores[ini:ini + lote] = acum[:, L, L] / (2 * L)
    if simetrica:
        D = np.zeros((nA, nA))
        D[ia, ib] = valores
        D[ib, ia] = valores
        return D
    return valores.reshape(nA, nB)


# --------------------------------------------------------------------------- modelos
def _validacion_interna(y, k=3):
    minimo = int(np.unique(y, return_counts=True)[1].min())
    k = min(k, minimo)
    return None if k < 2 else StratifiedKFold(n_splits=k, shuffle=True, random_state=0)


def ajustar_rbf(X, y):
    """SVM RBF sobre el vector fijo; C y gamma se eligen con validacion cruzada interna."""
    escalador = StandardScaler().fit(X)
    Xs = escalador.transform(X)
    base = 1.0 / Xs.shape[1]
    cv = _validacion_interna(y)
    mejor = (-1.0, (10.0, base))
    if cv is not None:
        for m in GRID_GAMMA_RBF:
            for C in GRID_C:
                aciertos = 0
                for a, b in cv.split(Xs, y):
                    svc = SVC(C=C, kernel='rbf', gamma=base * m).fit(Xs[a], y[a])
                    aciertos += int((svc.predict(Xs[b]) == y[b]).sum())
                if aciertos / len(y) > mejor[0] + 1e-12:
                    mejor = (aciertos / len(y), (C, base * m))
    C, gamma = mejor[1]
    return {'tipo': 'rbf', 'escalador': escalador, 'svc': SVC(C=C, kernel='rbf', gamma=gamma).fit(Xs, y),
            'C': C, 'gamma': gamma, 'cv_interna': mejor[0]}


def ajustar_dtw(D, y):
    """SVM con kernel exp(-g * D / mediana). `D` es la matriz de distancias DTW entre las muestras de entrenamiento."""
    n = len(y)
    escala = float(np.median(D[~np.eye(n, dtype=bool)])) or 1.0
    cv = _validacion_interna(y)
    mejor = (-1.0, (10.0, 2.0))
    if cv is not None:
        for g in GRID_GAMMA_DTW:
            K = np.exp(-g * D / escala)
            for C in GRID_C:
                aciertos = 0
                for a, b in cv.split(np.zeros(n), y):
                    svc = SVC(C=C, kernel='precomputed').fit(K[np.ix_(a, a)], y[a])
                    aciertos += int((svc.predict(K[np.ix_(b, a)]) == y[b]).sum())
                if aciertos / n > mejor[0] + 1e-12:
                    mejor = (aciertos / n, (C, g))
    C, g = mejor[1]
    return {'tipo': 'dtw', 'svc': SVC(C=C, kernel='precomputed').fit(np.exp(-g * D / escala), y),
            'C': C, 'g': g, 'escala': escala, 'cv_interna': mejor[0]}


# --------------------------------------------------------------------------- evaluacion
def particiones(y, personas):
    """Lista de (nombre, indices_entrenamiento, indices_prueba) y la estrategia usada."""
    unicas = sorted(set(personas))
    if len(unicas) >= 2:
        partes = []
        for p in unicas:
            te = np.flatnonzero(personas == p)
            tr = np.flatnonzero(personas != p)
            if len(set(y[tr])) >= 2:
                partes.append((str(p), tr, te))
        return partes, 'persona'
    minimo = int(np.unique(y, return_counts=True)[1].min())
    k = min(5, minimo)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    return [(f'pliegue {i + 1}', tr, te) for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))], 'pliegues'


def evaluar(X, S, y, personas, tipos=('rbf', 'dtw'), progreso=print):
    """Valida cada tipo de modelo. Devuelve un dict con exactitud, matriz de confusion y detalle por modelo."""
    partes, estrategia = particiones(y, personas)
    if not partes:
        raise DatasetNoValido('No se puede validar: al dejar fuera a cada persona no quedan 2 clases para entrenar. '
                              'Hace falta que cada persona grabe varias clases.')
    clases = sorted(set(y))
    progreso(f'Validacion: {"dejando una persona fuera" if estrategia == "persona" else "pliegues (una sola persona)"} '
             f'- {len(partes)} repeticiones.')
    D = None
    if 'dtw' in tipos:
        progreso('Calculando las distancias DTW entre todas las muestras...')
        t0 = time.time()
        D = dtw_distancias(S, S, simetrica=True)
        progreso(f'  listo en {time.time() - t0:.1f} s')

    resultados = {}
    for tipo in tipos:
        verdad, prediccion, por_grupo, omitidas = [], [], {}, 0
        for nombre, tr, te in partes:
            conocidas = np.isin(y[te], y[tr])          # una clase que no estaba en el entrenamiento no se puede acertar
            omitidas += int((~conocidas).sum())
            te = te[conocidas]
            if len(te) == 0:
                continue
            if tipo == 'rbf':
                m = ajustar_rbf(X[tr], y[tr])
                pred = m['svc'].predict(m['escalador'].transform(X[te]))
            else:
                m = ajustar_dtw(D[np.ix_(tr, tr)], y[tr])
                pred = m['svc'].predict(np.exp(-m['g'] * D[np.ix_(te, tr)] / m['escala']))
            verdad += list(y[te])
            prediccion += list(pred)
            por_grupo[nombre] = float(np.mean(np.array(pred) == y[te]))
            progreso(f'  [{tipo}] {nombre}: {por_grupo[nombre] * 100:.0f}% ({len(te)} muestras)')
        verdad, prediccion = np.array(verdad), np.array(prediccion)
        from sklearn.metrics import confusion_matrix
        resultados[tipo] = {'exactitud': float(np.mean(verdad == prediccion)), 'por_grupo': por_grupo,
                            'matriz': confusion_matrix(verdad, prediccion, labels=clases),
                            'verdad': verdad, 'prediccion': prediccion, 'omitidas': omitidas}
    return {'estrategia': estrategia, 'clases': clases, 'resultados': resultados, 'D': D}


def entrenar_final(X, S, y, tipo, D=None):
    """Entrena el modelo definitivo con TODAS las muestras."""
    if tipo == 'rbf':
        m = ajustar_rbf(X, y)
        return {'tipo': 'rbf', 'escalador': m['escalador'], 'svc': m['svc'], 'C': m['C'], 'gamma': m['gamma']}
    if D is None:
        D = dtw_distancias(S, S, simetrica=True)
    m = ajustar_dtw(D, y)
    return {'tipo': 'dtw', 'svc': m['svc'], 'C': m['C'], 'g': m['g'], 'escala': m['escala'],
            'secuencias_train': S.astype(np.float32)}


def predecir(modelo, vector, secuencia):
    """Clasifica UNA muestra. `vector` y `secuencia` son los que devuelve feat.extraer_caracteristicas.

    Devuelve (clase, detalle). Para el modelo DTW, detalle['dist_min'] es la distancia DTW a la muestra
    de entrenamiento mas parecida (sirve para rechazar senas que no son de ninguna clase).
    """
    if modelo['tipo'] == 'rbf':
        x = modelo['escalador'].transform(np.asarray(vector, dtype=np.float64)[None])
        return str(modelo['svc'].predict(x)[0]), {'dist_min': None}
    s = feat.remuestrear(np.asarray(secuencia, dtype=np.float64), modelo['longitud_dtw'])[None]
    d = dtw_distancias(s, modelo['secuencias_train'].astype(np.float64))
    clase = modelo['svc'].predict(np.exp(-modelo['g'] * d / modelo['escala']))[0]
    return str(clase), {'dist_min': float(d.min())}


def guardar_modelo(modelo, ruta=RUTA_MODELO):
    with open(ruta, 'wb') as f:
        pickle.dump(modelo, f)


def cargar_modelo(ruta=RUTA_MODELO):
    if not os.path.exists(ruta):
        raise DatasetNoValido('Todavia no hay un modelo dinamico entrenado.')
    with open(ruta, 'rb') as f:
        m = pickle.load(f)
    if m.get('formato') != feat.parametros():
        raise DatasetNoValido('El modelo se entreno con otros parametros de caracteristicas. Vuelve a entrenarlo.')
    return m


# --------------------------------------------------------------------------- informes
def confusiones_principales(matriz, clases, n=5):
    pares = [(clases[i], clases[j], int(matriz[i, j])) for i in range(len(clases)) for j in range(len(clases))
             if i != j and matriz[i, j] > 0]
    return sorted(pares, key=lambda t: -t[2])[:n]


def guardar_matriz(matriz, clases, ruta, titulo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(clases)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 2), max(5, 0.7 * n + 1.5)))
    ax.imshow(matriz, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(clases, rotation=45, ha='right')
    ax.set_yticklabels(clases)
    ax.set_xlabel('Predicha')
    ax.set_ylabel('Real')
    ax.set_title(titulo)
    tope = matriz.max() / 2 if matriz.size else 0
    for i in range(n):
        for j in range(n):
            if matriz[i, j]:
                ax.text(j, i, int(matriz[i, j]), ha='center', va='center',
                        color='white' if matriz[i, j] > tope else 'black')
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def texto_resumen(ev, elegido, avisos, conteo):
    nombres = {'rbf': 'SVM sobre el vector fijo (RBF)', 'dtw': 'SVM con kernel DTW'}
    unidad = 'persona' if ev['estrategia'] == 'persona' else 'pliegue'
    lineas = ['Muestras por clase: ' + ', '.join(f'{c} {n}' for c, n in conteo.items()), '']
    if ev['estrategia'] == 'pliegues':
        lineas += ['AVISO: hay una sola persona, asi que la validacion usa pliegues y la exactitud sale optimista.',
                   'Para una medida real, graba con otra persona y entrena de nuevo.', '']
    for tipo, r in ev['resultados'].items():
        marca = '  <- elegido' if tipo == elegido else ''
        lineas.append(f'{nombres[tipo]}: {r["exactitud"] * 100:.1f}%{marca}')
        if len(r['por_grupo']) > 1:
            lineas.append('   por ' + unidad + ': ' + ', '.join(f'{g} {v * 100:.0f}%' for g, v in r['por_grupo'].items()))
        if r['omitidas']:
            lineas.append(f'   ({r["omitidas"]} muestras omitidas: su clase no estaba en el entrenamiento)')
        for real, pred, n in confusiones_principales(r['matriz'], ev['clases'], 3):
            lineas.append(f'   confunde {real} con {pred}: {n} veces')
    if avisos:
        lineas += [''] + ['AVISO: ' + a for a in avisos]
    return '\n'.join(lineas)


# --------------------------------------------------------------------------- orquestacion
def entrenar(ruta_dataset=RUTA_DATASET, ruta_modelo=RUTA_MODELO, preferencia='auto', progreso=print,
             ruta_matriz=RUTA_MATRIZ, ruta_reporte=RUTA_REPORTE):
    """Evalua, elige, entrena el modelo definitivo y guarda modelo, matriz de confusion y reporte.

    preferencia: 'auto' (se queda con el que valide mejor), 'rbf' o 'dtw'.
    Devuelve un dict con 'resumen' (texto), 'elegido' y 'exactitud'. Lanza DatasetNoValido si no se puede.
    """
    d = cargar_dataset(ruta_dataset)
    X, S, y, personas, avisos = preparar(d)
    clases, cuentas = np.unique(y, return_counts=True)
    conteo = dict(zip(map(str, clases), map(int, cuentas)))
    progreso(f'{len(y)} muestras, {len(clases)} clases, {len(set(personas))} persona(s).')

    tipos = ('rbf', 'dtw') if preferencia == 'auto' else (preferencia,)
    ev = evaluar(X, S, y, personas, tipos, progreso)
    elegido = max(ev['resultados'], key=lambda t: (ev['resultados'][t]['exactitud'], t == 'rbf'))
    progreso(f'Modelo elegido: {elegido}. Entrenando con todas las muestras...')

    modelo = entrenar_final(X, S, y, elegido, ev['D'])
    modelo.update({'clases': list(map(str, clases)), 'formato': feat.parametros(), 'longitud_dtw': LONGITUD_DTW,
                   'creado': time.strftime('%Y-%m-%d %H:%M:%S'), 'personas': sorted(set(map(str, personas))),
                   'exactitud_validacion': ev['resultados'][elegido]['exactitud'], 'estrategia': ev['estrategia']})
    guardar_modelo(modelo, ruta_modelo)

    r = ev['resultados'][elegido]
    guardar_matriz(r['matriz'], ev['clases'], ruta_matriz, f'Matriz de confusion ({elegido}, {r["exactitud"] * 100:.1f}%)')
    resumen = texto_resumen(ev, elegido, avisos, conteo)
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(resumen + '\n')
    progreso('Modelo guardado.')
    return {'resumen': resumen, 'elegido': elegido, 'exactitud': r['exactitud'], 'estrategia': ev['estrategia'],
            'ruta_modelo': ruta_modelo, 'ruta_matriz': ruta_matriz, 'ruta_reporte': ruta_reporte}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=RUTA_DATASET)
    ap.add_argument('--modelo', default='auto', choices=('auto', 'rbf', 'dtw'))
    args = ap.parse_args()
    try:
        r = entrenar(args.dataset, preferencia=args.modelo)
    except DatasetNoValido as e:
        print(f'\nNo se puede entrenar: {e}')
        return 1
    print('\n' + r['resumen'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
