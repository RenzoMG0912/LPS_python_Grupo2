"""Reconocimiento en vivo de senas con movimiento: sin grabar video y sin esperar al final.

El flujo de captura (GUI/dynamic_collection_widget.py) guarda un video corto y despues lo
analiza. Eso suma la cuenta regresiva, los segundos de grabacion y un pase completo de
MediaPipe sobre frames que ya pasaron: medido en este equipo, ~62 ms por frame, es decir
~4,7 s de espera despues de grabar. Aqui se hace el mismo trabajo a medida que los frames
llegan:

 1. `Segmentador.agregar()` recibe los puntos de UN frame (los mismos que MediaPipe Holistic
    entrega al extraer un video).
 2. Mientras la mano se mueve, el buffer crece y no se devuelve nada.
 3. Cuando la mano se queda quieta INACTIVIDAD_FIN segundos, la seña ya termino: se devuelve
    el evento con el vector de caracteristicas calculado.

El corte del tramo y el vector los calcula SIEMPRE lsp_dynamic_features (las mismas
funciones que se usaron al crear el dataset y al entrenar). Por eso no hay que reentrenar
ni cambia la exactitud del modelo: lo que cambia es CUANDO se calcula, no QUE se calcula.

Solo usa numpy: no abre la camara y no importa MediaPipe ni sklearn, asi que el corte de la
seña se puede probar con secuencias sinteticas (ver tests/test_dinamico_en_vivo.py).
"""
import time

import numpy as np

import lsp_dynamic_features as feat

# ----- Cuando se da por terminada una seña -----
INACTIVIDAD_FIN = 0.45      # segundos de mano quieta que cierran la seña
MAX_DURACION = 6.0          # segundos maximos: si la mano nunca para, se corta igual
PRE_ROLL = 2 * feat.MARGEN  # frames de calma que se conservan antes del movimiento
MAX_FRAMES = 240            # tope de memoria del buffer (mas que MAX_DURACION a 30 fps)

QUIETO, MOVIENDO, ESPERANDO = 'quieto', 'moviendo', 'esperando'

# Cuanto movimiento de verdad se exige para dar por buena una seña.
# El bloque que devuelve el extractor incluye MARGEN frames de calma a cada lado, asi que
# mirar solo su largo dejaria pasar un temblor de 3 frames como si fuera una seña: por eso
# se cuentan los frames con actividad (info['frames_activos']).
MIN_FRAMES_ACTIVOS = feat.MIN_FRAMES_BLOQUE


class Segmentador:
    """Recorta las senas de un flujo continuo de frames.

    Se le entrega, frame a frame, lo mismo que MediaPipe Holistic le da a la extraccion de
    videos: pose (33,4), izq y der (21,3) con NaN donde no hubo deteccion, mas ancho y alto.
    Devuelve None mientras no haya una seña completa y, cuando la mano se queda quieta
    INACTIVIDAD_FIN segundos, devuelve el evento con el vector ya calculado.

    Atributos de estado, para mostrarlos en pantalla:
      * `estado`: QUIETO (esperando), MOVIENDO (la mano se mueve) o ESPERANDO (se movio y
        esta por cerrarse).
      * `motivo`: lo que dijo el analisis del ultimo frame cuando no se pudo usar.
    """

    def __init__(self):
        self.senas = 0        # seña cerradas y entregadas
        self.descartes = 0    # seña que empezaron a moverse pero no llegaron a nada usable
        self.reiniciar()

    # ------------------------------------------------------------------ estado interno
    def reiniciar(self):
        """Vacia el buffer (por ejemplo al encender la camara de nuevo)."""
        self._pose, self._izq, self._der, self._tiempos = [], [], [], []
        self._ancho = self._alto = 0
        self.en_sena = False
        self.estado = QUIETO
        self.motivo = ''
        self._t_inicio = None
        self._ultimo_mov = None
        self._candidato = None

    @property
    def frames(self):
        """Cuantos frames tiene el buffer ahora mismo."""
        return len(self._tiempos)

    def _podar(self):
        """Fuera de una seña solo se guardan los ultimos PRE_ROLL frames (calma previa).

        Asi el buffer no crece mientras nadie firma y, cuando empieza el movimiento, queda
        un poco de contexto antes igual que en un video grabado.
        """
        sobran = len(self._tiempos) - PRE_ROLL
        if sobran > 0:
            for lista in (self._pose, self._izq, self._der, self._tiempos):
                del lista[:sobran]

    def _candidato_forzado(self):
        """Ultimo recurso cuando la mano no se detuvo: clasificar lo que haya en el buffer.

        Se usa si la seña se paso de MAX_DURACION (o si el buffer se lleno). El vector sale
        del mismo analisis que en el flujo normal, pero la seña puede estar incompleta: eso
        queda marcado en info['cortada_al_final'].
        """
        datos, info = feat.analizar_movimiento(np.array(self._pose), np.array(self._izq),
                                               np.array(self._der), self._ancho, self._alto)
        if info['estado'] != feat.ESTADO_OK or info.get('frames_activos', 0) < MIN_FRAMES_ACTIVOS:
            return None
        bloque = info['bloque']
        return {'vector': feat.vector_de_tramo(datos, info), 'info': info,
                't_inicio': self._tiempos[info['primer_activo']], 't_fin': self._tiempos[info['ultimo_activo']]}

    def _cerrar(self, ahora, forzada=False):
        """Cierra la seña en curso. Devuelve el evento o None si no habia nada usable."""
        candidato, self._candidato = self._candidato, None
        self.en_sena = False
        self.estado = QUIETO
        self._t_inicio = None
        self._ultimo_mov = None
        if candidato is None and forzada:
            candidato = self._candidato_forzado()
        self._podar()
        if candidato is None:
            self.descartes += 1
            return None
        self.senas += 1
        info = candidato['info']
        t_fin = candidato['t_fin']
        return {'vector': candidato['vector'], 'info': info, 'forzada': bool(forzada),
                't_inicio': candidato['t_inicio'], 't_fin': t_fin, 't_emision': ahora,
                'latencia': ahora - t_fin, 'duracion': t_fin - candidato['t_inicio'],
                'frames': int(info['frames_activos'])}

    # ------------------------------------------------------------------ uso
    def agregar(self, pose, izq, der, ancho, alto, t=None):
        """Suma un frame al buffer y devuelve el evento de la seña terminada, o None.

        `t` es el instante del frame en segundos monontonos; por defecto se toma la hora
        actual. Se recibe de fuera para poder probar el corte con tiempos sinteticos.
        """
        ahora = time.monotonic() if t is None else t

        if self.en_sena and len(self._tiempos) >= MAX_FRAMES:
            return self._cerrar(ahora, forzada=True)
        if not self.en_sena:
            self._podar()
        self._pose.append(pose)
        self._izq.append(izq)
        self._der.append(der)
        self._tiempos.append(ahora)
        self._ancho, self._alto = ancho, alto

        datos, info = feat.analizar_movimiento(np.array(self._pose), np.array(self._izq),
                                               np.array(self._der), ancho, alto)
        estado = info['estado']
        self.motivo = info.get('motivo', '')

        if estado in (feat.ESTADO_OK, feat.ESTADO_CORTO):
            suficiente = info.get('frames_activos', 0) >= MIN_FRAMES_ACTIVOS
            if estado == feat.ESTADO_OK and not info.get('cortada_al_final') and suficiente:
                # La mano ya se detuvo dentro del buffer: el tramo esta completo y sirve como
                # candidato. Se vuelve a calcular solo mientras siga quieta (son pocos frames).
                self._candidato = {'vector': feat.vector_de_tramo(datos, info), 'info': info,
                                   't_inicio': self._tiempos[info['primer_activo']],
                                   't_fin': self._tiempos[info['ultimo_activo']]}
            else:
                self._candidato = None      # se sigue moviendo: el candidato anterior ya no vale
            if not self.en_sena:
                self._t_inicio = ahora
            self.en_sena = True
            # El cierre se mide desde el ultimo frame con movimiento de verdad, no desde el
            # final del bloque (que incluye MARGEN frames de calma).
            self._ultimo_mov = (self._tiempos[info['ultimo_activo']] if estado == feat.ESTADO_OK else ahora)
            self.estado = ESPERANDO if self._candidato is not None else MOVIENDO
        elif self.en_sena:
            self.estado = ESPERANDO
        else:
            self.estado = QUIETO

        if not self.en_sena:
            return None
        if self._t_inicio is not None and (ahora - self._t_inicio) >= MAX_DURACION:
            return self._cerrar(ahora, forzada=True)
        # La seña termina cuando pasa INACTIVIDAD_FIN desde el ultimo frame con movimiento
        # (o desde el final exacto del tramo, que es mas preciso cuando ya se conoce).
        referencia = self._candidato['t_fin'] if self._candidato is not None else self._ultimo_mov
        if referencia is not None and (ahora - referencia) >= INACTIVIDAD_FIN:
            return self._cerrar(ahora)
        return None


def reconocer(modelo, evento, rechazar=True):
    """Clasifica el evento del segmentador con el modelo dinamico ya entrenado.

    Devuelve un dict con el mismo contenido que la prueba con grabacion: ok, etiqueta,
    dist, umbral, rechazada, cortada; mas lo propio del modo en vivo: latencia (segundos
    desde que termino la seña), duracion, frames y forzada.
    """
    import lsp_dynamic_train as tl   # import local: asi este modulo sigue siendo numpy puro

    etiqueta, detalle = tl.predecir(modelo, evento['vector'], evento['info']['secuencia'])
    return {'ok': True, 'etiqueta': etiqueta, 'dist': detalle['dist_min'], 'umbral': detalle['umbral'],
            'rechazada': bool(rechazar and detalle['rechazada']),
            'cortada': bool(evento['info'].get('cortada_al_inicio')),
            'forzada': bool(evento['forzada']), 'latencia': float(evento['latencia']),
            'duracion': float(evento['duracion']), 'frames': int(evento['frames'])}
