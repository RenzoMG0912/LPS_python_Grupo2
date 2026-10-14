"""Prueba en vivo de senas con movimiento: se reconoce al terminar la sena, sin grabar video.

La diferencia con GUI/dynamic_testing_widget.py (que se conserva como respaldo) es el momento
en que se analiza. Aqui la camara y MediaPipe Holistic corren en un hilo aparte y alimentan
frame a frame a lsp_dynamic_live.Segmentador, que corta la sena en cuanto la mano se queda
quieta y la clasifica con el mismo modelo ya entrenado. No hay cuenta regresiva, no se escribe
ningun video y la interfaz no se congela: el resultado aparece ~0,5 s despues de la sena.

El modelo y el dataset son exactamente los mismos que los del flujo con grabacion: no hay que
reentrenar nada.
"""
import os
import sys
import time
import warnings

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel, QMessageBox,
                              QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extraer_secuencias as ext
import lsp_dynamic as dyn
import lsp_dynamic_live as env
import lsp_dynamic_train as tl
from dynamic_camera import AMARILLO, BLANCO, NARANJA, VERDE
from dynamic_testing_widget import NOMBRES_MODELO

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mediapipe')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONIDOS_DIR = os.path.join(ROOT_DIR, 'Sonidos')

ANCHO_VISTA = 800            # ancho al que se reduce el cuadro antes de mostrarlo
REPETIR_ANUNCIO_SEGUNDOS = 3.0   # si repites la misma sena, se vuelve a decir cada estos segundos

ESTILO_BOTON = ("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; "
                "color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; } "
                "QPushButton:disabled { background-color: #DDDDDD; color: #888888; border-color: #888888; }")

# Texto del estado del segmentador, en pantalla (sin tildes: se dibuja con OpenCV).
TEXTO_ESTADO = {env.QUIETO: ('Haz la sena cuando quieras', VERDE),
                env.MOVIENDO: ('Sena en curso...', AMARILLO),
                env.ESPERANDO: ('Cerrando la sena...', NARANJA)}


def _anotar(frame, resultado, segmentador, fps):
    """Copia del frame con el esqueleto y el estado dibujados encima.

    Se dibuja despues de reducir la imagen, para que el grosor de las lineas no dependa de la
    resolucion que entrego la camara.
    """
    alto, ancho = frame.shape[:2]
    if ancho > ANCHO_VISTA:
        vista = cv2.resize(frame, (ANCHO_VISTA, int(round(alto * ANCHO_VISTA / ancho))), interpolation=cv2.INTER_AREA)
    else:
        vista = frame.copy()

    dibujo = mp.solutions.drawing_utils
    if resultado.pose_landmarks:
        dibujo.draw_landmarks(vista, resultado.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=dibujo.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                              connection_drawing_spec=dibujo.DrawingSpec(color=(0, 120, 0), thickness=1))
    for mano in (resultado.left_hand_landmarks, resultado.right_hand_landmarks):
        if mano:
            dibujo.draw_landmarks(vista, mano, mp.solutions.hands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=dibujo.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=2),
                                  connection_drawing_spec=dibujo.DrawingSpec(color=(0, 200, 255), thickness=2))

    cv2.rectangle(vista, (0, 0), (vista.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(vista, 'EN VIVO   %.1f FPS' % fps, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLANCO, 2, cv2.LINE_AA)
    texto, color = TEXTO_ESTADO.get(segmentador.estado, ('', BLANCO))
    cv2.putText(vista, texto, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return np.ascontiguousarray(vista)


class _HiloCamara(QThread):
    """Camara + MediaPipe Holistic + segmentador, fuera del hilo de la interfaz.

    Emite cada cuadro para mostrarlo, y los eventos de reconocimiento cuando el segmentador
    cierra una sena.
    """
    cuadro = pyqtSignal(object)
    reconocida = pyqtSignal(object)
    estado = pyqtSignal(object)
    terminado = pyqtSignal(str)

    def __init__(self, modelo, rechazar, parent=None):
        super().__init__(parent)
        self._modelo = modelo
        self._rechazar = rechazar      # funcion sin argumentos: lee la casilla en el momento
        self._parar = False

    def detener(self):
        """Pide al bucle que termine (surte efecto en el siguiente frame)."""
        self._parar = True

    def run(self):
        motivo_final = ''
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                motivo_final = 'No se pudo abrir la camara.'
            else:
                motivo_final = self._bucle(cap)
        except Exception as error:   # cualquier fallo se muestra en pantalla en vez de morir en silencio
            motivo_final = f'Error inesperado: {type(error).__name__}: {error}'
        finally:
            cap.release()
        self.terminado.emit(motivo_final)

    def _bucle(self, cap):
        """Bucle de la camara. Devuelve el motivo por el que termino (vacio si fue normal)."""
        if dyn.RESOLUCION_CAMARA:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, dyn.RESOLUCION_CAMARA[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dyn.RESOLUCION_CAMARA[1])

        segmentador = env.Segmentador()
        motivo_final = ''
        # Un Holistic para toda la sesion. Mantener el modelo cargado entre frames es lo que
        # hace posible el reconocimiento en vivo: crearlo por seña costaba ~4,7 s.
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5) as holistic:
            t_previo, fps, i = None, 0.0, 0
            while not self._parar:
                frame = dyn.leer_frame(cap)
                if frame is None:
                    motivo_final = 'Se perdio la lectura de la camara.'
                    break
                t = time.monotonic()
                if t_previo is not None and t > t_previo:
                    instantaneo = 1.0 / (t - t_previo)
                    fps = instantaneo if fps == 0.0 else 0.9 * fps + 0.1 * instantaneo
                t_previo = t

                puntos, resultado = ext.landmarks_de_un_frame(frame, holistic)
                evento = segmentador.agregar(puntos['pose'], puntos['izq'], puntos['der'],
                                             puntos['ancho'], puntos['alto'], t)
                if evento is not None:
                    try:
                        respuesta = env.reconocer(self._modelo, evento, self._rechazar())
                    except Exception as error:   # un modelo raro no puede tumbar la camara
                        respuesta = {'ok': False, 'motivo': f'{type(error).__name__}: {error}'}
                    self.reconocida.emit(respuesta)

                self.cuadro.emit(_anotar(frame, resultado, segmentador, fps))
                i += 1
                if i % 4 == 0:      # el estado no necesita refrescarse a cada frame
                    self.estado.emit({'fps': fps, 'estado': segmentador.estado,
                                      'motivo': segmentador.motivo, 'senas': segmentador.senas,
                                      'descartes': segmentador.descartes, 'frames': segmentador.frames})
        return motivo_final


class DynamicLiveWidget(QWidget):
    """Pestana de prueba en vivo (sin grabar): la sena se reconoce en cuanto termina."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hilo = None
        self._hablante_obj = None
        self._rechazar = True
        self._anterior = None
        self._ultimo_anuncio = 0.0
        self.init_ui()
        self.actualizar_info()
        app = QApplication.instance()
        if app is not None:                       # red de seguridad si se cierra la aplicacion
            app.aboutToQuit.connect(self.detener)

    # ------------------------------------------------------------------ interfaz
    def init_ui(self):
        layout = QVBoxLayout()

        self.label_modelo = QLabel(' ', self)
        self.label_modelo.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_modelo.setWordWrap(True)
        layout.addWidget(self.label_modelo)

        self.info = QLabel('Reconoce la sena en cuanto termina, sin grabar video ni esperar. '
                           'Haz la sena y baja la mano: la etiqueta aparece sola.', self)
        self.info.setFont(QFont("Arial", 10))
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        fila = QHBoxLayout()

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(ANCHO_VISTA, 470)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        fila.addWidget(self.video_label)

        columna = QVBoxLayout()

        self.iniciar_button = QPushButton('Iniciar Prueba en Vivo', self)
        self.iniciar_button.setFixedSize(240, 55)
        self.iniciar_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.iniciar_button.setStyleSheet(ESTILO_BOTON)
        self.iniciar_button.clicked.connect(self.iniciar)
        columna.addWidget(self.iniciar_button)

        self.detener_button = QPushButton('Detener', self)
        self.detener_button.setFixedSize(240, 55)
        self.detener_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.detener_button.setStyleSheet(ESTILO_BOTON)
        self.detener_button.clicked.connect(self.detener)
        self.detener_button.setEnabled(False)
        columna.addWidget(self.detener_button)

        self.voz_button = QPushButton('Probar Voz', self)
        self.voz_button.setFixedSize(240, 55)
        self.voz_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.voz_button.setStyleSheet(ESTILO_BOTON)
        self.voz_button.clicked.connect(self.probar_voz)
        columna.addWidget(self.voz_button)

        self.label_resultado = QLabel(' ', self)
        self.label_resultado.setFont(QFont("Arial", 22, QFont.Bold))
        self.label_resultado.setAlignment(Qt.AlignCenter)
        self.label_resultado.setFixedSize(240, 70)
        columna.addWidget(self.label_resultado)

        self.label_estado = QLabel(' ', self)
        self.label_estado.setFont(QFont("Arial", 9))
        self.label_estado.setWordWrap(True)
        self.label_estado.setFixedWidth(240)
        columna.addWidget(self.label_estado)
        columna.addStretch()
        fila.addLayout(columna)
        layout.addLayout(fila)

        layout.addWidget(self.check_rechazo_linea())
        layout.addWidget(self.check_voz_linea())

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        self.log.setFixedHeight(120)
        self.log.setPlaceholderText('Aqui aparece cada sena reconocida y cuanto tardo.')
        layout.addWidget(self.log)

        self.setLayout(layout)

    def check_rechazo_linea(self):
        self.check_rechazo = QCheckBox('Decir "no reconocida" cuando la sena no se parece a ninguna grabada', self)
        self.check_rechazo.setFont(QFont("Arial", 10))
        self.check_rechazo.setChecked(True)
        # Se copia a un atributo simple porque el hilo de la camara lo lee mientras corre.
        self.check_rechazo.stateChanged.connect(lambda: setattr(self, '_rechazar', self.check_rechazo.isChecked()))
        return self.check_rechazo

    def check_voz_linea(self):
        self.check_voz = QCheckBox('Decir la sena en voz alta', self)
        self.check_voz.setFont(QFont("Arial", 10))
        self.check_voz.setChecked(True)
        self.label_voz = QLabel('Voz: se activa la primera vez que se usa', self)
        self.label_voz.setFont(QFont("Arial", 9))
        fila = QVBoxLayout()
        fila.addWidget(self.check_voz)
        fila.addWidget(self.label_voz)
        contenedor = QWidget(self)
        contenedor.setLayout(fila)
        return contenedor

    def actualizar_info(self):
        """Muestra que modelo hay cargado. Se puede llamar varias veces (al entrar a la pestana)."""
        try:
            m = tl.cargar_modelo(tl.RUTA_MODELO)
        except tl.DatasetNoValido as error:
            self.label_modelo.setText(f'Modelo: {error} Entrenalo en la pestana Entrenamiento.')
            return
        optimista = ' (una sola persona: optimista)' if m.get('estrategia') == 'pliegues' else ''
        self.label_modelo.setText(
            f'Modelo: {NOMBRES_MODELO.get(m["tipo"], m["tipo"])} | clases: {", ".join(m["clases"])} | '
            f'exactitud al validar: {m.get("exactitud_validacion", 0) * 100:.0f}%{optimista} | entrenado: {m.get("creado", "?")}')

    # ------------------------------------------------------------------ voz
    def _hablante(self):
        if self._hablante_obj is None:
            from lsp_tts import Hablante
            self._hablante_obj = Hablante()
        self.label_voz.setText(f'Voz: {self._hablante_obj.descripcion}')
        return self._hablante_obj

    def probar_voz(self):
        self._hablante().decir('Prueba de voz del reconocimiento de lengua de senas peruana')

    def _anunciar(self, etiqueta):
        """Usa el mp3 propio (Sonidos/<sena>.mp3) si existe; si no, la voz del sistema.

        Es la misma politica que en la prueba con grabacion.
        """
        mp3 = os.path.join(SONIDOS_DIR, f'{etiqueta}.mp3')
        if os.path.exists(mp3):
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(mp3)
                pygame.mixer.music.play()
                return
            except Exception as error:
                print(f'[En vivo] no se pudo reproducir {mp3}: {error}', flush=True)
        self._hablante().decir(dyn.texto_para_voz(etiqueta))

    def _debe_anunciar(self, etiqueta):
        """Dice la sena cuando cambia o cuando repites la misma despues de un rato."""
        ahora = time.time()
        if etiqueta != self._anterior:
            self._anterior = etiqueta
            self._ultimo_anuncio = ahora
            return True
        if (ahora - self._ultimo_anuncio) >= REPETIR_ANUNCIO_SEGUNDOS:
            self._ultimo_anuncio = ahora
            return True
        return False

    # ------------------------------------------------------------------ sesion
    def _ocupado(self, ocupado):
        self.iniciar_button.setEnabled(not ocupado)
        self.detener_button.setEnabled(ocupado)

    def iniciar(self):
        if self._hilo is not None and self._hilo.isRunning():
            return
        try:
            modelo = tl.cargar_modelo(tl.RUTA_MODELO)
        except tl.DatasetNoValido as error:
            QMessageBox.warning(self, 'Modelo', f'{error}\nEntrenalo en la pestana Entrenamiento.')
            return

        self.log.clear()
        self._anterior, self._ultimo_anuncio = None, 0.0
        self._hilo = _HiloCamara(modelo, lambda: self._rechazar, self)
        self._hilo.cuadro.connect(self._pintar)
        self._hilo.reconocida.connect(self._recibio)
        self._hilo.estado.connect(self._actualizar_estado)
        self._hilo.terminado.connect(self._termino)
        self._ocupado(True)
        self._registrar('Prueba en vivo iniciada. Haz la sena y baja la mano; no hay que esperar cuenta regresiva.')
        self._hilo.start()

    def detener(self):
        """Para el hilo de la camara y espera a que termine."""
        if self._hilo is None:
            return
        if self._hilo.isRunning():
            self._hilo.detener()
            if not self._hilo.wait(5000):
                self._registrar('El hilo de la camara no respondio; se cierra de todos modos.')
        self._hilo = None
        self._ocupado(False)

    def _termino(self, motivo):
        self._ocupado(False)
        if motivo:
            self._registrar(motivo)
            QMessageBox.warning(self, 'Camara', motivo)

    def _pintar(self, frame):
        alto, ancho = frame.shape[:2]
        imagen = QImage(frame.data, ancho, alto, 3 * ancho, QImage.Format_BGR888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(imagen))

    def _actualizar_estado(self, datos):
        texto, _ = TEXTO_ESTADO.get(datos['estado'], ('', None))
        extra = f" | {datos['motivo']}" if datos['motivo'] and datos['estado'] == env.QUIETO else ''
        self.label_estado.setText(f"{datos['fps']:.1f} FPS | {datos['senas']} senas | "
                                  f"{datos['descartes']} descartadas | {texto}{extra}")

    def _registrar(self, texto):
        self.log.appendPlainText(f'{time.strftime("%H:%M:%S")}  {texto}')
        self.log.repaint()

    def _recibio(self, r):
        """Una sena cerrada y clasificada: se muestra, se registra y (si toca) se dice."""
        if not r['ok']:
            self._registrar('no se pudo analizar: ' + str(r.get('motivo', '')))
            return
        nombre = dyn.etiqueta_en_pantalla(r['etiqueta'])
        hay_dist = r['dist'] is not None and r['umbral'] is not None
        detalle = f'  distancia {r["dist"]:.2f} (limite {r["umbral"]:.2f}).' if hay_dist else ''
        latencia = f'  reconocida {r["latencia"]:.2f} s despues de terminar la sena' \
                   f' ({r["frames"]} frames, {r["duracion"]:.2f} s de movimiento).'
        if r['rechazada']:
            self.label_resultado.setText('NO RECONOCIDA')
            self._registrar(f'no reconocida (se parecia a {r["etiqueta"]}).{detalle}{latencia}')
        else:
            self.label_resultado.setText(nombre)
            self._registrar(f'{r["etiqueta"]}.{detalle}{latencia}')
            if self.check_voz.isChecked() and self._debe_anunciar(r['etiqueta']):
                self._anunciar(r['etiqueta'])
        if r['forzada']:
            self._registrar('  ojo: la mano no se detuvo; la sena se corto por tiempo.')
        if r['cortada']:
            self._registrar('  ojo: puede faltar el inicio de la sena; empiezala antes.')
