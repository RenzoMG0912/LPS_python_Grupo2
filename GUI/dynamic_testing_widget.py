import os
import sys
import time
import warnings

import cv2
import mediapipe as mp
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                              QPushButton, QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lsp_dynamic as dyn
import lsp_dynamic_features as feat
import lsp_dynamic_train as tl
from dynamic_camera import NARANJA, ROJO, VERDE, PantallasCamara

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mediapipe')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONIDOS_DIR = os.path.join(ROOT_DIR, 'Sonidos')

ESTILO_BOTON = ("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; "
                "color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
NOMBRES_MODELO = {'rbf': 'SVM sobre el vector fijo (RBF)', 'dtw': 'SVM con kernel DTW'}


class DynamicTestingWidget(PantallasCamara, QWidget):
    """Prueba en vivo: grabas una sena, el programa dice cual es (o que no la reconoce)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hablante_obj = None      # la voz se crea la primera vez que se usa
        self.init_ui()

    # ------------------------------------------------------------------ interfaz
    def init_ui(self):
        layout = QVBoxLayout()

        self.label_modelo = QLabel(' ', self)
        self.label_modelo.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_modelo.setWordWrap(True)
        layout.addWidget(self.label_modelo)

        self.check_rechazo = QCheckBox('Decir "no reconocida" cuando la sena no se parece a ninguna grabada', self)
        self.check_rechazo.setFont(QFont("Arial", 10))
        self.check_rechazo.setChecked(True)
        layout.addWidget(self.check_rechazo)

        self.check_voz = QCheckBox('Decir la sena en voz alta', self)
        self.check_voz.setFont(QFont("Arial", 10))
        self.check_voz.setChecked(True)
        layout.addWidget(self.check_voz)

        self.label_voz = QLabel('Voz: se activa la primera vez que se usa', self)
        self.label_voz.setFont(QFont("Arial", 9))
        layout.addWidget(self.label_voz)

        fila = QHBoxLayout()
        self.iniciar_button = QPushButton('Iniciar Prueba', self)
        self.iniciar_button.setFixedSize(220, 60)
        self.iniciar_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.iniciar_button.setStyleSheet(ESTILO_BOTON)
        self.iniciar_button.clicked.connect(self.iniciar_prueba)
        fila.addWidget(self.iniciar_button)

        self.voz_button = QPushButton('Probar Voz', self)
        self.voz_button.setFixedSize(180, 60)
        self.voz_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.voz_button.setStyleSheet(ESTILO_BOTON)
        self.voz_button.clicked.connect(self.probar_voz)
        fila.addWidget(self.voz_button)
        fila.addStretch()
        layout.addLayout(fila)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        self.log.setMinimumHeight(170)
        self.log.setPlaceholderText('Aqui aparece lo que se va reconociendo.')
        layout.addWidget(self.log)

        self.setLayout(layout)
        self.actualizar_info()

    def actualizar_info(self):
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
        """Usa el mp3 propio (Sonidos/<sena>.mp3) si existe; si no, la voz del sistema."""
        mp3 = os.path.join(SONIDOS_DIR, f'{etiqueta}.mp3')
        if os.path.exists(mp3):
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(mp3)
                pygame.mixer.music.play()
                return
            except Exception as error:     # sin pygame o sin audio: se cae a la voz del sistema
                print(f'[Prueba] no se pudo reproducir {mp3}: {error}', flush=True)
        self._hablante().decir(dyn.texto_para_voz(etiqueta))

    def _registrar(self, texto):
        self.log.appendPlainText(f'{time.strftime("%H:%M:%S")}  {texto}')
        self.log.repaint()

    # ------------------------------------------------------------------ prueba
    def iniciar_prueba(self):
        try:
            modelo = tl.cargar_modelo(tl.RUTA_MODELO)
        except tl.DatasetNoValido as error:
            QMessageBox.warning(self, 'Modelo', f'{error}\nEntrenalo en la pestana Entrenamiento.')
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, 'Camara', 'No se pudo abrir la camara.')
            return
        if dyn.RESOLUCION_CAMARA:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, dyn.RESOLUCION_CAMARA[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dyn.RESOLUCION_CAMARA[1])
        pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._registrar('Prueba iniciada. Sigue las instrucciones de la ventana de video.')
        try:
            estado = self._sesion_prueba(cap, modelo, pose)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pose.close()
        if estado == 'error':
            QMessageBox.warning(self, 'Camara', 'Se perdio la lectura de la camara.')
        self._registrar('Prueba terminada.')

    def _sesion_prueba(self, cap, modelo, pose):
        """Repite: espera -> cuenta -> graba -> reconoce. Termina con ESC. Devuelve 'cancelado' o 'error'."""
        resultado, aviso = None, ''
        while True:
            estado = self._pantalla_espera(cap, 'PRUEBA', 'ESPACIO = grabar y reconocer    ESC = salir',
                                           '', aviso, pose, resultado=resultado)
            if estado != 'listo':
                return estado
            estado = self._pantalla_cuenta(cap, 'PRUEBA', 'Manos en reposo. Haz la sena cuando aparezca REC.    ESC = cancelar')
            if estado != 'ok':
                return estado
            frames, _, estado = self._pantalla_grabacion(cap, 'REC  PRUEBA')
            if estado != 'ok':
                return estado
            if len(frames) < dyn.MIN_FRAMES_GRABADOS:
                resultado, aviso = ('SIN RESULTADO', ROJO), f'DESCARTADA: la camara entrego solo {len(frames)} frames; va muy lenta'
                continue

            r = self._reconocer(frames, modelo, self.check_rechazo.isChecked())
            if not r['ok']:
                resultado, aviso = ('SIN RESULTADO', ROJO), 'DESCARTADA: ' + r['motivo']
                self._registrar(f'no se pudo analizar: {r["motivo"]}')
                continue

            nombre = dyn.etiqueta_en_pantalla(r['etiqueta'])
            hay_dist = r['dist'] is not None and r['umbral'] is not None      # un modelo antiguo puede no traer el umbral
            detalle = f' Distancia {r["dist"]:.2f} (limite {r["umbral"]:.2f}).' if hay_dist else ''
            if r['rechazada']:
                resultado = ('NO RECONOCIDA', NARANJA)
                aviso = f'DESCARTADA: no se parece a ninguna sena grabada. Parecia {nombre}.{detalle}'
                self._registrar(f'no reconocida (parecia {r["etiqueta"]}; distancia {r["dist"]:.2f} > {r["umbral"]:.2f})')
            else:
                resultado = (nombre, VERDE)
                aviso = f'Reconocida.{detalle}'
                self._registrar(f'{r["etiqueta"]}' + (f'  (distancia {r["dist"]:.2f}, limite {r["umbral"]:.2f})' if hay_dist else ''))
                if self.check_voz.isChecked():
                    self._anunciar(r['etiqueta'])
            if r['cortada']:
                aviso += ' Ojo: la sena parece cortada; empieza antes.'

    def _reconocer(self, frames, modelo, rechazar):
        """De frames grabados a una respuesta. Devuelve un dict: ok, etiqueta, rechazada, dist, umbral, cortada, motivo."""
        raw = self._procesar_frames(frames, 'Reconociendo...')
        vector, info = feat.extraer_caracteristicas(raw['pose'], raw['izq'], raw['der'], raw['ancho'], raw['alto'])
        if vector is None:
            return {'ok': False, 'motivo': info['motivo']}
        etiqueta, detalle = tl.predecir(modelo, vector, info['secuencia'])
        return {'ok': True, 'etiqueta': etiqueta, 'dist': detalle['dist_min'], 'umbral': detalle['umbral'],
                'rechazada': bool(rechazar and detalle['rechazada']),
                'cortada': bool(info.get('cortada_al_final') or info.get('cortada_al_inicio'))}
