import datetime
import os
import pickle
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMessageBox,
                              QPushButton, QSizePolicy, QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qt_video import mostrar_frame
from lsp_features import extraer_vector
from lsp_tts import Hablante

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONIDOS_DIR = os.path.join(ROOT_DIR, 'Sonidos')


class TestingWidget(QWidget):
    testing_requested = pyqtSignal()
    # Si mantienes la misma seña, se vuelve a decir cada estos segundos.
    REPETIR_ANUNCIO_SEGUNDOS = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_window_open = False
        self.testing_running = False
        self.modo_actual = 'letras'
        self.svm_letras = None
        self.svm_numeros = None
        self.anterior_prediccion = None
        self.ultimo_anuncio = 0.0
        pygame.mixer.init()
        self.hablante = Hablante()
        self.init_ui()
        self.load_models()
        self.init_mediapipe()

    def init_ui(self):
        layout = QVBoxLayout()

        first_row_layout = QHBoxLayout()
        first_row_layout.setContentsMargins(120, 0, 80, 0)

        self.label = QLabel("Reconocimiento de Señas LSP", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("georgia", 12, QFont.Bold))
        first_row_layout.addWidget(self.label)

        first_row_layout.addStretch()

        self.show_image_button = QPushButton('', self)
        self.show_image_button.setIcon(QIcon('images/image_button.png'))
        self.show_image_button.setIconSize(QSize(70, 50))
        self.show_image_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_image_button.setStyleSheet("QPushButton { border: 3px solid white; border-radius: 10px; background-color: white; color: black; } QPushButton:hover { border: 3px solid white; background-color: white; color: black; }")
        self.show_image_button.clicked.connect(self.opening)
        first_row_layout.addWidget(self.show_image_button)

        layout.addLayout(first_row_layout)
        layout.addSpacing(10)

        second_row_layout = QHBoxLayout()
        second_row_layout.setContentsMargins(70, 0, 20, 0)

        right_column_layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Crece con la ventana: al maximizar, el video ocupa todo el espacio disponible.
        self.video_label.setMinimumSize(480, 340)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        right_column_layout.addWidget(self.video_label)

        second_row_layout.addLayout(right_column_layout, 1)   # el video se lleva el espacio extra

        left_column_layout = QVBoxLayout()
        left_column_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.test_model_button = QPushButton('Probar Letras', self)
        self.test_model_button.setFixedSize(200, 50)
        self.test_model_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.test_model_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.test_model_button.clicked.connect(lambda: self.start_video('letras'))
        left_column_layout.addWidget(self.test_model_button)

        self.test_number_button = QPushButton('Probar Numeros', self)
        self.test_number_button.setFixedSize(200, 50)
        self.test_number_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.test_number_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.test_number_button.clicked.connect(lambda: self.start_video('numeros'))
        left_column_layout.addWidget(self.test_number_button)

        self.stop_test_button = QPushButton('Detener', self)
        self.stop_test_button.setFixedSize(200, 50)
        self.stop_test_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.stop_test_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.stop_test_button.clicked.connect(self.stop_video)
        left_column_layout.addWidget(self.stop_test_button)

        self.test_voice_button = QPushButton('Probar Voz', self)
        self.test_voice_button.setFixedSize(200, 50)
        self.test_voice_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.test_voice_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: lightgreen; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.test_voice_button.clicked.connect(self.probar_voz)
        left_column_layout.addWidget(self.test_voice_button)

        self.clock_label = QLabel(self)
        self.clock_label.setFont(QFont("Arial", 11))
        self.clock_label.setStyleSheet("border: 3px solid black; border-radius: 10px; background-color: white; color: black; padding: 5px;")
        self.clock_label.setAlignment(Qt.AlignCenter)
        left_column_layout.addWidget(self.clock_label)

        self.model_status_label = QLabel(self)
        self.model_status_label.setFont(QFont("Arial", 10))
        self.model_status_label.setWordWrap(True)
        # Ancho fijo: con wordWrap su sizeHint es el de la linea mas larga (~1100 px) y se quedaba
        # con el ancho de la ventana, dejando al video en su tamano minimo.
        self.model_status_label.setFixedWidth(260)
        self.model_status_label.setStyleSheet("border: 3px solid black; border-radius: 10px; background-color: #FFF3CD; color: black; padding: 5px;")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        left_column_layout.addWidget(self.model_status_label)

        self.calendar_label = QLabel(self)
        self.calendar_label.setFont(QFont("Arial", 11))
        self.calendar_label.setStyleSheet("border: 3px solid black; border-radius: 10px; background-color: white; color: black; padding: 5px;")
        self.calendar_label.setAlignment(Qt.AlignCenter)
        left_column_layout.addWidget(self.calendar_label)
        left_column_layout.addStretch(1)

        second_row_layout.addLayout(left_column_layout)
        layout.addLayout(second_row_layout)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.update_clock()

    def _cargar_modelo(self, nombre_archivo):
        ruta = os.path.join(ROOT_DIR, nombre_archivo)
        if not os.path.exists(ruta):
            return None
        try:
            with open(ruta, 'rb') as f:
                return pickle.load(f)['model']
        except Exception as e:
            print(f'No se pudo cargar {nombre_archivo}: {e}')
            return None

    def load_models(self):
        """Lee los modelos desde disco. Se puede llamar varias veces: si entrenas un
        modelo con la aplicacion abierta, aqui se vuelve a leer sin reiniciar."""
        self.svm_letras = self._cargar_modelo('svm_letras_lsp_classifier.p')
        self.svm_numeros = self._cargar_modelo('svm_numeros_lsp_classifier.p')
        self._actualizar_estado_modelos()

    def reload_models(self):
        self.load_models()

    def _describir_modelo(self, modelo):
        if modelo is None:
            return 'no entrenado'
        clases = getattr(modelo, 'classes_', None)
        if clases is None:
            return 'entrenado'
        etiquetas = ', '.join(str(c) for c in clases)
        return f'entrenado ({etiquetas})'

    def _actualizar_estado_modelos(self):
        self.model_status_label.setText(
            'Estado de modelos\n'
            f"Letras: {self._describir_modelo(self.svm_letras)}\n"
            f"Numeros: {self._describir_modelo(self.svm_numeros)}\n"
            f"Voz: {self.hablante.descripcion}"
        )

    def init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def start_video(self, modo):
        self.modo_actual = modo
        # Recarga desde disco: si entrenaste un modelo con la app abierta, se usa al instante.
        self.reload_models()
        modelo = self.svm_letras if modo == 'letras' else self.svm_numeros
        if modelo is None:
            archivo = 'svm_letras_lsp_classifier.p' if modo == 'letras' else 'svm_numeros_lsp_classifier.p'
            QMessageBox.warning(
                self, "Advertencia",
                f"No se encontro un modelo entrenado para {modo}.\n\n"
                f"Falta el archivo '{archivo}' en la carpeta del proyecto.\n"
                "Ve a 'Entrenamiento', elige el tipo correcto y pulsa 'Entrenar Modelo'.\n"
                "Recuerda que SVM necesita al menos 2 clases capturadas."
            )
            return

        self.anterior_prediccion = None
        self.ultimo_anuncio = 0.0
        self.testing_running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la camara.")
            return
        self.timer.start(30)

    def hide_elements(self):
        for widget in [self.label, self.show_image_button, self.test_model_button, self.test_number_button,
                       self.stop_test_button, self.test_voice_button]:
            widget.hide()

    def show_elements(self):
        self.reload_models()
        for widget in [self.label, self.show_image_button, self.test_model_button, self.test_number_button,
                       self.stop_test_button, self.test_voice_button]:
            widget.show()

    def show_test_model(self):
        self.testing_requested.emit()

    def opening(self):
        if not self.image_window_open:
            cv2.namedWindow("Alfabeto LSP", cv2.WINDOW_NORMAL)
            image = cv2.imread(os.path.join(ROOT_DIR, 'template.jpg'))
            if image is not None:
                cv2.imshow("Alfabeto LSP", image)
            cv2.moveWindow("Alfabeto LSP", 1500, 120)
            self.image_window_open = True
        else:
            cv2.destroyWindow("Alfabeto LSP")
            self.image_window_open = False

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock_label.setText(now.strftime("%H:%M:%S"))
        self.calendar_label.setText(now.strftime("%B %d, %Y"))
        QTimer.singleShot(1000, self.update_clock)

    def dibujar_texto(self, imagen, texto, posicion, tamano_fuente=60):
        imagen_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(imagen_pil)
        try:
            fuente = ImageFont.truetype('arial.ttf', tamano_fuente)
        except OSError:
            fuente = ImageFont.load_default()
        # Rectangulo negro detras del texto: asi la letra se lee sobre cualquier fondo.
        caja = draw.textbbox(posicion, texto, font=fuente)
        margen = 12
        draw.rectangle(
            [caja[0] - margen, caja[1] - margen, caja[2] + margen, caja[3] + margen],
            fill=(0, 0, 0),
        )
        # Texto en azul (RGB).
        draw.text(posicion, texto, font=fuente, fill=(0, 0, 255, 255))
        return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

    def _debe_anunciar(self, texto_predicho):
        """Decide si hay que anunciar la seña: cuando cambia, o cuando se mantiene
        la misma mas de REPETIR_ANUNCIO_SEGUNDOS sin haberse anunciado."""
        ahora = time.time()
        if texto_predicho != self.anterior_prediccion:
            self.anterior_prediccion = texto_predicho
            self.ultimo_anuncio = ahora
            return True
        if (ahora - self.ultimo_anuncio) >= self.REPETIR_ANUNCIO_SEGUNDOS:
            self.ultimo_anuncio = ahora
            return True
        return False

    def probar_voz(self):
        """Dice una frase de prueba para comprobar que el altavoz y la voz funcionan."""
        self.hablante.decir('Prueba de voz del reconocimiento de lengua de senas peruana')

    def _anunciar_prediccion(self, texto_predicho):
        """Dice la seña reconocida: usa el mp3 propio (Sonidos/<clase>.mp3) si existe;
        si no, la dice en voz alta con la voz del sistema (TTS)."""
        audio_path = os.path.join(SONIDOS_DIR, f'{texto_predicho}.mp3')
        if os.path.exists(audio_path):
            print(f'[Prueba] reproduciendo mp3: {audio_path}', flush=True)
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        else:
            print(f'[Prueba] sena reconocida: {texto_predicho} (se dice con TTS)', flush=True)
            self.hablante.decir(texto_predicho)

    def stop_video(self):
        self.testing_running = False
        self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def update_frame(self):
        if not self.testing_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Error", "No se pudo leer el frame de la camara.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        texto_predicho = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            vector = extraer_vector(results.multi_hand_landmarks, results.multi_handedness)
            modelo = self.svm_letras if self.modo_actual == 'letras' else self.svm_numeros

            if modelo is not None:
                prediccion = modelo.predict([vector])[0]
                texto_predicho = str(prediccion)

                if self._debe_anunciar(texto_predicho):
                    self._anunciar_prediccion(texto_predicho)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if texto_predicho is not None:
            frame_bgr = self.dibujar_texto(frame_bgr, texto_predicho, (20, 20), tamano_fuente=60)

        mostrar_frame(self.video_label, frame_bgr)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    testing_widget = TestingWidget()
    testing_widget.show()
    sys.exit(app.exec_())
