import glob
import os
import pickle
import sys
import time
import warnings

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QCursor, QFont, QIcon
from PyQt5.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                              QMessageBox, QPushButton, QSizePolicy, QSpacerItem,
                              QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsp_features import extraer_vector
from lsp_labels_dict import LETRAS_LSP, NUMEROS_LSP

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mediapipe')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'Mis_Datos')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset.pickle')

# Parametros de la captura de vectores en vivo.
VENTANA_CAPTURA = 'Capturar Senas (MediaPipe)'
ANCHO_VENTANA, ALTO_VENTANA = 960, 720
INTERVALO_CAPTURA = 0.08  # segundos minimos entre dos capturas (~12 por segundo)
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
CUENTA_REGRESIVA = 3  # segundos de cuenta regresiva tras pulsar ESPACIO (0 = sin cuenta)


class DataCollectionWidget(QWidget):
    data_collection_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.number_of_classes = 0
        self.dataset_size = 0
        self.image_window_open = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        horizontal_layout = QHBoxLayout()

        self.label_tipo = QLabel('Tipo de seña', self)
        self.label_tipo.setFont(QFont("Arial", 11, QFont.Bold))
        horizontal_layout.addWidget(self.label_tipo)

        self.combo_tipo = QComboBox(self)
        self.combo_tipo.addItem('Letras (A-Z, Ñ)')
        self.combo_tipo.addItem('Numeros (0-10)')
        horizontal_layout.addWidget(self.combo_tipo)

        horizontal_layout.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.label_clase = QLabel('Clase a capturar', self)
        self.label_clase.setFont(QFont("Arial", 11, QFont.Bold))
        horizontal_layout.addWidget(self.label_clase)

        self.combo_clase = QComboBox(self)
        self.combo_clase.setFont(QFont("Arial", 11, QFont.Bold))
        horizontal_layout.addWidget(self.combo_clase)

        horizontal_layout.addStretch()

        self.show_image_button = QPushButton('', self)
        self.show_image_button.setIcon(QIcon('images/image_button.png'))
        self.show_image_button.setIconSize(QSize(60, 45))
        self.show_image_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_image_button.setStyleSheet("QPushButton { border: 3px solid white; border-radius: 10px; background-color: white; color: black; } QPushButton:hover { border: 3px solid white; background-color: white; color: black; }")
        self.show_image_button.clicked.connect(self.opening)
        horizontal_layout.addWidget(self.show_image_button)

        layout.addLayout(horizontal_layout)
        layout.addSpacing(8)

        self.entry1 = QLineEdit(self)
        self.entry1.setFixedHeight(40)
        self.entry1.setFixedWidth(200)
        self.entry1.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry1.setAlignment(Qt.AlignCenter)

        self.label1 = QLabel('Numero de clases a capturar ....', self)
        self.label1.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label1)
        layout.addWidget(self.entry1)

        layout.addSpacing(8)

        self.label2 = QLabel('Numero de capturas (vectores) por clase ....', self)
        self.label2.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label2)

        self.entry2 = QLineEdit(self)
        self.entry2.setFixedHeight(40)
        self.entry2.setFixedWidth(200)
        self.entry2.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry2.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry2)

        layout.addSpacing(8)

        self.label3 = QLabel('Nombre de la clase (Modificar Datos)', self)
        self.label3.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label3)

        self.entry3 = QLineEdit(self)
        self.entry3.setFixedHeight(40)
        self.entry3.setFixedWidth(200)
        self.entry3.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry3.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry3)

        layout.addSpacing(10)

        self.label4 = QLabel(' ', self)
        self.label4.setFont(QFont("Arial", 12, QFont.Bold))
        self.label4.setFixedSize(600, 70)
        self.label4.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        layout.addWidget(self.label4, alignment=Qt.AlignCenter)

        layout.addSpacing(20)

        buttons_layout = QHBoxLayout()
        self.open_camera_button = QPushButton('Iniciar Camara', self)
        self.open_camera_button.setFixedSize(220, 60)
        self.open_camera_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.open_camera_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.open_camera_button.clicked.connect(self.open_data_collection)
        buttons_layout.addWidget(self.open_camera_button)

        buttons_layout.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.data_modification_button = QPushButton('Modificar Datos', self)
        self.data_modification_button.setFixedSize(230, 60)
        self.data_modification_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.data_modification_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.data_modification_button.clicked.connect(self.open_collect_images_in_specific_folder)
        buttons_layout.addWidget(self.data_modification_button)

        buttons_layout.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.directory_button = QPushButton('Abrir Carpeta', self)
        self.directory_button.setFixedSize(230, 60)
        self.directory_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.directory_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.directory_button.clicked.connect(self.open_data_directory)
        buttons_layout.addWidget(self.directory_button)

        buttons_layout.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.create_dataset_button = QPushButton('Crear Dataset', self)
        self.create_dataset_button.setIcon(QIcon('images/Create_Dataset.png'))
        self.create_dataset_button.setIconSize(QSize(70, 50))
        self.create_dataset_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.create_dataset_button.setStyleSheet("QPushButton { border: 3px solid darkblue; border-radius: 10px; background-color: lightblue; color: black; } QPushButton:hover { border: 3px solid darkorange; background-color: #CCCCCC; color: black; }")
        self.create_dataset_button.clicked.connect(self.create_dataset)
        buttons_layout.addWidget(self.create_dataset_button)

        buttons_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

        self._actualizar_combo_clase()
        self.combo_tipo.currentIndexChanged.connect(self._actualizar_combo_clase)
        self.hide_elements()

    def hide_elements(self):
        for widget in [self.label_tipo, self.combo_tipo, self.label_clase, self.combo_clase, self.label1, self.entry1,
                       self.label2, self.entry2, self.label3, self.entry3, self.label4, self.show_image_button,
                       self.open_camera_button, self.data_modification_button, self.directory_button,
                       self.create_dataset_button]:
            widget.hide()

    def show_elements(self):
        for widget in [self.label_tipo, self.combo_tipo, self.label_clase, self.combo_clase, self.label1, self.entry1,
                       self.label2, self.entry2, self.label3, self.entry3, self.label4, self.show_image_button,
                       self.open_camera_button, self.data_modification_button, self.directory_button,
                       self.create_dataset_button]:
            widget.show()

    def show_data_collection(self):
        self.data_collection_requested.emit()

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

    def open_data_directory(self):
        try:
            subprocess_open(ROOT_DIR)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir la carpeta: {e}")

    def open_dataset_directory(self):
        dataset_full_path = os.path.join(ROOT_DIR, 'dataset.pickle')
        if os.path.exists(dataset_full_path):
            try:
                subprocess_open(ROOT_DIR)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo abrir la carpeta: {e}")
        else:
            QMessageBox.critical(self, "Error", "El archivo dataset.pickle no existe.")

    def update_number_of_classes(self):
        try:
            self.number_of_classes = int(self.entry1.text())
        except ValueError:
            self.number_of_classes = 0

    def update_dataset_size(self):
        try:
            self.dataset_size = int(self.entry2.text())
        except ValueError:
            self.dataset_size = 0

    def etiquetas_activas(self):
        return LETRAS_LSP if self.combo_tipo.currentIndex() == 0 else NUMEROS_LSP

    def _actualizar_combo_clase(self):
        """Rellena el selector con las etiquetas del tipo activo (letras o numeros),
        conservando la seleccion previa si sigue existiendo."""
        seleccion = self.combo_clase.currentText()
        self.combo_clase.blockSignals(True)
        self.combo_clase.clear()
        for etiqueta in self.etiquetas_activas().values():
            self.combo_clase.addItem(etiqueta)
        indice = self.combo_clase.findText(seleccion)
        if indice >= 0:
            self.combo_clase.setCurrentIndex(indice)
        self.combo_clase.blockSignals(False)

    # ------------------------------------------------------------------
    # Manejo del dataset en disco
    # ------------------------------------------------------------------
    def _cargar_dataset(self):
        """Devuelve (data, labels) del dataset.pickle existente, o listas vacias."""
        if os.path.exists(DATASET_PATH):
            try:
                with open(DATASET_PATH, 'rb') as f:
                    contenido = pickle.load(f)
                data = list(contenido.get('data', []))
                labels = [str(etiqueta) for etiqueta in contenido.get('labels', [])]
                return data, labels
            except Exception as e:
                print(f'No se pudo leer dataset.pickle: {e}')
        return [], []

    def _guardar_dataset(self, data, labels):
        with open(DATASET_PATH, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)

    def _fusionar_dataset(self, nuevos_vectores, nuevas_etiquetas):
        """Agrega los vectores capturados al dataset existente y guarda.
        Devuelve el total de muestras acumuladas."""
        data, labels = self._cargar_dataset()
        data.extend(nuevos_vectores)
        labels.extend(nuevas_etiquetas)
        self._guardar_dataset(data, labels)
        return len(data)

    # ------------------------------------------------------------------
    # Captura en vivo de vectores de landmarks (MediaPipe)
    # ------------------------------------------------------------------
    def _crear_manos(self, estatico=False):
        return mp.solutions.hands.Hands(
            static_image_mode=estatico,
            max_num_hands=2,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

    def _leer_frame(self, cap, manos):
        """Lee un frame, lo procesa con MediaPipe y dibuja los landmarks encima.

        Devuelve (frame, resultados), o (None, None) si fallo la lectura de la camara.
        """
        ret, frame = cap.read()
        if not ret:
            return None, None

        frame = cv2.resize(frame, (ANCHO_VENTANA, ALTO_VENTANA))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = manos.process(frame_rgb)

        if resultados.multi_hand_landmarks:
            dibujo = mp.solutions.drawing_utils
            estilos = mp.solutions.drawing_styles
            for mano in resultados.multi_hand_landmarks:
                dibujo.draw_landmarks(
                    frame, mano, mp.solutions.hands.HAND_CONNECTIONS,
                    estilos.get_default_hand_landmarks_style(),
                    estilos.get_default_hand_connections_style(),
                )
        return frame, resultados

    def _esperar_inicio(self, cap, manos, class_label, progreso='', aviso_previo=''):
        """Pantalla de espera previa a grabar una clase. NO captura ningun vector.

        Muestra la camara en vivo para que el usuario coloque la mano. ESPACIO
        inicia (tras la cuenta regresiva, si CUENTA_REGRESIVA > 0) y ESC cancela.

        Devuelve 'ok', 'cancelado' o 'error'.
        """
        titulo = f'Clase {class_label}' + (f' ({progreso})' if progreso else '')

        while True:
            frame, _ = self._leer_frame(cap, manos)
            if frame is None:
                return 'error'

            self._dibujar_espera(frame, titulo, 'ESPACIO = empezar    ESC = cancelar', aviso_previo)
            cv2.imshow(VENTANA_CAPTURA, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                return 'cancelado'
            if tecla == 32:
                break

        if CUENTA_REGRESIVA > 0:
            fin_cuenta = time.time() + CUENTA_REGRESIVA
            while True:
                restante = fin_cuenta - time.time()
                if restante <= 0:
                    break

                frame, _ = self._leer_frame(cap, manos)
                if frame is None:
                    return 'error'

                self._dibujar_espera(frame, titulo, 'Coloca la mano en posicion    ESC = cancelar',
                                     cuenta=int(restante) + 1)
                cv2.imshow(VENTANA_CAPTURA, frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    return 'cancelado'

        return 'ok'

    def _dibujar_espera(self, frame, titulo, ayuda, aviso_previo='', cuenta=None):
        alto, ancho = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (ancho, 120), (0, 0, 0), -1)
        cv2.putText(frame, titulo, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, ayuda, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if aviso_previo:
            cv2.putText(frame, aviso_previo, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if cuenta is not None:
            texto = str(cuenta)
            (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 8, 16)
            cv2.putText(frame, texto, ((ancho - ancho_texto) // 2, (alto + alto_texto) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 255), 16, cv2.LINE_AA)

    def _capturar_clase(self, cap, manos, class_label, objetivo, progreso='', aviso_previo=''):
        """Captura en vivo vectores de landmarks para una clase.
        """
        estado_espera = self._esperar_inicio(cap, manos, class_label, progreso, aviso_previo)
        if estado_espera != 'ok':
            return [], estado_espera

        vectores = []
        ultimo_capturado = 0.0

        while True:
            frame, resultados = self._leer_frame(cap, manos)
            if frame is None:
                return vectores, 'error'

            mano_detectada = bool(resultados.multi_hand_landmarks)

            if mano_detectada:
                ahora = time.time()
                if len(vectores) < objetivo and (ahora - ultimo_capturado) >= INTERVALO_CAPTURA:
                    vectores.append(extraer_vector(resultados.multi_hand_landmarks, resultados.multi_handedness))
                    ultimo_capturado = ahora

            self._dibujar_overlay(frame, class_label, len(vectores), objetivo, mano_detectada)
            cv2.imshow(VENTANA_CAPTURA, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                return vectores, 'cancelado'
            if tecla in (ord('q'), ord('Q')):
                return vectores, 'ok'
            if len(vectores) >= objetivo:
                return vectores, 'ok'

    def _dibujar_overlay(self, frame, class_label, capturadas, objetivo, mano_detectada):
        ancho = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (ancho, 120), (0, 0, 0), -1)
        cv2.putText(frame, f'Clase {class_label}: {capturadas}/{objetivo}', (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Q = terminar y guardar    ESC = cancelar', (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if not mano_detectada:
            cv2.putText(frame, 'Muestra la mano a la camara...', (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    def _abrir_ventana_captura(self):
        cv2.namedWindow(VENTANA_CAPTURA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VENTANA_CAPTURA, ANCHO_VENTANA, ALTO_VENTANA)
        cv2.moveWindow(VENTANA_CAPTURA, 600, 10)

    # ------------------------------------------------------------------
    # Boton "Iniciar Camara": captura varias clases y actualiza el dataset
    # ------------------------------------------------------------------
    def open_data_collection(self):
        self.entry3.setReadOnly(True)
        self.update_number_of_classes()
        self.update_dataset_size()

        os.makedirs(DATA_DIR, exist_ok=True)

        if self.number_of_classes <= 0:
            QMessageBox.warning(self, "Advertencia", "Ingresa un numero de clases valido.")
            self.entry3.setReadOnly(False)
            return

        if self.dataset_size <= 0:
            QMessageBox.warning(self, "Advertencia", "Ingresa un numero de capturas por clase valido.")
            self.entry3.setReadOnly(False)
            return

        lista_etiquetas = list(self.etiquetas_activas().values())
        inicio = self.combo_clase.currentIndex()
        if inicio < 0:
            inicio = 0
        clase_inicial = lista_etiquetas[inicio]

        if self.number_of_classes > len(lista_etiquetas) - inicio:
            QMessageBox.warning(
                self, "Advertencia",
                f"Empezando en la clase '{clase_inicial}' solo quedan "
                f"{len(lista_etiquetas) - inicio} etiquetas.\n\n"
                "Reduce el 'Numero de clases' o elige otra 'Clase a capturar'."
            )
            self.entry3.setReadOnly(False)
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la camara.")
            self.entry3.setReadOnly(False)
            return

        self._abrir_ventana_captura()
        manos = self._crear_manos(estatico=False)

        guardadas = []      # [(etiqueta, n_capturados), ...]
        total_nuevo = 0
        estado_final = 'ok'

        try:
            for j in range(inicio, inicio + self.number_of_classes):
                class_label = lista_etiquetas[j]
                posicion = j - inicio + 1
                self.label4.setText(f'Capturando clase {class_label} ({posicion}/{self.number_of_classes})')
                self.label4.repaint()

                aviso_previo = (f'{guardadas[-1][0]}: {guardadas[-1][1]} muestras guardadas'
                                if guardadas else '')
                vectores, estado = self._capturar_clase(
                    cap, manos, class_label, self.dataset_size,
                    progreso=f'{posicion}/{self.number_of_classes}', aviso_previo=aviso_previo)
                if estado == 'error':
                    estado_final = 'error'
                    break
                if estado == 'cancelado':
                    estado_final = 'cancelado'
                    break

                # Se guarda de inmediato: si luego se cancela, lo capturado no se pierde.
                if vectores:
                    self._fusionar_dataset(vectores, [class_label] * len(vectores))
                    total_nuevo += len(vectores)
                guardadas.append((class_label, len(vectores)))
        finally:
            manos.close()
            cap.release()
            cv2.destroyAllWindows()
            self.entry3.setReadOnly(False)

        if estado_final == 'error':
            self.label4.setText('Recoleccion interrumpida por un error de camara.')
            QMessageBox.critical(self, "Error", "Fallo al leer la camara.")
            return

        if total_nuevo == 0 and estado_final == 'cancelado':
            self.label4.setText('Recoleccion cancelada. El dataset no cambio.')
            return

        if total_nuevo == 0:
            self.label4.setText('No se capturo ninguna muestra. El dataset no cambio.')
            QMessageBox.warning(self, "Advertencia", "No se detectaron manos. No se guardo nada.")
            return

        _, labels_actuales = self._cargar_dataset()
        resumen = ', '.join(f'{etiqueta}: {n}' for etiqueta, n in guardadas)
        sin_capturas = [etiqueta for etiqueta, n in guardadas if n == 0]
        aviso = ''
        if sin_capturas:
            aviso = (f"\n\nATENCION: no se capturo nada para {', '.join(sin_capturas)}. "
                     "Repite esa clase (o usa 'Modificar Datos') para que puedas entrenar.")
        if estado_final == 'cancelado':
            self.label4.setText(f'Recoleccion cancelada. Dataset: {len(labels_actuales)} muestras.')
            QMessageBox.information(
                self, "Recoleccion cancelada",
                f"Se cancelo la recoleccion; las clases ya capturadas se conservaron.\n"
                f"Guardado en esta sesion: {resumen}\n\n"
                f"Total acumulado: {len(labels_actuales)} muestras."
            )
        else:
            self.label4.setText(f'Dataset actualizado: {len(labels_actuales)} muestras totales.')
            QMessageBox.information(
                self, "Exito",
                f"Se agregaron {total_nuevo} vectores al dataset.\n"
                f"{resumen}\n\n"
                f"Total acumulado: {len(labels_actuales)} muestras."
                f"{aviso}"
            )

    # ------------------------------------------------------------------
    # Boton "Modificar Datos": agrega vectores a la clase ya existente
    # ------------------------------------------------------------------
    def open_collect_images_in_specific_folder(self):
        self.entry1.setReadOnly(True)
        self.update_dataset_size()

        # Si se escribio un nombre personalizado se usa ese; si no, la clase elegida en el selector.
        folder_name = self.entry3.text().strip() or self.combo_clase.currentText().strip()

        if not folder_name:
            QMessageBox.warning(self, "Advertencia", "Ingresa o selecciona un nombre de clase valido.")
            self.entry1.setReadOnly(False)
            return

        if self.dataset_size <= 0:
            QMessageBox.warning(self, "Advertencia", "Ingresa un numero de capturas por clase valido.")
            self.entry1.setReadOnly(False)
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la camara.")
            self.entry1.setReadOnly(False)
            return

        self._abrir_ventana_captura()
        manos = self._crear_manos(estatico=False)

        self.label4.setText(f'Modificando clase {folder_name}')
        self.label4.repaint()

        try:
            vectores, estado = self._capturar_clase(cap, manos, folder_name, self.dataset_size)
        finally:
            manos.close()
            cap.release()
            cv2.destroyAllWindows()
            self.entry1.setReadOnly(False)

        if estado == 'error':
            self.label4.setText('Modificacion interrumpida por un error de camara.')
            QMessageBox.critical(self, "Error", "Fallo al leer la camara.")
            return

        if estado == 'cancelado':
            self.label4.setText('Modificacion cancelada. No se modifico el dataset.')
            return

        if not vectores:
            self.label4.setText('No se capturo ninguna muestra.')
            QMessageBox.warning(self, "Advertencia", "No se detectaron manos. No se guardo nada.")
            return

        total = self._fusionar_dataset(vectores, [folder_name] * len(vectores))
        self.label4.setText(f'Dataset actualizado: {total} muestras totales.')
        QMessageBox.information(
            self, "Exito",
            f"Se agregaron {len(vectores)} vectores a la clase '{folder_name}'.\n\n"
            f"Total acumulado: {total} muestras."
        )

    # ------------------------------------------------------------------
    # Compatibilidad: construir dataset a partir de imagenes en Mis_Datos
    # ------------------------------------------------------------------
    def process_image(self, img_path, hands):
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f'Error al leer la imagen: {img_path}')
            print(e)
            return None

        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)

        results = hands.process(img_np)
        if not results.multi_hand_landmarks:
            return None

        return extraer_vector(results.multi_hand_landmarks, results.multi_handedness)

    def create_dataset(self):
        image_files = glob.glob(os.path.join(DATA_DIR, '**/*.jpg'), recursive=True)
        if not image_files:
            QMessageBox.information(
                self, "Sin imagenes",
                "No hay imagenes en Mis_Datos.\n\n"
                "La camara ahora vectoriza las senas en vivo y las guarda directamente en "
                "dataset.pickle, por lo que no hace falta crear el dataset desde imagenes. "
                "Se conserva el dataset actual."
            )
            return

        manos = self._crear_manos(estatico=True)
        data = []
        labels = []
        try:
            for img_path in image_files:
                result = self.process_image(img_path, manos)
                if result:
                    data.append(result)
                    labels.append(os.path.basename(os.path.dirname(img_path)))
        finally:
            manos.close()

        if not data:
            QMessageBox.warning(self, "Advertencia",
                                "No se detectaron manos en las imagenes. Se conserva el dataset actual.")
            return

        total = self._fusionar_dataset(data, labels)
        QMessageBox.information(None, "Exito",
                                f"Se agregaron {len(data)} vectores de imagenes al dataset.\n\n"
                                f"Total acumulado: {total} muestras.")
        self.open_dataset_directory()


def subprocess_open(path):
    import platform
    import subprocess

    sistema = platform.system()
    if sistema == 'Windows':
        subprocess.Popen(f'explorer {os.path.realpath(path)}')
    elif sistema == 'Darwin':
        subprocess.Popen(['open', os.path.realpath(path)])
    else:
        subprocess.Popen(['xdg-open', os.path.realpath(path)])