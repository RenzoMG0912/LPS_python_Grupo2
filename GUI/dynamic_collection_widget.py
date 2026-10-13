import os
import subprocess
import sys
import textwrap
import time
import warnings

import cv2
import mediapipe as mp
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                              QMessageBox, QPushButton, QSizePolicy, QSpacerItem,
                              QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lsp_dynamic as dyn

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mediapipe')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(ROOT_DIR, 'Videos_capturados')
VENTANA = 'Capturar Senas con Movimiento'

ESC, ESPACIO = 27, 32
AMARILLO, BLANCO, VERDE, ROJO, NARANJA = (0, 255, 255), (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 165, 255)

ESTILO_BOTON = ("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; "
                "color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")


class DynamicCollectionWidget(QWidget):
    """Graba senas con movimiento: un video corto por repeticion, con control de calidad.

    Flujo de cada muestra: espera (ESPACIO) -> cuenta regresiva -> grabacion -> analisis con
    MediaPipe -> se guarda si cumple los minimos de deteccion; si no, se descarta y se repite.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    # ------------------------------------------------------------------ interfaz
    def init_ui(self):
        layout = QVBoxLayout()

        fila = QHBoxLayout()
        self.label_clase = QLabel('Sena con movimiento', self)
        self.label_clase.setFont(QFont("Arial", 11, QFont.Bold))
        fila.addWidget(self.label_clase)

        self.combo_clase = QComboBox(self)
        self.combo_clase.setFont(QFont("Arial", 11, QFont.Bold))
        for i, clase in enumerate(dyn.CLASES_DINAMICAS):
            self.combo_clase.addItem(f'{clase.etiqueta}  ({clase.tipo.lower()})', i)  # se guarda el indice
        fila.addWidget(self.combo_clase)
        fila.addStretch()
        layout.addLayout(fila)
        layout.addSpacing(8)

        self.label_info = QLabel(' ', self)
        self.label_info.setFont(QFont("Arial", 10))
        self.label_info.setWordWrap(True)
        self.label_info.setFixedWidth(700)
        layout.addWidget(self.label_info)
        layout.addSpacing(8)

        self.label_rep = QLabel('Repeticiones a grabar en esta sesion ....', self)
        self.label_rep.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label_rep)

        self.entry_rep = QLineEdit(str(dyn.REPETICIONES_POR_DEFECTO), self)
        self.entry_rep.setFixedHeight(40)
        self.entry_rep.setFixedWidth(200)
        self.entry_rep.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry_rep.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry_rep)
        layout.addSpacing(8)

        self.label_persona = QLabel('ID de la persona que se graba (p01, p02, ...) ....', self)
        self.label_persona.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.label_persona)

        self.entry_persona = QLineEdit('p01', self)
        self.entry_persona.setFixedHeight(40)
        self.entry_persona.setFixedWidth(200)
        self.entry_persona.setFont(QFont("Arial", 11, QFont.Bold))
        self.entry_persona.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry_persona)
        layout.addSpacing(10)

        self.label_estado = QLabel(' ', self)
        self.label_estado.setFont(QFont("Arial", 12, QFont.Bold))
        self.label_estado.setFixedSize(700, 90)
        self.label_estado.setWordWrap(True)
        self.label_estado.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.label_estado.setAlignment(Qt.AlignCenter)
        self.label_estado.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        layout.addWidget(self.label_estado, alignment=Qt.AlignCenter)
        layout.addSpacing(20)

        botones = QHBoxLayout()
        self.iniciar_button = QPushButton('Iniciar Captura', self)
        self.iniciar_button.setFixedSize(220, 60)
        self.iniciar_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.iniciar_button.setStyleSheet(ESTILO_BOTON)
        self.iniciar_button.clicked.connect(self.iniciar_captura)
        botones.addWidget(self.iniciar_button)

        botones.addSpacerItem(QSpacerItem(25, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.carpeta_button = QPushButton('Abrir Carpeta', self)
        self.carpeta_button.setFixedSize(230, 60)
        self.carpeta_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.carpeta_button.setStyleSheet(ESTILO_BOTON)
        self.carpeta_button.clicked.connect(self.abrir_carpeta)
        botones.addWidget(self.carpeta_button)

        botones.setAlignment(Qt.AlignCenter)
        layout.addLayout(botones)
        self.setLayout(layout)

        self.combo_clase.currentIndexChanged.connect(self._actualizar_info)
        self.entry_persona.textChanged.connect(self._actualizar_info)
        self._actualizar_info()
        self.hide_elements()

    def _elementos(self):
        return [self.label_clase, self.combo_clase, self.label_info, self.label_rep, self.entry_rep,
                self.label_persona, self.entry_persona, self.label_estado, self.iniciar_button,
                self.carpeta_button]

    def hide_elements(self):
        for widget in self._elementos():
            widget.hide()

    def show_elements(self):
        for widget in self._elementos():
            widget.show()

    def _clase_actual(self):
        return dyn.CLASES_DINAMICAS[self.combo_clase.currentData()]

    def _actualizar_info(self):
        clase = self._clase_actual()
        persona = dyn.limpiar_id(self.entry_persona.text())
        total = dyn.contar_muestras(VIDEOS_DIR, clase)
        propias = dyn.contar_muestras(VIDEOS_DIR, clase, persona) if persona else 0
        texto = f'Ya hay {total} muestras de {clase.etiqueta} ({propias} de {persona or "?"}).'
        if clase.manos == 2:
            texto += ' Es una sena de dos manos.'
        if clase.nota:
            texto += f'  {clase.nota}.'
        self.label_info.setText(texto)

    def abrir_carpeta(self):
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        if sys.platform.startswith('win'):
            os.startfile(VIDEOS_DIR)
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', VIDEOS_DIR])

    # ------------------------------------------------------------------ captura
    def iniciar_captura(self):
        clase = self._clase_actual()
        persona = dyn.limpiar_id(self.entry_persona.text())
        if not persona:
            QMessageBox.warning(self, 'Falta el ID', 'Escribe el ID de la persona (por ejemplo p01). '
                                                      'Solo letras, numeros, guion y guion bajo.')
            return
        try:
            objetivo = int(self.entry_rep.text())
        except ValueError:
            objetivo = 0
        if objetivo <= 0:
            QMessageBox.warning(self, 'Repeticiones', 'Escribe un numero de repeticiones mayor que 0.')
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, 'Camara', 'No se pudo abrir la camara.')
            return
        if dyn.RESOLUCION_CAMARA:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, dyn.RESOLUCION_CAMARA[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dyn.RESOLUCION_CAMARA[1])

        # `pose` solo guia el encuadre mientras esperas; `holistic` revisa la calidad despues de grabar.
        # model_complexity=1 a proposito: el modelo de complejidad 0 NO viene dentro del paquete de
        # MediaPipe y se descarga de internet la primera vez (fallaria sin conexion).
        pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.label_estado.setText('Camara abierta. Sigue las instrucciones de la ventana de video.')
        self.label_estado.repaint()
        try:
            estado, guardadas = self._sesion(cap, clase, persona, objetivo, pose, holistic)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pose.close()
            holistic.close()

        self._mostrar_resultado(clase, persona, estado, guardadas)
        self._actualizar_info()

    def _mostrar_resultado(self, clase, persona, estado, guardadas):
        n = len(guardadas)
        if estado == 'error':
            texto = f'No se pudo leer la camara. Se guardaron {n} muestras de {clase.etiqueta} en esta sesion.'
        elif estado == 'error_guardado':
            texto = (f'No se pudo guardar el video: {getattr(self, "_error_guardado", "")}. '
                     f'Se guardaron {n} muestras antes del error.')
        elif estado == 'cancelado':
            texto = f'Captura terminada. {n} muestras de {clase.etiqueta} guardadas en esta sesion.'
        else:
            texto = f'Listo: {n} muestras de {clase.etiqueta} guardadas ({persona}).'
        self.label_estado.setText(texto)

    def _sesion(self, cap, clase, persona, objetivo, pose, holistic):
        """Graba repeticiones hasta llegar a `objetivo`. Devuelve (estado, rutas_guardadas).

        estado: 'completo', 'cancelado' (ESC), 'error' (camara) o 'error_guardado' (disco).
        Cada muestra se guarda en cuanto pasa el control de calidad, asi que cancelar
        conserva todo lo ya grabado. Con B se borra la ultima guardada de ESTA sesion.
        """
        guardadas = []   # (ruta_video, ruta_json)
        aviso = ''
        while len(guardadas) < objetivo:
            estado = self._esperar(cap, clase, persona, len(guardadas), objetivo, aviso, pose)
            if estado in ('cancelado', 'error'):
                return estado, guardadas
            if estado == 'borrar':
                if guardadas:
                    dyn.borrar_muestra(*guardadas.pop())
                    aviso = 'Ultima muestra borrada'
                else:
                    aviso = 'No hay muestras de esta sesion para borrar'
                continue

            estado = self._cuenta_regresiva(cap, clase)
            if estado != 'ok':
                return estado, guardadas
            frames, tiempos, estado = self._grabar(cap, clase)
            if estado != 'ok':
                return estado, guardadas
            if len(frames) < dyn.MIN_FRAMES_GRABADOS:
                aviso = f'DESCARTADA: la camara entrego solo {len(frames)} frames; va demasiado lenta'
                continue

            resumen = self._analizar(frames, holistic)
            aceptada, motivo, estadisticas = dyn.evaluar_calidad(resumen, clase.manos == 2)
            if not aceptada:
                aviso = 'DESCARTADA: ' + motivo
                continue

            fps = dyn.fps_real(tiempos)
            estadisticas['fps_medidos'] = round(fps, 2)
            try:
                rutas = dyn.guardar_muestra(VIDEOS_DIR, clase, persona, frames, fps, estadisticas)
            except OSError as error:
                self._error_guardado = str(error)
                return 'error_guardado', guardadas
            guardadas.append(rutas)
            aviso = f'Guardada {len(guardadas)}/{objetivo}'
        return 'completo', guardadas

    # ------------------------------------------------------------------ pantallas
    def _leer_tecla(self):
        return cv2.waitKey(1) & 0xFF

    def _leer_frame(self, cap):
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        alto, ancho = frame.shape[:2]
        if ancho > dyn.ANCHO_MAX:
            frame = cv2.resize(frame, (dyn.ANCHO_MAX, int(round(alto * dyn.ANCHO_MAX / ancho))),
                               interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _banda(frame, alto):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], alto), (0, 0, 0), -1)

    @staticmethod
    def _texto(frame, texto, posicion, escala, color, grosor=2):
        cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)

    def _dibujar_aviso(self, frame, aviso):
        """Mensaje del resultado de la ultima muestra, en la parte baja (partido en lineas)."""
        if not aviso:
            return
        color = ROJO if aviso.startswith(('DESCARTADA', 'No hay')) else VERDE
        lineas = textwrap.wrap(aviso, 62)
        y = frame.shape[0] - 14 - 28 * (len(lineas) - 1)
        for linea in lineas:
            self._texto(frame, linea, (14, y), 0.7, color)
            y += 28

    def _evaluar_encuadre(self, frame, pose):
        """Guia simple: se ven los dos hombros y ocupan un ancho razonable de la imagen."""
        resultado = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if resultado.pose_landmarks is None:
            return False, 'No se detecta el cuerpo'
        izq, der = resultado.pose_landmarks.landmark[11], resultado.pose_landmarks.landmark[12]
        dentro = all(0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for p in (izq, der))
        if min(izq.visibility, der.visibility) < 0.5 or not dentro:
            return False, 'No se ven los dos hombros: ajusta el encuadre'
        ancho = abs(izq.x - der.x)
        if ancho > dyn.ANCHO_HOMBROS_MAX:
            return False, 'Muy cerca: aleja la camara'
        if ancho < dyn.ANCHO_HOMBROS_MIN:
            return False, 'Muy lejos: acercate'
        return True, 'Encuadre OK'

    def _esperar(self, cap, clase, persona, hechas, objetivo, aviso, pose):
        """Vista previa sin grabar. Devuelve 'listo' (ESPACIO), 'borrar' (B), 'cancelado' (ESC) o 'error'."""
        titulo = f'{dyn.nombre_en_pantalla(clase)}   {hechas + 1}/{objetivo}   [{persona}]'
        while True:
            frame = self._leer_frame(cap)
            if frame is None:
                return 'error'
            encuadre_ok, texto_encuadre = self._evaluar_encuadre(frame, pose)

            self._banda(frame, 150 if clase.nota else 120)
            self._texto(frame, titulo, (14, 36), 1.0, AMARILLO, 2)
            self._texto(frame, 'ESPACIO = grabar    B = borrar la ultima    ESC = terminar', (14, 70), 0.62, BLANCO)
            self._texto(frame, texto_encuadre, (14, 102), 0.7, VERDE if encuadre_ok else NARANJA)
            if clase.nota:
                self._texto(frame, clase.nota, (14, 136), 0.58, AMARILLO, 1)
            self._dibujar_aviso(frame, aviso)
            cv2.imshow(VENTANA, frame)

            tecla = self._leer_tecla()
            if tecla == ESC:
                return 'cancelado'
            if tecla == ESPACIO:
                return 'listo'
            if tecla in (ord('b'), ord('B')):
                return 'borrar'

    def _cuenta_regresiva(self, cap, clase):
        """Cuenta antes de grabar, para que puedas dejar el teclado y ponerte en posicion. No graba."""
        if dyn.CUENTA_REGRESIVA <= 0:
            return 'ok'
        fin = time.time() + dyn.CUENTA_REGRESIVA
        while True:
            restante = fin - time.time()
            if restante <= 0:
                return 'ok'
            frame = self._leer_frame(cap)
            if frame is None:
                return 'error'
            alto, ancho = frame.shape[:2]
            self._banda(frame, 80)
            self._texto(frame, dyn.nombre_en_pantalla(clase), (14, 36), 1.0, AMARILLO, 2)
            self._texto(frame, 'Manos en reposo. Haz la sena cuando aparezca REC.    ESC = cancelar',
                        (14, 66), 0.6, BLANCO)
            numero = str(int(restante) + 1)
            (w, h), _ = cv2.getTextSize(numero, cv2.FONT_HERSHEY_SIMPLEX, 8, 16)
            self._texto(frame, numero, ((ancho - w) // 2, (alto + h) // 2), 8, AMARILLO, 16)
            cv2.imshow(VENTANA, frame)
            if self._leer_tecla() == ESC:
                return 'cancelado'

    def _grabar(self, cap, clase):
        """Graba DURACION_GRABACION segundos. Se guardan los frames SIN dibujos encima.

        Devuelve (frames, hora_de_cada_frame, estado). No se procesa con MediaPipe mientras se graba:
        hacerlo ralentiza la camara y el video saldria con menos cuadros por segundo.
        """
        frames, tiempos = [], []
        inicio = time.time()
        while True:
            frame = self._leer_frame(cap)
            if frame is None:
                return None, None, 'error'
            ahora = time.time()
            frames.append(frame)
            tiempos.append(ahora)
            transcurrido = ahora - inicio

            vista = frame.copy()
            ancho = vista.shape[1]
            self._banda(vista, 60)
            cv2.circle(vista, (30, 30), 12, ROJO, -1)
            self._texto(vista, f'REC  {dyn.nombre_en_pantalla(clase)}', (54, 40), 0.9, BLANCO, 2)
            avance = min(1.0, transcurrido / dyn.DURACION_GRABACION)
            cv2.rectangle(vista, (0, 52), (int(ancho * avance), 60), ROJO, -1)
            cv2.imshow(VENTANA, vista)

            if self._leer_tecla() == ESC:
                return None, None, 'cancelado'
            if transcurrido >= dyn.DURACION_GRABACION:
                return frames, tiempos, 'ok'

    def _analizar(self, frames, holistic):
        """Revisa con MediaPipe Holistic que se vean los hombros y las manos. Un dict por frame."""
        resumen = []
        total = len(frames)
        for i, frame in enumerate(frames):
            r = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            resumen.append({
                'pose': r.pose_landmarks is not None,
                'manos': int(r.left_hand_landmarks is not None) + int(r.right_hand_landmarks is not None),
            })
            if i % 8 == 0:   # mantiene viva la ventana y muestra el avance
                vista = frames[-1].copy()
                self._banda(vista, 60)
                self._texto(vista, f'Analizando la muestra...  {i + 1}/{total}', (14, 40), 0.9, AMARILLO, 2)
                cv2.imshow(VENTANA, vista)
                self._leer_tecla()
        return resumen
