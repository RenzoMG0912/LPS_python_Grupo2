import os
import subprocess
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMessageBox,
                              QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extraer_secuencias as ext
import lsp_dynamic_train as tl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(ROOT_DIR, 'Videos_capturados')
CACHE_DIR = os.path.join(ROOT_DIR, 'Landmarks_extraidos')

ESTILO_BOTON = ("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; "
                "color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; } "
                "QPushButton:disabled { background-color: #DDDDDD; color: #888888; border-color: #888888; }")

MODELOS = [('Automatico (el que valide mejor)', 'auto'),
           ('SVM sobre el vector fijo (RBF)', 'rbf'),
           ('SVM con kernel DTW', 'dtw')]


class _Hilo(QThread):
    """Corre una tarea larga fuera del hilo de la interfaz, para que la ventana no se congele."""
    linea = pyqtSignal(str)
    terminado = pyqtSignal(object)

    def __init__(self, tarea, parent=None):
        super().__init__(parent)
        self._tarea = tarea

    def run(self):
        try:
            self.terminado.emit(self._tarea(self.linea.emit))
        except tl.DatasetNoValido as error:
            self.terminado.emit({'ok': False, 'motivo': str(error)})
        except Exception as error:     # cualquier otro fallo se muestra en pantalla en vez de cerrar el programa
            self.terminado.emit({'ok': False, 'motivo': f'Error inesperado: {type(error).__name__}: {error}'})


class DynamicTrainingWidget(QWidget):
    """Convierte los videos grabados en dataset y entrena el modelo de senas con movimiento."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hilo = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label_dataset = QLabel(' ', self)
        self.label_dataset.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_dataset.setWordWrap(True)
        layout.addWidget(self.label_dataset)

        fila = QHBoxLayout()
        self.procesar_button = QPushButton('1. Procesar videos', self)
        self.procesar_button.setFixedSize(220, 55)
        self.procesar_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.procesar_button.setStyleSheet(ESTILO_BOTON)
        self.procesar_button.clicked.connect(self.procesar_videos)
        fila.addWidget(self.procesar_button)

        self.check_forzar = QCheckBox('Volver a analizar todos los videos', self)
        self.check_forzar.setFont(QFont("Arial", 10))
        self.check_forzar.setToolTip('Ignora los puntos ya guardados. Hace falta si cambiaste algun parametro; es lento.')
        fila.addWidget(self.check_forzar)
        fila.addStretch()
        layout.addLayout(fila)

        fila2 = QHBoxLayout()
        self.combo_modelo = QComboBox(self)
        self.combo_modelo.setFont(QFont("Arial", 11, QFont.Bold))
        self.combo_modelo.setFixedHeight(40)
        for texto, valor in MODELOS:
            self.combo_modelo.addItem(texto, valor)
        fila2.addWidget(self.combo_modelo)

        self.entrenar_button = QPushButton('2. Entrenar y evaluar', self)
        self.entrenar_button.setFixedSize(240, 55)
        self.entrenar_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.entrenar_button.setStyleSheet(ESTILO_BOTON)
        self.entrenar_button.clicked.connect(self.entrenar)
        fila2.addWidget(self.entrenar_button)
        fila2.addStretch()
        layout.addLayout(fila2)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMinimumHeight(190)
        layout.addWidget(self.log)

        fila3 = QHBoxLayout()
        self.matriz_button = QPushButton('Ver matriz de confusion', self)
        self.matriz_button.setFixedSize(260, 50)
        self.matriz_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.matriz_button.setStyleSheet(ESTILO_BOTON)
        self.matriz_button.clicked.connect(lambda: self._abrir(tl.RUTA_MATRIZ))
        fila3.addWidget(self.matriz_button)

        self.reporte_button = QPushButton('Ver reporte', self)
        self.reporte_button.setFixedSize(200, 50)
        self.reporte_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.reporte_button.setStyleSheet(ESTILO_BOTON)
        self.reporte_button.clicked.connect(lambda: self._abrir(tl.RUTA_REPORTE))
        fila3.addWidget(self.reporte_button)
        fila3.addStretch()
        layout.addLayout(fila3)

        self.setLayout(layout)
        self.actualizar_info()

    # ------------------------------------------------------------------ utilidades
    def actualizar_info(self):
        """Resume el dataset que hay en disco (clases, muestras, personas) o explica por que no se puede usar."""
        try:
            d = tl.cargar_dataset(tl.RUTA_DATASET)
        except tl.DatasetNoValido as error:
            self.label_dataset.setText(f'Dataset: {error}')
            return
        etiquetas = list(d['labels'])
        clases = ', '.join(f'{c} {etiquetas.count(c)}' for c in sorted(set(etiquetas)))
        self.label_dataset.setText(f'Dataset: {len(etiquetas)} muestras de {len(set(etiquetas))} clases '
                                   f'({clases}) | personas: {", ".join(sorted(set(d["personas"])))}')

    def _abrir(self, ruta):
        if not os.path.exists(ruta):
            QMessageBox.information(self, 'Todavia no existe', 'Ese archivo se crea al entrenar.')
            return
        if sys.platform.startswith('win'):
            os.startfile(ruta)
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', ruta])

    def _ocupado(self, ocupado):
        for b in (self.procesar_button, self.entrenar_button, self.check_forzar, self.combo_modelo):
            b.setEnabled(not ocupado)

    def _lanzar(self, tarea, al_terminar):
        if self._hilo is not None and self._hilo.isRunning():
            return
        self.log.clear()
        self._ocupado(True)
        self._al_terminar = al_terminar
        self._hilo = _Hilo(tarea, self)
        # Se conectan metodos del widget (y no lambdas) para que Qt los ejecute en el hilo de la interfaz.
        self._hilo.linea.connect(self._escribir)
        self._hilo.terminado.connect(self._cerrar_tarea)
        self._hilo.start()

    def _cerrar_tarea(self, resultado):
        self._ocupado(False)
        self._al_terminar(resultado)

    def _escribir(self, texto):
        self.log.appendPlainText(texto)

    # ------------------------------------------------------------------ acciones
    def procesar_videos(self):
        forzar = self.check_forzar.isChecked()
        self._lanzar(lambda log: ext.procesar(VIDEOS_DIR, CACHE_DIR, tl.RUTA_DATASET, forzar, log),
                     self._termino_procesar)

    def _termino_procesar(self, r):
        if r['ok']:
            self._escribir('\n' + r['texto'])
            self._escribir(f'\nDataset guardado: {r["aceptadas"]} muestras aceptadas, {r["rechazadas"]} rechazadas.')
        else:
            self._escribir('\n' + r['motivo'])
        self.actualizar_info()

    def entrenar(self):
        preferencia = self.combo_modelo.currentData()
        self._lanzar(lambda log: {'ok': True, **tl.entrenar(tl.RUTA_DATASET, tl.RUTA_MODELO, preferencia, log,
                                                              tl.RUTA_MATRIZ, tl.RUTA_REPORTE)},
                     self._termino_entrenar)

    def _termino_entrenar(self, r):
        if r['ok']:
            self._escribir('\n' + r['resumen'])
            self._escribir(f'\nModelo guardado en {os.path.basename(r["ruta_modelo"])}.')
        else:
            self._escribir('\nNo se pudo entrenar: ' + r['motivo'])
        self.actualizar_info()
