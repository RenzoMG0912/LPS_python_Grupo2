import os
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamic_collection_widget import DynamicCollectionWidget
from dynamic_live_widget import DynamicLiveWidget
from dynamic_testing_widget import DynamicTestingWidget
from dynamic_training_widget import DynamicTrainingWidget


class DynamicHubWidget(QWidget):
    """Pantalla "Senas con Movimiento": agrupa la captura, el entrenamiento y las dos pruebas.

    Sigue el mismo contrato que las demas pantallas de Main.py (show_elements / hide_elements).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        capa = QVBoxLayout(self)
        capa.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget(self)
        self.tabs.setFont(QFont("Arial", 11, QFont.Bold))
        self.captura = DynamicCollectionWidget(self, embebido=True)
        self.entrenamiento = DynamicTrainingWidget(self)
        self.en_vivo = DynamicLiveWidget(self)
        self.prueba = DynamicTestingWidget(self)
        self.tabs.addTab(self.captura, 'Captura')
        self.tabs.addTab(self.entrenamiento, 'Entrenamiento')
        self.tabs.addTab(self.en_vivo, 'Prueba en vivo')
        self.tabs.addTab(self.prueba, 'Prueba con grabacion')
        self.tabs.currentChanged.connect(self._actualizar)
        capa.addWidget(self.tabs)

        self.hide_elements()

    def _actualizar(self, *_):
        """Al entrar o cambiar de pestana se refrescan los conteos (los videos pueden haber cambiado)."""
        self.captura._actualizar_info()
        self.entrenamiento.actualizar_info()
        self.en_vivo.actualizar_info()
        self.prueba.actualizar_info()
        # La camara en vivo tiene que soltarse al salir de su pestana o al ocultar la pantalla.
        if self.tabs.currentWidget() is not self.en_vivo:
            self.en_vivo.detener()

    def hide_elements(self):
        self.en_vivo.detener()
        self.tabs.hide()

    def show_elements(self):
        self.tabs.show()
        self._actualizar()
