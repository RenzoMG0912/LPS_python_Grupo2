"""Muestra un cuadro de OpenCV dentro de un QLabel, ajustandolo al tamano de la etiqueta.

Lo usan las dos pantallas que muestran la camara (señas estaticas y prueba en vivo). La
etiqueta crece con la ventana, asi que al maximizar el programa el video ocupa el espacio
disponible: se conserva la proporcion de la imagen y lo que sobra queda en negro.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


def mostrar_frame(label, frame_bgr):
    """Dibuja un cuadro BGR (numpy) en `label`, escalado a su tamano actual.

    Si el cuadro no tiene datos, o la etiqueta aun no tiene tamano, no hace nada.
    """
    if frame_bgr is None:
        return
    alto, ancho = frame_bgr.shape[:2]
    if alto <= 0 or ancho <= 0 or label.width() <= 0 or label.height() <= 0:
        return
    # .copy() para que el QImage sea dueño de sus datos: si no, la imagen depende del numpy
    # que la origino y puede quedar con basura cuando ese arreglo se libera.
    imagen = QImage(frame_bgr.data, ancho, alto, frame_bgr.strides[0], QImage.Format_BGR888).copy()
    pixmap = QPixmap.fromImage(imagen).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(pixmap)
