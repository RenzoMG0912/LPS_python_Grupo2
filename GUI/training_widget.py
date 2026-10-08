import os
import pickle
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyQt5.QtCore import QSize, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QCursor, QDesktopServices, QFont, QIcon
from PyQt5.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget,
                              QListWidgetItem, QMessageBox, QPushButton, QSizePolicy,
                              QSpacerItem, QVBoxLayout, QWidget)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsp_labels_dict import LETRAS_LSP, NUMEROS_LSP

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrainingWidget(QWidget):
    training_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.first_click = True
        self.image_window_open = False
        self.y_val = None
        self.y_predict = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        horizontal_layout1 = QHBoxLayout()

        self.label = QLabel("Entrenamiento del Modelo", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 12, QFont.Bold))
        horizontal_layout1.addWidget(self.label)

        self.combo_tipo = QComboBox(self)
        self.combo_tipo.addItem('Letras (A-Z, Ñ)')
        self.combo_tipo.addItem('Numeros (0-10)')
        self.combo_tipo.currentIndexChanged.connect(self.cargar_clases_disponibles)
        horizontal_layout1.addWidget(self.combo_tipo)

        horizontal_layout1.addStretch()

        self.show_image_button = QPushButton('', self)
        self.show_image_button.setIcon(QIcon('images/image_button.png'))
        self.show_image_button.setIconSize(QSize(70, 50))
        self.show_image_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_image_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.show_image_button.setStyleSheet("QPushButton { border: 3px solid white; border-radius: 10px; background-color: white; color: black; } QPushButton:hover { border: 3px solid white; background-color: white; color: black; }")
        self.show_image_button.clicked.connect(self.opening)
        horizontal_layout1.addWidget(self.show_image_button)

        layout.addLayout(horizontal_layout1)
        layout.addSpacing(20)

        horizontal_layout2 = QHBoxLayout()

        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setWordWrap(True)
        self.result_label.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.result_label.setMinimumHeight(200)
        self.result_label.setMinimumWidth(280)
        self.result_label.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        horizontal_layout2.addWidget(self.result_label)

        horizontal_layout2.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.metrics_label = QLabel('', self)
        self.metrics_label.setAlignment(Qt.AlignLeft)
        self.metrics_label.setFont(QFont("Arial", 12))
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.metrics_label.setMinimumHeight(200)
        self.metrics_label.setMinimumWidth(280)
        self.metrics_label.setStyleSheet("border: 1px solid black; background-color: white; padding: 5px;")
        horizontal_layout2.addWidget(self.metrics_label)

        layout.addLayout(horizontal_layout2)
        layout.addSpacing(20)

        buttons_layout = QHBoxLayout()

        self.open_dataset_file_button = QPushButton('Archivo Dataset', self)
        self.open_dataset_file_button.setFixedSize(180, 60)
        self.open_dataset_file_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.open_dataset_file_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.open_dataset_file_button.clicked.connect(self.open_dataset_directory)
        buttons_layout.addWidget(self.open_dataset_file_button)

        buttons_layout.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.train_button = QPushButton("Entrenar Modelo", self)
        self.train_button.setFixedSize(180, 60)
        self.train_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.train_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.train_button.clicked.connect(self.train_model)
        buttons_layout.addWidget(self.train_button)

        buttons_layout.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.classification_report_button = QPushButton("Reporte de Clasificacion", self)
        self.classification_report_button.setFixedSize(260, 60)
        self.classification_report_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.classification_report_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.classification_report_button.clicked.connect(self.open_classification_report)
        buttons_layout.addWidget(self.classification_report_button)

        buttons_layout.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

        self.confusion_matrix_button = QPushButton("Matriz de Confusion", self)
        self.confusion_matrix_button.setFixedSize(260, 60)
        self.confusion_matrix_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.confusion_matrix_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.confusion_matrix_button.clicked.connect(self.show_confusion_matrix)
        buttons_layout.addWidget(self.confusion_matrix_button)

        buttons_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(buttons_layout)

        clases_layout = QHBoxLayout()

        self.classes_label = QLabel('Clases a incluir en el entrenamiento:', self)
        self.classes_label.setFont(QFont("Arial", 11, QFont.Bold))
        clases_layout.addWidget(self.classes_label)

        self.lista_clases = QListWidget(self)
        self.lista_clases.setFixedSize(360, 110)
        self.lista_clases.setFont(QFont("Arial", 11))
        clases_layout.addWidget(self.lista_clases)

        self.refresh_classes_button = QPushButton('Actualizar clases', self)
        self.refresh_classes_button.setFixedSize(180, 50)
        self.refresh_classes_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.refresh_classes_button.setStyleSheet("QPushButton { border: 3px solid black; border-radius: 10px; background-color: orange; color: black; } QPushButton:hover { background-color: #CCCCCC; color: black; }")
        self.refresh_classes_button.clicked.connect(self.cargar_clases_disponibles)
        clases_layout.addWidget(self.refresh_classes_button)

        clases_layout.addStretch()
        layout.addLayout(clases_layout)
        layout.addSpacing(15)

        self.setLayout(layout)
        self.hide_elements()

    def hide_elements(self):
        for widget in [self.label, self.combo_tipo, self.show_image_button, self.result_label, self.metrics_label,
                       self.classes_label, self.lista_clases, self.refresh_classes_button,
                       self.open_dataset_file_button, self.train_button, self.classification_report_button,
                       self.confusion_matrix_button]:
            widget.hide()

    def show_elements(self):
        for widget in [self.label, self.combo_tipo, self.show_image_button, self.result_label, self.metrics_label,
                       self.classes_label, self.lista_clases, self.refresh_classes_button,
                       self.open_dataset_file_button, self.train_button, self.classification_report_button,
                       self.confusion_matrix_button]:
            widget.show()
        self.cargar_clases_disponibles()

    def show_training(self):
        self.training_requested.emit()

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

    def update_label(self, new_text):
        current_text = self.result_label.text()
        self.result_label.setText(current_text + "\n" + new_text)

    def open_dataset_directory(self):
        dataset_full_path = os.path.join(ROOT_DIR, 'dataset.pickle')
        if not os.path.exists(dataset_full_path):
            QMessageBox.critical(self, "Error", "El archivo dataset.pickle no existe.")
            return
        try:
            import platform
            sistema = platform.system()
            if sistema == 'Windows':
                subprocess.Popen(f'explorer {ROOT_DIR}')
            elif sistema == 'Darwin':
                subprocess.Popen(['open', ROOT_DIR])
            else:
                subprocess.Popen(['xdg-open', ROOT_DIR])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir la carpeta: {e}")

    def tipo_actual(self):
        return 'letras' if self.combo_tipo.currentIndex() == 0 else 'numeros'

    def etiquetas_del_tipo(self):
        return set(LETRAS_LSP.values()) if self.combo_tipo.currentIndex() == 0 else set(NUMEROS_LSP.values())

    def nombre_modelo(self):
        return os.path.join(ROOT_DIR, f'svm_{self.tipo_actual()}_lsp_classifier.p')

    def cargar_clases_disponibles(self):
        """Muestra, como lista con casillas, las clases que existen en el dataset para el
        tipo seleccionado (letras o numeros). Todas vienen marcadas por defecto."""
        seleccion_previas = {self.lista_clases.item(i).text() for i in range(self.lista_clases.count())
                             if self.lista_clases.item(i).checkState() == Qt.Checked}

        dataset_path = os.path.join(ROOT_DIR, 'dataset.pickle')
        etiquetas = []
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'rb') as f:
                    datos = pickle.load(f)
                labels = [str(e) for e in datos.get('labels', [])]
                permitidas = self.etiquetas_del_tipo()
                etiquetas = sorted({l for l in labels if l in permitidas},
                                   key=lambda s: (len(s), s))
            except Exception as e:
                print(f'No se pudo leer dataset.pickle: {e}')

        self.lista_clases.clear()
        for etiqueta in etiquetas:
            item = QListWidgetItem(etiqueta)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            marcado = (etiqueta in seleccion_previas) if seleccion_previas else True
            item.setCheckState(Qt.Checked if marcado else Qt.Unchecked)
            self.lista_clases.addItem(item)

    def train_model(self):
        try:
            dataset_path = os.path.join(ROOT_DIR, 'dataset.pickle')

            self.result_label.setText('')
            self.update_label("Cargando dataset...")

            data_dict = pickle.load(open(dataset_path, 'rb'))
            self.update_label("Dataset cargado.")

            data = data_dict['data']
            labels = [str(etiqueta) for etiqueta in data_dict['labels']]
            tipo = self.tipo_actual()
            self.update_label(f"Total de muestras en el dataset: {len(data)}")

            if not data:
                QMessageBox.warning(self, "Advertencia", "El dataset esta vacio. Captura datos primero.")
                self.update_label("Entrenamiento cancelado: el dataset esta vacio.")
                return

            # 1) Filtra por tipo de seña (letras o numeros) para el modelo seleccionado.
            permitidas = self.etiquetas_del_tipo()
            pares = [(vector, etiqueta) for vector, etiqueta in zip(data, labels)
                     if etiqueta in permitidas]

            if not pares:
                QMessageBox.warning(
                    self, "Sin datos de este tipo",
                    f"No hay muestras de tipo '{tipo}' en el dataset.\n\n"
                    "Captura senas de ese tipo en 'Recoleccion de Datos' o cambia el Tipo."
                )
                self.update_label(f"Entrenamiento cancelado: no hay datos de {tipo}.")
                return

            # 2) Filtra por las clases marcadas en la lista (si hay alguna marcada).
            marcadas = {self.lista_clases.item(i).text() for i in range(self.lista_clases.count())
                        if self.lista_clases.item(i).checkState() == Qt.Checked}
            if marcadas:
                pares = [(vector, etiqueta) for vector, etiqueta in pares if etiqueta in marcadas]
            self.update_label(
                f"Tipo '{tipo}' - clases marcadas: {', '.join(sorted(marcadas)) if marcadas else 'todas'}"
            )

            if not pares:
                QMessageBox.warning(self, "Advertencia", "Ninguna clase marcada tiene datos.")
                self.update_label("Entrenamiento cancelado: no hay clases seleccionadas con datos.")
                return

            data = np.asarray([vector for vector, _ in pares])
            labels = np.asarray([etiqueta for _, etiqueta in pares])
            self.update_label(f"Muestras a usar: {len(data)}")

            clases, conteos = np.unique(labels, return_counts=True)
            detalle = ', '.join(f'{c}: {n}' for c, n in zip(clases, conteos))
            self.update_label(f"Clases detectadas ({len(clases)}): {detalle}")

            if len(clases) < 2:
                QMessageBox.warning(
                    self, "Datos insuficientes",
                    f"Solo hay una clase ('{clases[0]}' con {conteos[0]} muestras) para el tipo "
                    f"'{self.tipo_actual()}'.\n\n"
                    "Un clasificador SVM necesita al menos 2 clases. Captura otra letra o numero "
                    "(usa 'Clase a capturar' en Recoleccion de Datos para elegir cual)."
                )
                self.update_label("Entrenamiento cancelado: se necesitan al menos 2 clases.")
                return

            if conteos.min() < 2:
                clase_corta = clases[int(np.argmin(conteos))]
                QMessageBox.warning(
                    self, "Datos insuficientes",
                    f"La clase '{clase_corta}' solo tiene {int(conteos.min())} muestra(s).\n\n"
                    "Cada clase necesita al menos 2 muestras para poder separar entrenamiento y "
                    "prueba. Captura mas muestras de esa clase."
                )
                self.update_label("Entrenamiento cancelado: cada clase necesita al menos 2 muestras.")
                return

            self.update_label("Separando datos de entrenamiento y prueba...")
            try:
                x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
            except ValueError:
                x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, shuffle=True)
            self.update_label("Conjuntos de entrenamiento y prueba creados.")

            parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
            classifier = SVC(probability=True)
            n_folds = max(2, min(10, int(conteos.min())))
            grid_search = GridSearchCV(classifier, parameters, cv=n_folds, n_jobs=-1, refit=True, verbose=2)
            self.update_label(f"Validacion cruzada con {n_folds} particiones...")

            start_time = time.time()
            grid_search.fit(x_train, y_train)
            end_time = time.time()

            best_svm_classifier = grid_search.best_estimator_
            best_svm_classifier.fit(x_train, y_train)

            with open(self.nombre_modelo(), 'wb') as f:
                pickle.dump({'model': best_svm_classifier}, f)

            training_time = end_time - start_time
            self.update_label(f"Modelo entrenado en: {training_time:.2f} segundos")
            self.first_click = False

            y_predict = best_svm_classifier.predict(x_val)
            acc = accuracy_score(y_val, y_predict)
            prec = precision_score(y_val, y_predict, average="weighted", zero_division=0)
            rec = recall_score(y_val, y_predict, average="weighted", zero_division=0)
            f1 = f1_score(y_val, y_predict, average="weighted", zero_division=0)

            metrics_text = (
                "Metricas de validacion:\n"
                f"Exactitud: {acc * 100:.2f}%\n"
                f"Precision: {prec:.2f}\n"
                f"Recall: {rec:.2f}\n"
                f"F1-Score: {f1:.2f}"
            )
            self.metrics_label.setText(metrics_text)

            conf_matrix = confusion_matrix(y_val, y_predict)
            class_report = classification_report(y_val, y_predict, zero_division=0)

            with open(os.path.join(ROOT_DIR, 'reporte_clasificacion.txt'), 'w', encoding='utf-8') as f:
                f.write(f'Mejores parametros: {grid_search.best_params_}\n\n')
                f.write(f'Mejor puntaje de validacion cruzada: {grid_search.best_score_}\n\n')
                f.write(f'Tiempo de entrenamiento: {training_time:.2f} segundos\n\n')
                f.write(f'Exactitud de validacion: {acc:.2f}\n\n')
                f.write(f'Matriz de confusion:\n{conf_matrix}\n\n')
                f.write(f'Reporte de clasificacion:\n{class_report}\n')

            self.y_val = y_val
            self.y_predict = y_predict

        except Exception as e:
            self.update_label(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al entrenar el modelo: {e}")

    def open_classification_report(self):
        reporte_path = os.path.join(ROOT_DIR, 'reporte_clasificacion.txt')
        if not os.path.exists(reporte_path):
            QMessageBox.warning(self, "Advertencia", "Primero entrena un modelo.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(reporte_path))

    def show_confusion_matrix(self):
        try:
            if self.y_val is None:
                QMessageBox.warning(self, "Advertencia", "Primero entrena un modelo.")
                return

            etiquetas = sorted(np.unique(self.y_val), key=lambda k: str(k))
            cm = confusion_matrix(self.y_val, self.y_predict, labels=etiquetas)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=etiquetas, yticklabels=etiquetas)
            plt.xlabel('Prediccion')
            plt.ylabel('Real')
            plt.title('Matriz de Confusion - LSP - SVM')
            plt.savefig(os.path.join(ROOT_DIR, 'matriz_confusion_svm.png'))
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallo al mostrar la matriz de confusion: {e}")
