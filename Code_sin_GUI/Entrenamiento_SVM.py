import argparse
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from sklearn.svm import SVC

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--tipo', choices=['letras', 'numeros'], default='letras')
parser.add_argument('--dataset', default=os.path.join(ROOT_DIR, 'dataset.pickle'))
args = parser.parse_args()

nombre_modelo = os.path.join(ROOT_DIR, f'svm_{args.tipo}_lsp_classifier.p')

data_dict = pickle.load(open(args.dataset, 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

parametros = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
clasificador = SVC(probability=True)
busqueda = GridSearchCV(clasificador, parametros, cv=10, n_jobs=-1, refit=True, verbose=2)

inicio = time.time()
busqueda.fit(x_train, y_train)
fin = time.time()

mejor_modelo = busqueda.best_estimator_
mejor_modelo.fit(x_train, y_train)

with open(nombre_modelo, 'wb') as f:
    pickle.dump({'model': mejor_modelo}, f)

y_pred = mejor_modelo.predict(x_test)
exactitud = accuracy_score(y_test, y_pred) * 100

matriz_confusion = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred, zero_division=0)

reporte_path = os.path.join(ROOT_DIR, f'reporte_clasificacion_{args.tipo}.txt')
with open(reporte_path, 'w', encoding='utf-8') as f:
    f.write(f'Mejores parametros: {busqueda.best_params_}\n\n')
    f.write(f'Mejor puntaje de validacion cruzada: {busqueda.best_score_}\n\n')
    f.write(f'Tiempo de entrenamiento: {fin - inicio:.2f} segundos\n\n')
    f.write(f'Exactitud SVM: {exactitud:.2f}%\n\n')
    f.write(f'Total de datos: {len(data)}\n')
    f.write(f'Datos de entrenamiento: {len(x_train)}\n')
    f.write(f'Datos de prueba: {len(x_test)}\n\n')
    f.write(f'Matriz de confusion:\n{matriz_confusion}\n\n')
    f.write(f'Reporte de clasificacion:\n{reporte}\n')

print(f'Exactitud SVM ({args.tipo}): {exactitud:.2f}%')
print(f'Tiempo de entrenamiento: {fin - inicio:.2f} segundos')

etiquetas_orden = sorted(np.unique(labels), key=lambda k: str(k))
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Greens', xticklabels=etiquetas_orden, yticklabels=etiquetas_orden)
plt.xlabel('Prediccion')
plt.ylabel('Real')
plt.title(f'Matriz de Confusion - LSP {args.tipo.capitalize()} - SVM')
plt.savefig(os.path.join(ROOT_DIR, f'matriz_confusion_{args.tipo}.png'))
plt.show()
