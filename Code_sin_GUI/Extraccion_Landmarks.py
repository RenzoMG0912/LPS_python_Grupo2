import os
import sys
import pickle
from collections import defaultdict

import cv2
import mediapipe as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsp_features import extraer_vector

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Mis_Datos')

mp_hands = mp.solutions.hands

data = []
labels = []
class_count = defaultdict(int)

with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f'No se pudo leer la imagen {img_path}')
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                continue

            vector = extraer_vector(results.multi_hand_landmarks, results.multi_handedness)
            data.append(vector)
            labels.append(dir_)
            class_count[dir_] += 1

if class_count:
    max_por_clase = max(class_count.values())

    for clase in list(class_count.keys()):
        indices_clase = [i for i, etiqueta in enumerate(labels) if etiqueta == clase]
        while class_count[clase] < max_por_clase:
            idx = indices_clase[class_count[clase] % len(indices_clase)]
            data.append(data[idx])
            labels.append(clase)
            class_count[clase] += 1

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset.pickle')
with open(dataset_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f'Dataset generado con {len(data)} muestras en {len(class_count)} clases.')
