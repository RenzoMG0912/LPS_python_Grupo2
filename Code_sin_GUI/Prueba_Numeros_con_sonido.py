import os
import sys
import pickle

import cv2
import mediapipe as mp
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsp_features import extraer_vector
from lsp_tts import Hablante

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONIDOS_DIR = os.path.join(ROOT_DIR, 'Sonidos')
MODELO_PATH = os.path.join(ROOT_DIR, 'svm_numeros_lsp_classifier.p')

pygame.mixer.init()
hablante = Hablante()

with open(MODELO_PATH, 'rb') as f:
    svm_numeros = pickle.load(f)['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)


def dibujar_texto(imagen, texto, posicion, tamano_fuente=50, probabilidad=None):
    imagen_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imagen_pil)
    try:
        fuente = ImageFont.truetype('arial.ttf', tamano_fuente)
    except OSError:
        fuente = ImageFont.load_default()
    draw.text(posicion, texto, font=fuente, fill=(255, 0, 0, 255))
    if probabilidad is not None:
        pos_prob = (posicion[0] + tamano_fuente + 5, posicion[1])
        draw.text(pos_prob, f'{probabilidad:.2f}', font=fuente, fill=(255, 0, 0, 255))
    return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)


anterior = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        vector = extraer_vector(resultados.multi_hand_landmarks, resultados.multi_handedness)
        prediccion = svm_numeros.predict([vector])[0]
        probabilidades = svm_numeros.predict_proba([vector])[0]
        clases = list(svm_numeros.classes_)
        indice_clase = clases.index(prediccion)

        if prediccion != anterior:
            anterior = prediccion
            audio_path = os.path.join(SONIDOS_DIR, f'{prediccion}.mp3')
            if os.path.exists(audio_path):
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
            else:
                # Sin archivo propio (Sonidos/<clase>.mp3): se dice con la voz del sistema.
                hablante.decir(prediccion)

        frame = dibujar_texto(frame, prediccion, (10, 30), tamano_fuente=50, probabilidad=probabilidades[indice_clase])

    cv2.imshow('Reconocimiento LSP - Numeros', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
hablante.cerrar()
