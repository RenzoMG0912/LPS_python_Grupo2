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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONIDOS_DIR = os.path.join(ROOT_DIR, 'Sonidos')
MODELO_PATH = os.path.join(ROOT_DIR, 'svm_letras_lsp_classifier.p')

pygame.mixer.init()

with open(MODELO_PATH, 'rb') as f:
    svm_letras = pickle.load(f)['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)


def dibujar_texto(imagen, texto, posicion, tamano_fuente=90):
    imagen_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imagen_pil)
    try:
        fuente = ImageFont.truetype('arial.ttf', tamano_fuente)
    except OSError:
        fuente = ImageFont.load_default()
    draw.text(posicion, texto, font=fuente, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)


def predecir(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)
    if not resultados.multi_hand_landmarks:
        return None, resultados
    vector = extraer_vector(resultados.multi_hand_landmarks, resultados.multi_handedness)
    prediccion = svm_letras.predict([vector])[0]
    return str(prediccion), resultados


anterior = None
actual = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    letra, resultados = predecir(frame)

    if letra is not None and letra != anterior:
        anterior = letra
        actual = letra
        audio_path = os.path.join(SONIDOS_DIR, f'{letra}.mp3')
        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()

    if resultados and resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    if actual is not None:
        frame = dibujar_texto(frame, actual, (80, 80), tamano_fuente=90)

    cv2.imshow('Reconocimiento LSP - Letras', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
