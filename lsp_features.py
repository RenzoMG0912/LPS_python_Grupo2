NUM_LANDMARKS = 21
VECTOR_POR_MANO = NUM_LANDMARKS * 2
VECTOR_TOTAL = VECTOR_POR_MANO * 2


def _normalizar_mano(hand_landmarks):
    xs = [p.x for p in hand_landmarks.landmark]
    ys = [p.y for p in hand_landmarks.landmark]
    min_x, min_y = min(xs), min(ys)

    vector = []
    for p in hand_landmarks.landmark:
        vector.append(p.x - min_x)
        vector.append(p.y - min_y)
    return vector


def extraer_vector(multi_hand_landmarks, multi_handedness):
    vector_izq = [0.0] * VECTOR_POR_MANO
    vector_der = [0.0] * VECTOR_POR_MANO

    if multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(multi_hand_landmarks):
            etiqueta = 'Right'
            if multi_handedness and idx < len(multi_handedness):
                etiqueta = multi_handedness[idx].classification[0].label

            vector_mano = _normalizar_mano(hand_landmarks)

            if etiqueta == 'Left':
                vector_izq = vector_mano
            else:
                vector_der = vector_mano

    return vector_izq + vector_der
