"""Pantallas de camara (OpenCV) que comparten la captura y la prueba de senas con movimiento.

Es un "mixin": una clase con metodos listos para mezclar con un QWidget. Los metodos dibujan
sobre la ventana de OpenCV: espera con guia de encuadre, cuenta regresiva, grabacion y analisis.
"""
import os
import sys
import textwrap
import time

import cv2
import mediapipe as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extraer_secuencias as ext
import lsp_dynamic as dyn

VENTANA = 'Senas con Movimiento'
ESC, ESPACIO = 27, 32
AMARILLO, BLANCO, VERDE, ROJO, NARANJA = (0, 255, 255), (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 165, 255)


class PantallasCamara:
    # ------------------------------------------------------------------ utilidades
    def _leer_tecla(self):
        return cv2.waitKey(1) & 0xFF

    def _leer_frame(self, cap):
        """Un frame de la camara, ya reducido a ANCHO_MAX (lo mismo que hace la prueba en vivo)."""
        return dyn.leer_frame(cap)

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

    def _dibujar_resultado(self, frame, resultado):
        """Resultado grande (texto, color) centrado en la parte baja: lo que se reconocio."""
        if not resultado:
            return
        texto, color = resultado
        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
        escala = 1.8 if w < frame.shape[1] - 30 else 1.8 * (frame.shape[1] - 30) / w
        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 4)
        x, y = (frame.shape[1] - w) // 2, frame.shape[0] - 70
        cv2.rectangle(frame, (x - 12, y - h - 14), (x + w + 12, y + 16), (0, 0, 0), -1)
        self._texto(frame, texto, (x, y), escala, color, 4)

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

    # ------------------------------------------------------------------ pantallas
    def _pantalla_espera(self, cap, titulo, ayuda, nota, aviso, pose, permitir_borrar=False, resultado=None):
        """Vista previa sin grabar. Devuelve 'listo' (ESPACIO), 'borrar' (B), 'cancelado' (ESC) o 'error'."""
        while True:
            frame = self._leer_frame(cap)
            if frame is None:
                return 'error'
            encuadre_ok, texto_encuadre = self._evaluar_encuadre(frame, pose)

            self._banda(frame, 150 if nota else 120)
            self._texto(frame, titulo, (14, 36), 1.0, AMARILLO, 2)
            self._texto(frame, ayuda, (14, 70), 0.62, BLANCO)
            self._texto(frame, texto_encuadre, (14, 102), 0.7, VERDE if encuadre_ok else NARANJA)
            if nota:
                self._texto(frame, nota, (14, 136), 0.58, AMARILLO, 1)
            self._dibujar_resultado(frame, resultado)
            self._dibujar_aviso(frame, aviso)
            cv2.imshow(VENTANA, frame)

            tecla = self._leer_tecla()
            if tecla == ESC:
                return 'cancelado'
            if tecla == ESPACIO:
                return 'listo'
            if permitir_borrar and tecla in (ord('b'), ord('B')):
                return 'borrar'

    def _pantalla_cuenta(self, cap, titulo, ayuda):
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
            self._texto(frame, titulo, (14, 36), 1.0, AMARILLO, 2)
            self._texto(frame, ayuda, (14, 66), 0.6, BLANCO)
            numero = str(int(restante) + 1)
            (w, h), _ = cv2.getTextSize(numero, cv2.FONT_HERSHEY_SIMPLEX, 8, 16)
            self._texto(frame, numero, ((ancho - w) // 2, (alto + h) // 2), 8, AMARILLO, 16)
            cv2.imshow(VENTANA, frame)
            if self._leer_tecla() == ESC:
                return 'cancelado'

    def _pantalla_grabacion(self, cap, titulo):
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
            self._texto(vista, titulo, (54, 40), 0.9, BLANCO, 2)
            avance = min(1.0, transcurrido / dyn.DURACION_GRABACION)
            cv2.rectangle(vista, (0, 52), (int(ancho * avance), 60), ROJO, -1)
            cv2.imshow(VENTANA, vista)

            if self._leer_tecla() == ESC:
                return None, None, 'cancelado'
            if transcurrido >= dyn.DURACION_GRABACION:
                return frames, tiempos, 'ok'

    def _procesar_frames(self, frames, mensaje):
        """Pasa los frames por MediaPipe Holistic mostrando el avance. Devuelve el dict de puntos (o None).

        Se crea un Holistic NUEVO en cada llamada, igual que al extraer un video: el seguimiento de
        MediaPipe arrastra el estado de lo que vio antes, y reutilizarlo entre grabaciones degrada la
        deteccion de la siguiente (se midio perder frames con mano que con uno nuevo si se detectan).
        """
        total = len(frames)

        def avance(i):
            if i % 8 == 0:   # mantiene viva la ventana y muestra el avance
                vista = frames[-1].copy()
                self._banda(vista, 60)
                self._texto(vista, f'{mensaje}  {i + 1}/{total}', (14, 40), 0.9, AMARILLO, 2)
                cv2.imshow(VENTANA, vista)
                self._leer_tecla()

        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            return ext.landmarks_de_frames(frames, holistic, avance)
