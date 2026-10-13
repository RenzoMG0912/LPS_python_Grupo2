import json
import os
import re
import time
from collections import namedtuple

import cv2

DURACION_GRABACION = 2.5
CUENTA_REGRESIVA = 3
REPETICIONES_POR_DEFECTO = 20
RESOLUCION_CAMARA = (1280, 720)
ANCHO_MAX = 960
FPS_MIN, FPS_MAX = 5.0, 60.0
MIN_FRAMES_GRABADOS = 10

MIN_POSE = 0.90
MIN_FRAMES_MANO = 12
MIN_FRAMES_DOS_MANOS = 8

ANCHO_HOMBROS_MIN = 0.15
ANCHO_HOMBROS_MAX = 0.35

ClaseDinamica = namedtuple('ClaseDinamica', 'etiqueta carpeta tipo manos nota')

CLASES_DINAMICAS = [
    ClaseDinamica('J', 'LETRA_J', 'Letra', 1, ''),
    ClaseDinamica('Ñ', 'LETRA_ENIE', 'Letra', 1, ''),
    ClaseDinamica('Z', 'LETRA_Z', 'Letra', 1, ''),
    ClaseDinamica('10', 'NUM_10', 'Numero', 1, ''),
    ClaseDinamica('HOLA', 'PAL_HOLA', 'Palabra', 1,
                  'Variante 1: saludo con la mano abierta junto al hombro'),
    ClaseDinamica('CHAU', 'PAL_CHAU', 'Palabra', 1,
                  'Variante 2: mano abierta; los dedos se doblan y se abren varias veces'),
    ClaseDinamica('LO-SIENTO', 'PAL_LO-SIENTO', 'Palabra', 1, ''),
    ClaseDinamica('BIEN', 'PAL_BIEN', 'Palabra', 1, ''),
    ClaseDinamica('MALO', 'PAL_MALO', 'Palabra', 1, ''),
    ClaseDinamica('AMIGO', 'PAL_AMIGO', 'Palabra', 2, ''),
    ClaseDinamica('ENTENDER', 'PAL_ENTENDER', 'Palabra', 1, ''),
    ClaseDinamica('PREGUNTAR', 'PAL_PREGUNTAR', 'Palabra', 1, 'Variante 2 del dataset'),
    ClaseDinamica('DONDE', 'PAL_DONDE', 'Palabra', 2, ''),
    ClaseDinamica('AHÍ', 'PAL_AHI', 'Palabra', 1, ''),
]

_NOMBRE_EN_PANTALLA = {'Ñ': 'N (con tilde)', 'AHÍ': 'AHI'}


def nombre_en_pantalla(clase):
    return _NOMBRE_EN_PANTALLA.get(clase.etiqueta, clase.etiqueta)


def limpiar_id(texto):
    """Deja solo letras, numeros, guion y guion bajo (el ID va en el nombre de los archivos)."""
    return re.sub(r'[^A-Za-z0-9_-]', '', (texto or '').strip())


def carpeta_clase(raiz, clase):
    return os.path.join(raiz, clase.carpeta)


def contar_muestras(raiz, clase, persona=None):
    """Cuantos videos hay de una clase (y, si se indica, de una persona)."""
    carpeta = carpeta_clase(raiz, clase)
    if not os.path.isdir(carpeta):
        return 0
    prefijo = f'{persona}_' if persona else ''
    return sum(1 for nombre in os.listdir(carpeta)
               if nombre.lower().endswith('.mp4') and nombre.startswith(prefijo))


def evaluar_calidad(resumen, dos_manos):
    """Decide si una muestra grabada sirve.
    `resumen` es una lista con un dict por frame: {'pose': bool, 'manos': 0, 1 o 2}.
    Devuelve (aceptada, motivo, estadisticas). Los motivos van sin tildes porque
    tambien se dibujan sobre el video.
    """
    n = len(resumen)
    if n == 0:
        return False, 'no se grabo ningun frame', {}

    con_pose = sum(1 for r in resumen if r['pose'])
    con_mano = sum(1 for r in resumen if r['manos'] >= 1)
    con_dos = sum(1 for r in resumen if r['manos'] >= 2)
    estadisticas = {'frames': n, 'pose': round(con_pose / n, 3),
                    'frames_con_mano': con_mano, 'frames_con_dos_manos': con_dos}

    if con_pose / n < MIN_POSE:
        return False, (f'no se ven bien los hombros ({100 * con_pose / n:.0f}% de los frames, '
                       f'minimo {100 * MIN_POSE:.0f}%): ajusta el encuadre'), estadisticas
    if con_mano < MIN_FRAMES_MANO:
        return False, (f'la mano se detecto en solo {con_mano} frames (minimo {MIN_FRAMES_MANO}): '
                       'mejora la luz o el contraste con la ropa'), estadisticas
    if dos_manos and con_dos < MIN_FRAMES_DOS_MANOS:
        return False, (f'las dos manos se vieron juntas en solo {con_dos} frames '
                       f'(minimo {MIN_FRAMES_DOS_MANOS})'), estadisticas
    return True, 'ok', estadisticas


def fps_real(tiempos):
    """Cuadros por segundo medidos a partir de la hora de cada frame, dentro de un rango razonable."""
    if len(tiempos) < 2 or tiempos[-1] <= tiempos[0]:
        return FPS_MAX
    fps = (len(tiempos) - 1) / (tiempos[-1] - tiempos[0])
    return max(FPS_MIN, min(FPS_MAX, fps))


def guardar_muestra(raiz, clase, persona, frames, fps, estadisticas=None):
    """Escribe el video y su .json en <raiz>/<carpeta de la clase>/.

    Devuelve (ruta_video, ruta_json). Lanza OSError si no se pudo escribir el video.
    """
    carpeta = carpeta_clase(raiz, clase)
    os.makedirs(carpeta, exist_ok=True)

    base = f'{persona}_{time.strftime("%Y%m%d_%H%M%S")}'
    ruta_video = os.path.join(carpeta, base + '.mp4')
    k = 1
    while os.path.exists(ruta_video):          # dos muestras en el mismo segundo
        ruta_video = os.path.join(carpeta, f'{base}_{k}.mp4')
        k += 1

    alto, ancho = frames[0].shape[:2]
    escritor = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto))
    if not escritor.isOpened():
        raise OSError(f'No se pudo crear el video en {ruta_video}')
    for frame in frames:
        escritor.write(frame)
    escritor.release()
    if not os.path.exists(ruta_video) or os.path.getsize(ruta_video) == 0:
        raise OSError(f'El video quedo vacio: {ruta_video}')

    ruta_json = ruta_video[:-4] + '.json'
    datos = {
        'etiqueta': clase.etiqueta,
        'tipo': clase.tipo,
        'manos': clase.manos,
        'persona': persona,
        'fecha': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fps': round(fps, 3),
        'frames': len(frames),
        'ancho': ancho,
        'alto': alto,
        'duracion_s': round(len(frames) / fps, 3),
        'calidad': estadisticas or {},
    }
    with open(ruta_json, 'w', encoding='utf-8') as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    return ruta_video, ruta_json


def borrar_muestra(ruta_video, ruta_json):
    """Borra un video y su .json (si alguno ya no existe, no pasa nada)."""
    for ruta in (ruta_video, ruta_json):
        try:
            os.remove(ruta)
        except FileNotFoundError:
            pass
