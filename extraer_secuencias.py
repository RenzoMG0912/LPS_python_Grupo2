"""Convierte los videos de la captura con movimiento en el dataset dinamico.

Uso (desde la raiz del proyecto, con el entorno virtual activado):

    python extraer_secuencias.py
    python extraer_secuencias.py --forzar        # vuelve a pasar los videos por MediaPipe

Hace tres cosas por cada video de Videos_capturados/<clase>/*.mp4:
  1. Saca los puntos de manos y cuerpo con MediaPipe Holistic. Es lo lento (unos segundos por
     video), asi que el resultado se guarda en Landmarks_extraidos/ y no se repite.
  2. Recorta la sena y calcula el vector de caracteristicas (ver lsp_dynamic_features.py).
  3. Junta todo en dataset_dinamico.pickle y escribe un reporte con lo que se rechazo y por que.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mediapipe')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import lsp_dynamic as dyn
import lsp_dynamic_features as feat


def _mano(landmarks):
    if landmarks is None:
        return np.full((21, 3), np.nan, np.float32)
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark], np.float32)


def landmarks_de_frames(frames, holistic, al_avanzar=None):
    """Pasa una secuencia de imagenes BGR por un `holistic` ya creado.

    Lo usan la extraccion de videos y la prueba en vivo, asi los dos producen exactamente lo mismo.
    Devuelve un dict con pose (T,33,4), izq y der (T,21,3) (NaN donde no hubo deteccion), ancho, alto.
    `al_avanzar(i)` se llama despues de cada frame (para mostrar el avance). None si no hubo frames.
    """
    pose, izq, der, ancho, alto = [], [], [], 0, 0
    for i, frame in enumerate(frames):
        alto, ancho = frame.shape[:2]
        r = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if r.pose_landmarks:
            pose.append(np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks.landmark], np.float32))
        else:
            pose.append(np.full((33, 4), np.nan, np.float32))
        izq.append(_mano(r.left_hand_landmarks))
        der.append(_mano(r.right_hand_landmarks))
        if al_avanzar is not None:
            al_avanzar(i)
    if not pose:
        return None
    return {'pose': np.array(pose), 'izq': np.array(izq), 'der': np.array(der), 'ancho': ancho, 'alto': alto}


def _frames_del_video(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def extraer_landmarks(ruta_video):
    """Pasa un video por MediaPipe Holistic. Devuelve dict con pose (T,33,4), izq y der (T,21,3), ancho, alto, fps."""
    import mediapipe as mp   # se importa aqui para que el resto del modulo se pueda usar sin MediaPipe

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        raw = landmarks_de_frames(_frames_del_video(cap), holistic)
    cap.release()
    if raw is None:
        return None
    raw['fps'] = fps
    return raw


def cargar_o_extraer(ruta_video, ruta_cache, forzar):
    """Usa la cache de puntos si existe y es mas nueva que el video; si no, la calcula y la guarda."""
    if not forzar and os.path.exists(ruta_cache) and os.path.getmtime(ruta_cache) >= os.path.getmtime(ruta_video):
        d = np.load(ruta_cache)
        return {k: d[k] for k in ('pose', 'izq', 'der')} | {'ancho': int(d['ancho']), 'alto': int(d['alto']), 'fps': float(d['fps'])}, True
    raw = extraer_landmarks(ruta_video)
    if raw is not None:
        os.makedirs(os.path.dirname(ruta_cache), exist_ok=True)
        np.savez_compressed(ruta_cache, **raw)
    return raw, False


def datos_de_la_muestra(ruta_video, carpeta_a_etiqueta):
    """Etiqueta y persona de un video, desde su .json (o, si falta, desde la carpeta y el nombre)."""
    ruta_json = ruta_video[:-4] + '.json'
    if os.path.exists(ruta_json):
        with open(ruta_json, encoding='utf-8') as f:
            meta = json.load(f)
        return meta.get('etiqueta'), meta.get('persona', '?')
    carpeta = os.path.basename(os.path.dirname(ruta_video))
    persona = os.path.basename(ruta_video).split('_')[0]
    return carpeta_a_etiqueta.get(carpeta), persona


def procesar(dir_videos, dir_cache, salida, forzar=False, log=print):
    """Convierte los videos en el dataset dinamico. Es lo que ejecuta tanto este script como el boton de la interfaz.

    `log(texto)` recibe cada aviso de progreso. Devuelve un dict:
      {'ok': False, 'motivo': ...}                      si no se pudo (no hay videos, o ninguna muestra sirvio)
      {'ok': True, 'aceptadas': n, 'rechazadas': n, 'texto': reporte, 'salida': ruta}  si se guardo el dataset
    """
    videos = sorted(glob.glob(os.path.join(dir_videos, '*', '*.mp4')))
    if not videos:
        return {'ok': False, 'motivo': f'No hay videos en {dir_videos}. Graba muestras con la pestana Captura.'}

    carpeta_a_etiqueta = {c.carpeta: c.etiqueta for c in dyn.CLASES_DINAMICAS}
    data, labels, personas, archivos, infos, secuencias = [], [], [], [], [], []
    ambiguas, cortadas, rechazadas = [], [], []
    inicio = time.time()

    for i, ruta in enumerate(videos, 1):
        rel = os.path.relpath(ruta, dir_videos)
        etiqueta, persona = datos_de_la_muestra(ruta, carpeta_a_etiqueta)
        if etiqueta is None:
            rechazadas.append((rel, '?', 'no se pudo saber la etiqueta (falta el .json y la carpeta no es una clase)'))
            log(f'[{i}/{len(videos)}] {rel}: RECHAZADA (sin etiqueta)')
            continue

        ruta_cache = os.path.join(dir_cache, os.path.dirname(rel), os.path.basename(ruta)[:-4] + '.npz')
        raw, de_cache = cargar_o_extraer(ruta, ruta_cache, forzar)
        if raw is None:
            rechazadas.append((rel, etiqueta, 'no se pudo leer el video'))
            log(f'[{i}/{len(videos)}] {rel}: RECHAZADA (no se pudo leer el video)')
            continue

        vector, info = feat.extraer_caracteristicas(raw['pose'], raw['izq'], raw['der'], raw['ancho'], raw['alto'])
        origen = 'cache' if de_cache else 'MediaPipe'
        if vector is None:
            rechazadas.append((rel, etiqueta, info['motivo']))
            log(f'[{i}/{len(videos)}] {rel}: RECHAZADA ({info["motivo"]})')
            continue

        data.append(vector)
        secuencias.append(info.pop('secuencia'))      # la secuencia recortada, sin remuestrear (para DTW)
        labels.append(etiqueta)
        personas.append(persona)
        archivos.append(rel.replace('\\', '/'))
        infos.append(info)
        aviso = ''
        if info.get('ambiguo'):
            ambiguas.append((rel, etiqueta))
            aviso = '  [RECORTE AMBIGUO: revisalo]'
        if info.get('cortada_al_final') or info.get('cortada_al_inicio'):
            donde = 'al final' if info.get('cortada_al_final') else 'al inicio'
            cortadas.append((rel, etiqueta, donde))
            aviso += f'  [SENA POSIBLEMENTE CORTADA {donde.upper()}]'
        log(f'[{i}/{len(videos)}] {rel}: ok [{etiqueta}, {persona}] bloque {info["bloque"][0]}-{info["bloque"][1]} '
            f'({info["frames_bloque"]} frames), mano {info["lado_principal"]}'
            f'{", dos manos" if info["dos_manos"] else ""}  ({origen}){aviso}')

    if not data:
        return {'ok': False, 'motivo': 'Ninguna muestra paso el control. Revisa los motivos de arriba.'}

    dataset = {'data': np.vstack(data).astype(np.float32), 'labels': labels, 'personas': personas,
               'archivos': archivos, 'info': infos, 'secuencias': secuencias,
               'formato': feat.parametros(), 'dim': feat.DIM_VECTOR}
    with open(salida, 'wb') as f:
        pickle.dump(dataset, f)

    lineas = [f'Videos revisados: {len(videos)} | aceptados: {len(data)} | rechazados: {len(rechazadas)} '
              f'| tiempo: {time.time() - inicio:.0f} s', '', 'Por clase:']
    for etiqueta in sorted(set(labels)):
        n = labels.count(etiqueta)
        quienes = sorted({p for p, l in zip(personas, labels) if l == etiqueta})
        lineas.append(f'  {etiqueta:10s} {n:4d} muestras | personas: {", ".join(quienes)}')
    if cortadas:
        lineas += ['', 'Sena posiblemente cortada (la mano seguia moviendose cuando termino o empezo el video):'] + \
                  [f'  {rel}  [{et}]  {donde}' for rel, et, donde in cortadas]
    if ambiguas:
        lineas += ['', 'Recorte ambiguo (el tramo elegido casi empata con otro; conviene mirar el video):'] + \
                  [f'  {rel}  [{et}]' for rel, et in ambiguas]
    if rechazadas:
        lineas += ['', 'Rechazadas:'] + [f'  {rel}  [{et}]  {motivo}' for rel, et, motivo in rechazadas]
    texto = '\n'.join(lineas)
    with open(os.path.join(os.path.dirname(salida) or '.', 'reporte_extraccion.txt'), 'w', encoding='utf-8') as f:
        f.write(texto + '\n')
    return {'ok': True, 'aceptadas': len(data), 'rechazadas': len(rechazadas), 'texto': texto, 'salida': salida}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--videos', default=os.path.join(ROOT_DIR, 'Videos_capturados'))
    ap.add_argument('--cache', default=os.path.join(ROOT_DIR, 'Landmarks_extraidos'))
    ap.add_argument('--salida', default=os.path.join(ROOT_DIR, 'dataset_dinamico.pickle'))
    ap.add_argument('--forzar', action='store_true', help='ignora la cache y vuelve a usar MediaPipe')
    args = ap.parse_args()

    r = procesar(args.videos, args.cache, args.salida, args.forzar)
    if not r['ok']:
        print('\n' + r['motivo'])
        return 1
    print('\n' + r['texto'])
    print(f'\nDataset guardado en {r["salida"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
