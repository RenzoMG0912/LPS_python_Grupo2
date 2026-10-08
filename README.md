## LSP-ML: Adaptación para Lengua de Señas Peruana del proyecto ArSLR-ML

Este proyecto es una adaptación de **ArSLR-ML** (reconocimiento de Lengua de Señas Árabe con SVM + MediaPipe) al contexto de la **Lengua de Señas Peruana (LSP)**. Mantiene la misma arquitectura de fondo (extracción de landmarks con MediaPipe Hands, clasificación con SVM multiclase, GUI en PyQt5), pero corrige limitaciones del código original y lo adapta para el alfabeto manual y los números en español.

Artículo base: "Static Arabic Sign Language Recognition in Real Time Using Machine Learning and MediaPipe", ICETI 2024, DOI: 10.1109/ICETI63946.2024.10777193.

### Cambios realizados respecto al proyecto original

**Adaptación a LSP**
- Diccionario de clases (`lsp_labels_dict.py`) con el alfabeto en español (A-Z, Ñ) y números (0-10) en vez del alfabeto y numerales árabes.
- Textos de la interfaz traducidos al español.
- Eliminado el renderizado y conversión de numerales arábigos (`convert_to_arabic_numerals`), innecesario para español.
- Rutas de audio y modelos ya no dependen de nombres de archivo en escritura árabe (unicode), sino de las etiquetas en español (`A.mp3`, `0.mp3`, etc.).
- Salida de voz: la seña reconocida se anuncia por el altavoz. Si existe `Sonidos/<clase>.mp3` se reproduce ese archivo; si no, la etiqueta se dice con la **voz del sistema** (TTS offline con `pyttsx3`), prefiriendo una voz en español si está instalada.

**Captura por vectores en vivo (`GUI/data_collection_widget.py`)**
El flujo original guardaba imágenes `.jpg` por clase y solo después las vectorizaba con MediaPipe al pulsar "Crear Dataset"; en la práctica la captura no llegaba a guardar nada. Ahora la cámara vectoriza directamente: por cada frame se dibujan los landmarks con MediaPipe, se extrae el vector bimanual (`extraer_vector`) y se acumula. Al alcanzar el número de capturas pedido (o al pulsar `Q`) se cierra la cámara y los vectores se **fusionan** con el `dataset.pickle` existente, conservando y ampliando los datos previos.

**Soporte bimanual real (`lsp_features.py`)**
El código original solo procesaba de forma consistente una mano; cuando había dos manos, el vector de características usado para entrenar y el usado en inferencia no eran iguales, y en el reconocimiento de números el bucle sobre `multi_hand_landmarks` predecía cada mano por separado en vez de combinarlas. Como la LSP usa señas de una y dos manos, se creó un extractor único que:
- Siempre genera un vector de longitud fija (2 manos × 21 landmarks × 2 coordenadas).
- Ordena las manos por lateralidad (`Left`/`Right`) para que la misma mano ocupe siempre la misma posición del vector.
- Rellena con ceros la mano ausente, permitiendo entrenar con señas de una sola mano y de dos manos con el mismo pipeline.
- Se usa exactamente igual en recolección de datos, entrenamiento y las tres formas de probar el modelo (script suelto, GUI de letras, GUI de números).

**Corrección de errores del repositorio original**
- `GUI/testing_widget.py` importaba `from Testing_Letter_sound import labels_dict`, un módulo que no existe con ese nombre ni esa ruta (rompía la aplicación). Se eliminó esa dependencia; ahora la predicción del SVM ya es la etiqueta en texto, sin necesidad de diccionario intermedio.
- El botón "Testing Number" nunca tenía conectado un evento `clicked`; ahora arranca la cámara en modo números.
- `Hand_landmark.py` reabría `mp_hands.Hands` en cada imagen (muy lento); ahora se abre una sola vez por ejecución.
- El aumento de datos por padding siempre duplicaba la primera muestra de cada clase (no era aleatorio pese al comentario); ahora recorre todas las muestras disponibles de la clase.
- Rutas absolutas de Windows (`C:/Users/hp/...`) reemplazadas por rutas relativas a la carpeta del proyecto, y `explorer` (solo Windows) reemplazado por apertura de carpeta multiplataforma (Windows/macOS/Linux).
- Inconsistencia de mayúsculas/minúsculas en el ícono `Image_button.png` vs `Image_Button.PNG` (fallaba en sistemas con distinción de mayúsculas, como Linux/macOS).
- `sklearnex` (Intel Extension for Scikit-learn) ahora es una importación opcional (`try/except`); antes, si no estaba instalado, la aplicación no arrancaba.
- Los botones "Reporte de Clasificación" y "Matriz de Confusión" apuntaban a la misma función; ahora cada uno hace lo que su nombre indica.

### Estructura del proyecto
```
LSP-ML/
├── lsp_labels_dict.py          Diccionario de clases (letras/números) para captura automática
├── lsp_features.py             Extractor de vector de landmarks bimanual (usado por todo el proyecto)
├── lsp_tts.py                  Voz del sistema (TTS offline) para anunciar la seña reconocida
├── requirements.txt
├── Mis_Datos/                  (Legado) Imágenes por clase; la captura actual guarda vectores directamente en dataset.pickle
├── Sonidos/                    Archivos .mp3 nombrados igual que cada clase (A.mp3, 0.mp3, ...)
├── Code_sin_GUI/
│   ├── Extraccion_Landmarks.py     Genera dataset.pickle a partir de Mis_Datos/
│   ├── Entrenamiento_SVM.py        Entrena y guarda el modelo SVM (--tipo letras|numeros)
│   ├── Prueba_Letras_con_sonido.py
│   └── Prueba_Numeros_con_sonido.py
└── GUI/
    ├── Main.py                     Punto de entrada de la aplicación
    ├── data_collection_widget.py   Recolección y vectorización de landmarks en vivo (Letras/Números)
    ├── training_widget.py          Entrenamiento y evaluación del modelo
    └── testing_widget.py           Reconocimiento en tiempo real
```

### Uso
1. Instala dependencias: `pip install -r requirements.txt`
2. Desde la carpeta raíz `LSP-ML/`, ejecuta `python GUI/Main.py` (debe ejecutarse con la raíz del proyecto como directorio de trabajo, ya que los íconos e imágenes se referencian con rutas relativas).
3. En **Recolección de Datos**, elige "Letras" o "Números". En **Clase a capturar** selecciona la letra o número por el que quieres empezar (por ejemplo `A`); si vas a capturar varias seguidas, se tomarán las siguientes del diccionario (`A`, `B`, `C`, ...). Define cuántas clases y cuántas capturas (vectores) por clase, y pulsa **Iniciar Camara**. Muestra la seña frente a la cámara: MediaPipe extrae los landmarks de la mano y los vectoriza **en vivo** (no se guardan imágenes). Para señas bimanuales, simplemente muestra ambas manos: el sistema ya las combina automáticamente.
   - Para capturar una **sola** clase (por ejemplo solo `B`), pon "Numero de clases" en `1` y elige `B` en **Clase a capturar**. Repite esto para cada letra que quieras agregar, en el orden que prefieras.
   - El contador en pantalla indica cuántos vectores se llevan capturados de la clase actual. Al alcanzar la cantidad pedida se pasa automáticamente a la siguiente clase.
   - Pulsa **`Q`** para terminar la clase actual antes de tiempo (guarda lo capturado); **`ESC`** cancela la sesión (las clases ya guardadas se conservan).
   - Cada clase se guarda **en cuanto se completa** (o al pulsar `Q`), añadiéndose al `dataset.pickle` existente sin sobrescribirlo. Al terminar la última clase, la cámara se cierra.
4. El botón **Modificar Datos** permite agregar más capturas a una clase ya existente. La clase se toma del selector **Clase a capturar**; si escribes algo en el campo de texto, ese nombre tiene prioridad. También actualiza `dataset.pickle`.
5. **Crear Dataset** ya no es necesario para el flujo normal: solo sirve para vectorizar imágenes `.jpg` que ya estén en `Mis_Datos/` (compatibilidad con el flujo anterior).
6. En **Entrenamiento**, elige el tipo (Letras/Números). El combo de tipo ahora **filtra los datos**: solo se entrena con las clases que correspondan a ese tipo, evitando mezclar letras y números en un mismo modelo. La lista **Clases a incluir en el entrenamiento** muestra las clases que existen en el dataset para ese tipo, todas marcadas; desmarca las que no quieras usar y pulsa **Actualizar clases** si acabas de capturar datos nuevos.
   - Se necesitan **al menos 2 clases marcadas** (un SVM no puede entrenar con una sola); si hay menos, la app lo avisa en lugar de fallar. Cada clase necesita además ≥2 muestras. El número de particiones de la validación cruzada se ajusta automáticamente al tamaño de la clase más pequeña. Revisa el reporte y la matriz de confusión.
   - El nombre del archivo de salida depende del tipo: `svm_letras_lsp_classifier.p` o `svm_numeros_lsp_classifier.p`. Cada vez que entrenas, **se reemplaza** solo el modelo de ese tipo.
7. En **Prueba**, usa "Probar Letras" o "Probar Numeros" para el reconocimiento en tiempo real con salida de audio. El panel muestra el **estado de los modelos** (qué clases tiene cada uno); los modelos se **recargan al abrir esta pantalla** y antes de encender la cámara, así que puedes entrenar y probar **sin reiniciar la aplicación**. La seña predicha se anuncia por el altavoz: si colocas tu propio `Sonidos/<clase>.mp3` se reproduce ese archivo; si no, se dice en voz alta con la voz del sistema (TTS offline). La seña se anuncia al reconocerla y **se repite cada 3 segundos mientras la mantengas**. El botón **Probar Voz** dice una frase de prueba para comprobar el audio sin encender la cámara, y el panel indica el estado de la voz (`Voz: activada (...)` o `Voz: no disponible (falta pyttsx3)`).

> Nota técnica: `pyttsx3` tiene un fallo conocido por el que solo suena la **primera** locución si se reutiliza el mismo motor; por eso `lsp_tts.py` crea un motor nuevo para cada frase (la cola igual se vaciaba, pero no hablaba).

### Nota sobre el contenido lingüístico
Este repositorio **no incluye un dataset de imágenes ni grabaciones de audio de la LSP**: el diseño original de la aplicación es que el propio usuario recolecte su dataset con la cámara (módulo de Recolección de Datos). Para que el sistema sea lingüísticamente válido, se recomienda grabar las señas con la guía de un intérprete certificado o material de referencia de la LSP (por ejemplo, el Diccionario de LSP de la PUCP), en vez de improvisar las formas de la mano.

Si vas a usar la foto de referencia del alfabeto en el botón de ayuda de la interfaz, coloca tu propia imagen en `template.jpg` (raíz del proyecto); el original venía con el afiche del alfabeto árabe, que no aplica aquí.

### Limitaciones y siguiente fase
Este proyecto sigue siendo un reconocedor de **señas estáticas aisladas** (una imagen = una letra o número), igual que el original. No reconoce secuencias continuas ni arma oraciones. Esa es una fase posterior (reconocimiento continuo con CTC + traducción glosa→español), tal como se discutió como siguiente paso de investigación.
