"""Sintesis de voz (TTS) para decir en voz alta la seña reconocida.

Usa pyttsx3, que habla con el motor de voz del sistema operativo
(SAPI5 en Windows, NSSpeechSynthesizer en macOS, eSpeak en Linux). Es
completamente offline: no necesita internet ni claves de API.

Dos decisiones importantes:

1. Hilo trabajador. El motor de pyttsx3 no es seguro entre hilos y hablar
   bloquea, asi que un unico hilo trabajador es el dueño de la voz y consume
   una cola de textos. Asi la interfaz (y el bucle de video) nunca se congela.

2. Un motor NUEVO por locucion. pyttsx3 tiene un fallo conocido: si se
   reutiliza el mismo motor, solo la PRIMERA llamada a runAndWait() suena de
   verdad; las siguientes vuelven casi al instante sin hablar. Por eso aqui se
   crea un motor para cada frase y se libera al terminar. Si pyttsx3 no esta
   instalado, el modulo queda deshabilitado sin romper la aplicacion.
"""

import gc
import queue
import threading

try:
    import pyttsx3
    _PYTTSX3_DISPONIBLE = True
except ImportError:
    _PYTTSX3_DISPONIBLE = False


def _preferir_voz_espanol(motor):
    """Selecciona una voz en espanol si el sistema tiene alguna instalada.

    Devuelve el nombre de la voz elegida, o None si se usa la de por defecto.
    """
    try:
        voces = motor.getProperty('voices') or []
    except Exception:
        return None

    for voz in voces:
        idiomas = [str(i).lower() for i in (getattr(voz, 'languages', None) or [])]
        nombre = str(getattr(voz, 'name', '') or '').lower()
        if any(i.startswith('es') for i in idiomas) or 'spanish' in nombre or 'espanol' in nombre:
            try:
                motor.setProperty('voice', voz.id)
                return str(getattr(voz, 'name', '') or '')
            except Exception:
                pass
    return None


class Hablante:
    """Dice texto en voz alta sin bloquear el hilo que llama."""

    def __init__(self, rate=160, volumen=1.0):
        self.disponible = _PYTTSX3_DISPONIBLE
        self.voz_nombre = None
        self._rate = rate
        self._volumen = volumen
        self._cola = queue.Queue()
        self._cerrado = False
        self._hilo = None
        if self.disponible:
            self._hilo = threading.Thread(target=self._bucle, name='TTS-LSP', daemon=True)
            self._hilo.start()

    @property
    def descripcion(self):
        """Texto corto del estado de la voz, para mostrarlo en la interfaz."""
        if not self.disponible:
            return 'no disponible (falta pyttsx3)'
        if self.voz_nombre:
            return f'activada ({self.voz_nombre})'
        return 'activada (voz por defecto)'

    def _bucle(self):
        # En Windows, pyttsx3 usa COM (comtypes): cada hilo debe inicializarlo.
        comtypes_iniciado = False
        try:
            import comtypes
            comtypes.CoInitialize()
            comtypes_iniciado = True
        except Exception:
            pass

        # Se comprueba una vez que el motor arranque y se detecta la voz.
        motor_prueba = None
        try:
            motor_prueba = pyttsx3.init()
            self.voz_nombre = _preferir_voz_espanol(motor_prueba)
            print(f'[TTS] voz lista: {self.descripcion}', flush=True)
        except Exception as e:
            print(f'[TTS] no disponible: {e}', flush=True)
            self.disponible = False
        finally:
            if motor_prueba is not None:
                try:
                    motor_prueba.stop()
                except Exception:
                    pass
            del motor_prueba
            gc.collect()

        while self.disponible and not self._cerrado:
            try:
                texto = self._cola.get(timeout=0.2)
            except queue.Empty:
                continue
            if texto is None:
                break
            self._decir_una_vez(texto)

        if comtypes_iniciado:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass

    def _decir_una_vez(self, texto):
        """Dice una frase con un motor recien creado.

        El motor nuevo es obligatorio: reutilizarlo hace que pyttsx3 solo hable
        la primera vez (ver la nota del modulo).
        """
        motor = None
        try:
            motor = pyttsx3.init()
            motor.setProperty('rate', self._rate)
            motor.setProperty('volume', self._volumen)
            _preferir_voz_espanol(motor)
            print(f'[TTS] diciendo: {texto}', flush=True)
            motor.say(texto)
            motor.runAndWait()
        except Exception as e:
            print(f'[TTS] error al hablar: {e}', flush=True)
        finally:
            if motor is not None:
                try:
                    motor.stop()
                except Exception:
                    pass
            del motor
            gc.collect()

    def decir(self, texto):
        """Encola un texto para decirlo en voz alta.

        Descarta los textos aun no pronunciados: solo interesa la ultima seña
        reconocida, no acumular repeticiones atrasadas.
        """
        if not self.disponible or texto is None:
            return
        while True:
            try:
                self._cola.get_nowait()
            except queue.Empty:
                break
        self._cola.put(str(texto))

    def cerrar(self):
        """Detiene el hilo trabajador y libera la voz."""
        if self._cerrado:
            return
        self._cerrado = True
        if self._hilo is not None:
            self._cola.put(None)
            self._hilo.join(timeout=3.0)


if __name__ == '__main__':
    import sys
    import time

    hablante = Hablante()
    print('TTS disponible:', hablante.disponible)
    textos = sys.argv[1:] or ['A', 'B', 'E']
    for texto in textos:
        print('Diciendo:', texto)
        hablante.decir(texto)
        time.sleep(1.5)
    hablante.cerrar()
