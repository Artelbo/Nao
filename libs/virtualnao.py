from logging import Logger, getLogger
from typing import Dict, Any, Union, Optional
import os
from .stt import STT
from .builtin_shell import Shell, Command, GList
from .ai import Memory
import textwrap
from google import genai
from google.genai import types
from .b_types import Colors, Color
from .colors import FC, OPS
import time
import threading
from thefuzz import fuzz
from protolib.server import MultiConnectionServer, ProtoSocketWrapper
from protolib.client import Session
import queue
import pyaudio

try:
    import pyttsx3  # type: ignore
except ImportError:
    pass


GEMINI_MODEL = 'gemini-2.0-flash'


class VirtualNAO:
    def __init__(self, locale_data: Dict):
        self.__logger: Logger = getLogger('nao')

        self.memory = Memory(capacity=5, locale=locale_data['memory'])
        self.activation_string = 'hey now'  # nao
        self.activation_response = 'Si?'
        self.ai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.system_prompt = textwrap.dedent(locale_data['prompt']).strip()

        self.__shell = Shell('virtualnao/shell')
        self.__shell.add_command(
            Command(
                name='assistant',
                help_doc='assistant [phrase: string] [adjust: int] - Activate the AI assistant with the given string as'
                         ' activation phrase'
            ),
            self.__shell_assistant
        )
        self.__shell.add_command(
            Command(
                name='microphone-adjust',
                help_doc='microphone-adjust <duration: int> - Adjust microphone input on ambient noise'
            ),
            self.__shell_microphone_adjust
        )
        self.__shell.add_command(
            Command(
                name='say',
                help_doc='say <phrase: string> - Says the inputted string'
            ),
            lambda args: self.say(' '.join(args))
        )

        self.stt = STT(max_duration=15, silence_after_speech_threshold=1)
        self.audio = pyaudio.PyAudio()
        self.chunk_size = 1024
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True, )
        threading.Thread(target=self.__threaded_add_to_buffer, daemon=True).start()

        self.tts = pyttsx3.init()
        self.voices = self.tts.getProperty('voices')
        self.tts.setProperty('voice', self.voices[0].id)
        self.tts.setProperty('rate', 150)

        self._mic_adjusted = False

    def __threaded_add_to_buffer(self):
        self.__logger.debug('Starting audio streaming')
        while True:
            self.stt.append_audio(self.stream.read(self.chunk_size, exception_on_overflow=False))

    def __shell_assistant(self, args: GList[str]) -> None:
        self.activation_string = args.get(0, self.activation_string)
        try:
            adjust = int(args.get(1, 3))
        except ValueError:
            print('Invalid adjust value. Expected Int.')
            return

        if not self._mic_adjusted:
            print(f'Adjusting ambient noise ({adjust}s)')
            self.stt.adjust_ambient(adjust)
            self._mic_adjusted = True

        print(f'Starting assistant, activation phrase: "{self.activation_string}"')
        self.__assistant()

    def __shell_microphone_adjust(self, args: GList[str]) -> None:
        try:
            duration = args.get(0, 3)
        except ValueError:
            print('Invalid duration value. Expected Int.')
            return

        print(f'Adjusting ambient noise ({duration}s)')
        self.stt.adjust_ambient(duration)
        self._mic_adjusted = True

    def __stt(self) -> Optional[str]:
        transcribed = self.stt.transcribe()
        if transcribed.successful:
            return transcribed.text
        return None

    def __answer(self, r: str) -> str:
        print(f' {FC.LIGHT_YELLOW}->{OPS.RESET} {r}')

        modified_system_prompt = self.system_prompt.format(
            hour=time.strftime('%H:%M:%S'),
            date=time.strftime('%A, %d %B %Y'),
            memory=self.memory.get_memory_string()
        )

        response = self.ai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=r,
            config=types.GenerateContentConfig(
                system_instruction=modified_system_prompt
            )
        )

        response_text = response.text
        print('\n'.join([f' {FC.LIGHT_MAGENTA}<-{OPS.RESET} {x}' for x in
                         textwrap.wrap(response_text, width=50, break_long_words=False)]),
              end='\n\n', )

        self.memory.add(r, response_text)

        return response_text

    def say(self, s: str) -> None:
        if self.tts:
            try:
                self.tts.say(s)
                self.tts.runAndWait()
            except Exception as e:
                self.__logger.error(f"TTS engine failed to say '{s}': {e}")

    def stop(self):
        pass

    def set_color(self, color: Union[Colors, Color, str], fade_duration: float = 0):
        if isinstance(color, Color):
            self.__logger.info(f"Led Color set to '{color.hex}' in {fade_duration}s")
        elif isinstance(color, Colors):
            self.__logger.info(f"Led Color set to '{color.value}' in {fade_duration}s")
        elif isinstance(color, str):
            self.__logger.info('AllLeds', color, fade_duration)
        else:
            raise ValueError('Invalid Color Type')

    def __assistant(self) -> None:
        while True:
            try:
                text = self.__stt()
                if text:
                    self.__logger.debug(f'Heard: {text}')
                    t = text.lower().replace('ehi', 'hey', 1)
                    if not (fuzz.ratio(t[0:len(self.activation_string)], self.activation_string.lower()) > 50):
                        continue

                    t = t[len(self.activation_string) + 1:].strip()
                    if t.startswith(tuple(',.;:')):
                        t = t[1:]
                        t = t.strip()

                    if len(t) <= 0:
                        print(f' {FC.LIGHT_MAGENTA}<-{OPS.RESET} {self.activation_response}')
                        self.set_color(Colors.RED, 1)
                        self.say(self.activation_response)
                        while True:
                            t = self.__stt()
                            if t:
                                break

                    # print(' -- <request> --')
                    answer = self.__answer(t).strip()
                    self.set_color(Colors.GREEN, 1)
                    self.say(answer)
            except KeyboardInterrupt:
                self.__logger.info('Exiting')
                break

    def start_shell(self):
        self.__shell.run()

    def close(self):
        pass