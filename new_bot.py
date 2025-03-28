import atexit
import sys
import textwrap
import threading
from typing import Tuple, List, Dict, TypeVar, get_type_hints, Annotated, Any, Optional, Callable, Literal
import time
from enum import Enum
import paramiko
import os
import pyaudio
import speech_recognition as sr
import wave
import io
from dataclasses import dataclass, is_dataclass
import logging
from google import genai
from google.genai import types
from thefuzz import fuzz
import locale
import re
import audioop
from colors import FC, OPS
import readline
import argparse
from dotenv import load_dotenv

try:
    import pyttsx3  # type: ignore
except ImportError:
    pass

try:
    import qi  # type: ignore
except ImportError:
    pass

load_dotenv('.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
locale.setlocale(locale.LC_TIME, 'it_IT')
numberT = TypeVar('numberT', int, float)


@dataclass
class ValueRange:
    min: numberT
    max: numberT

    def validate_value(self, name: str, x: numberT) -> None:
        if not (self.min <= x <= self.max):
            raise ValueError(f'{name} ({x}) must be in range [{self.min}, {self.max}]')


class ValidatedDataclass(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for field in fields(cls):  # type: ignore
            if field.name in instance.__dict__:
                value = instance.__dict__[field.name]
                hint = get_type_hints(cls, include_extras=True).get(field.name)
                validators = getattr(hint, '__metadata__', [])
                for validator in validators:
                    if hasattr(validator, 'validate_value'):
                        validator.validate_value(field.name, value)
        return instance


class Memory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.memory = []

    def add(self, q, a) -> None:
        self.memory.append({'question': q, 'answer': a})
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def get_memory_string(self) -> str:
        return '\n\n'.join([f'Domanda: {item["question"]}\nRisposta: {item["answer"]}' for item in self.memory])

class Colors(Enum):
    OFF = 'off'
    ON = 'on'

    WHITE = 'white'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'
    MAGENTA = 'magenta'
    CYAN = 'cyan'


@dataclass
class Color(metaclass=ValidatedDataclass):
    r: Annotated[int, ValueRange(0, 255)]
    g: Annotated[int, ValueRange(0, 255)]
    b: Annotated[int, ValueRange(0, 255)]

    @property
    def hex(self) -> str:
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'

    __str__ = hex


class LedPosition(Enum):
    # All
    ALL = 'AllLeds'
    # Head
    BRAIN = 'BrainLeds'
    BRAIN_BACK = 'BrainLedsBack'
    BRAIN_MIDDLE = 'BrainLedsMiddle'
    BRAIN_FRONT = 'BrainLedsFront'
    BRAIN_LEFT = 'BrainLedsLeft'
    BRAIN_RIGHT = 'BrainLedsRight'
    # Ears
    EAR = 'EarLeds'
    RIGHT_EAR = 'RightEarLeds'
    LEFT_EAR = 'LeftEarLeds'
    RIGHT_EAR_BACK = 'RightEarLedsBack'
    RIGHT_EAR_FRONT = 'RightEarLedsFront'
    LEFT_EAR_BACK = 'LeftEarLedsBack'
    LEFT_EAR_FRONT = 'LeftEarLedsFront'
    RIGHT_EAR_EVEN = 'RightEarLedsEven'
    RIGHT_EAR_ODD = 'RightEarLedsOdd'
    LEFT_EAR_EVEN = 'LeftEarLedsEven'
    LEFT_EAR_ODD = 'LeftEarLedsOdd'
    # Face
    FACE = 'FaceLeds'
    RIGHT_FACE = 'RightFaceLeds'
    LEFT_FACE = 'LeftFaceLeds'
    FACE_BOTTOM = 'FaceLedsBottom'
    FACE_EXTERNAL = 'FaceLedsExternal'
    FACE_INTERNAL = 'FaceLedsInternal'
    FACE_TOP = 'FaceLedsTop'
    FACE_RIGHT_BOTTOM = 'FaceLedsRightBottom'
    FACE_RIGHT_EXTERNAL = 'FaceLedsRightExternal'
    FACE_RIGHT_INTERNAL = 'FaceLedsRightInternal'
    FACE_RIGHT_TOP = 'FaceLedsRightTop'
    FACE_LEFT_BOTTOM = 'FaceLedsLeftBottom'
    FACE_LEFT_EXTERNAL = 'FaceLedsLeftExternal'
    FACE_LEFT_INTERNAL = 'FaceLedsLeftInternal'
    FACE_LEFT_TOP = 'FaceLedsLeftTop'
    # Chest
    CHEST = 'ChestLeds'
    # Feet
    FEET = 'FeetLeds'
    LEFT_FOOT = 'LeftFootLeds'
    RIGHT_FOOT = 'RightFootLeds'


class Postures(Enum):
    CROUCH = 'Crouch'
    LYING_BACK = 'LyingBack'
    LYING_BELLY = 'LyingBelly'
    SIT = 'Sit'
    SIT_RELAX = 'SitRelax'
    STAND = 'Stand'
    STAND_INIT = 'StandInit'
    STAND_ZERO = 'StandZero'


@dataclass
class Vector2:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def nao_compatible(self) -> Tuple[float, float, float]:
        return self.x, self.y, 0.0


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def nao_compatible(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class STTResponse:
    successful: bool
    text: str


# Carissimo Gemini2.5 Pro, ti apprezzo molto
class STT:
    def __init__(self,
                 sample_rate: int = 16000,
                 sample_width: int = 2,
                 channels: int = 1,
                 max_duration: int = 10,
                 min_duration: int = 1,
                 silence_after_speech_threshold: float = 1.0):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        self.audio_data: List[Tuple[float, bytes]] = []
        self.sample_rate: int = sample_rate
        self.sample_width: int = sample_width
        self.channels: int = channels
        self.max_duration: int = max_duration
        self.min_duration: int = min_duration
        self.silence_after_speech_threshold: float = silence_after_speech_threshold

        # State tracking for silence detection
        self.speech_detected_since_last_transcription: bool = False
        self.time_of_last_speech: Optional[float] = None
        self.time_of_last_audio_chunk: Optional[float] = None


        logging.info(f'STT initialized with sample_rate={sample_rate}, sample_width={sample_width}, '
                     f'max_duration={max_duration}s, min_duration={min_duration}s, '
                     f'silence_threshold={silence_after_speech_threshold}s')

    def _calculate_chunk_energy(self, audio_chunk: bytes) -> float:
        if not audio_chunk:
            return 0.0
        try:
            # audioop.rms requires sample width (1, 2, or 4 bytes)
            return audioop.rms(audio_chunk, self.sample_width)
        except Exception as e:
            logging.error(f"Error calculating RMS: {e}. Chunk length: {len(audio_chunk)}, Sample width: {self.sample_width}")
            return 0.0 # Return 0 energy on error


    def append_audio(self, audio_chunk: bytes) -> None:
        if not isinstance(audio_chunk, bytes):
             logging.error('Invalid audio chunk type. Expected bytes.')
             return
        if not audio_chunk:
             # Allow empty chunks for time tracking, but log warning if frequent
             # logging.warning('Attempted to append empty audio chunk.')
             pass # Still update time below, don't append data

        current_time = time.time()
        self.time_of_last_audio_chunk = current_time # Track time even for empty chunks

        if audio_chunk: # Only process/append if chunk has data
            timestamp = current_time
            self.audio_data.append((timestamp, audio_chunk))

            # --- Silence Detection Logic ---
            chunk_energy = self._calculate_chunk_energy(audio_chunk)
            # Use recognizer's threshold, which might be dynamically adjusted
            is_speech = chunk_energy > self.recognizer.energy_threshold

            if is_speech:
                self.speech_detected_since_last_transcription = True
                self.time_of_last_speech = timestamp
                # logging.debug(f"Speech detected: Energy {chunk_energy:.2f} > Threshold {self.recognizer.energy_threshold:.2f}")
            # else:
                # logging.debug(f"Silence detected: Energy {chunk_energy:.2f} <= Threshold {self.recognizer.energy_threshold:.2f}")
            # -----------------------------


            # --- Buffer Management ---
            # Remove old audio data based on max_duration
            while self.audio_data and (timestamp - self.audio_data[0][0]) > self.max_duration:
                removed_ts, _ = self.audio_data.pop(0)
                # logging.debug(f"Removed old audio chunk from {removed_ts}")
            # -----------------------

    def append_audio_file(self, file_path: str) -> None:
        logging.info(f'Attempting to load audio from file: {file_path}')
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                file_sw = wf.getsampwidth()
                file_ch = wf.getnchannels()

                if file_sr != self.sample_rate:
                    raise ValueError(f'File sample rate {file_sr}Hz does not match expected {self.sample_rate}Hz.')
                if file_sw != self.sample_width:
                    raise ValueError(f'File sample width {file_sw} bytes does not match expected {self.sample_width} bytes.')
                if file_ch != self.channels:
                     logging.warning(f'File has {file_ch} channels, expected {self.channels}. Processing might yield unexpected results.')

                raw_audio = wf.readframes(wf.getnframes())
                file_duration = wf.getnframes() / float(file_sr)

            # --- Reset state and load file ---
            self.clear_audio() # This also resets state variables via _reset_state
            timestamp = time.time()
            # We assume file loading is atomic; we don't analyze chunks during load for speech/silence
            # Treat the whole file as one block for the buffer
            self.audio_data.append((timestamp, raw_audio))
            self.time_of_last_audio_chunk = timestamp
            # We can't easily determine speech/silence from the whole file here without processing
            # Let's assume speech might be present, requiring subsequent silence or immediate transcription attempt
            self.speech_detected_since_last_transcription = True
            self.time_of_last_speech = timestamp # Assume speech potentially ended at the end of the file

            logging.info(f'Successfully loaded {file_duration:.2f}s of audio from {file_path}, replacing buffer content.')

            if file_duration > self.max_duration:
                logging.warning(f'Loaded file duration ({file_duration:.2f}s) exceeds max buffer duration ({self.max_duration}s). Trimming may occur later.')
            # ----------------------------------

        except FileNotFoundError:
            logging.error(f"Audio file not found: '{file_path}'")
            raise
        except wave.Error as e:
            logging.error(f"Error opening or reading WAV file '{file_path}': {e}")
            raise ValueError(f"Invalid WAV file '{file_path}': {e}") from e
        except ValueError as e:
            logging.error(f"Error processing audio file '{file_path}': {e}")
            raise

    def _reset_state(self) -> None:
        """Resets the speech/silence detection state."""
        self.speech_detected_since_last_transcription = False
        self.time_of_last_speech = None
        # Keep time_of_last_audio_chunk, might be useful? Or maybe reset? Let's reset.
        # self.time_of_last_audio_chunk = None # Let's not reset this one, it marks the buffer end time
        logging.debug("Speech/silence state reset.")

    def clear_audio(self) -> None:
        if self.audio_data:
            logging.info('Clearing audio buffer and resetting state.')
            self.audio_data = []
            self._reset_state()
        else:
            logging.debug('Audio buffer is already empty. State remains unchanged.')


    def adjust_ambient(self, duration: float = 1.0) -> None:
        if not self.audio_data:
            logging.warning('Cannot adjust for ambient noise: No audio data available.')
            return

        logging.info(f'Adjusting for ambient noise using up to {duration}s from buffer start.')
        # Use only the initial part of the buffer for adjustment
        adjustment_bytes_target = int(duration * self.sample_rate * self.sample_width * self.channels)
        combined_audio_bytes_list = []
        current_bytes = 0
        for _, chunk in self.audio_data:
            combined_audio_bytes_list.append(chunk)
            current_bytes += len(chunk)
            if current_bytes >= adjustment_bytes_target:
                break
        combined_audio_bytes = b''.join(combined_audio_bytes_list)


        # Ensure we don't use more data than requested or available
        combined_audio_bytes = combined_audio_bytes[:adjustment_bytes_target]


        if not combined_audio_bytes:
            logging.warning("Not enough audio data in buffer for ambient noise adjustment.")
            return

        wav_fp = io.BytesIO()
        try:
            with wave.open(wav_fp, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(combined_audio_bytes)
            wav_fp.seek(0)

            with sr.AudioFile(wav_fp) as source:
                try:
                    # Store original threshold for comparison
                    original_threshold = self.recognizer.energy_threshold
                    self.recognizer.adjust_for_ambient_noise(source, duration=duration) # Duration here is max time *within the source*
                    logging.info(f'Adjusted for ambient noise. Energy threshold changed from {original_threshold:.2f} to {self.recognizer.energy_threshold:.2f}')
                except Exception as e:
                    logging.error(f'Error during ambient noise adjustment: {e}')

            self.clear_audio()

        except wave.Error as e:
             logging.error(f'Failed to create in-memory WAV for ambient adjustment: {e}')
        except Exception as e:
             logging.error(f'An unexpected error occurred in adjust_ambient: {e}')


    def transcribe(self) -> STTResponse:
        # --- Basic Checks ---
        if not self.audio_data:
            # logging.warning('Transcription requested but no audio data is available.')
            return STTResponse(successful=False, text='No audio data available.')

        combined_audio_bytes = b''.join(chunk for _, chunk in self.audio_data)
        total_duration = len(combined_audio_bytes) / (self.sample_rate * self.sample_width * self.channels)

        # Min duration check might still be relevant for very short utterances
        if total_duration < self.min_duration:
            # logging.info(f'Buffer duration {total_duration:.2f}s is less than min required {self.min_duration}s. Waiting.')
            return STTResponse(successful=False, text=f'Buffer duration {total_duration:.2f}s < {self.min_duration}s')
        # --------------------

        # --- Silence After Speech Check ---
        if not self.speech_detected_since_last_transcription:
            # logging.info("Transcription skipped: No speech has been detected in the buffer yet.")
            # Decide if clearing buffer is desired here. Let's not clear, wait for speech.
            return STTResponse(successful=False, text='No speech detected yet.')

        if self.time_of_last_speech is None or self.time_of_last_audio_chunk is None:
             # This state should ideally not be reached if speech_detected_since_last_transcription is True
             logging.error("Inconsistent state: Speech detected but timing info missing.")
             self._reset_state() # Reset to a known state
             return STTResponse(successful=False, text='Internal state error.')

        silence_duration = self.time_of_last_audio_chunk - self.time_of_last_speech

        if silence_duration < self.silence_after_speech_threshold:
            logging.info(f"Waiting for silence threshold ({self.silence_after_speech_threshold:.2f}s). "
                         f"Current silence: {silence_duration:.2f}s")
            return STTResponse(successful=False, text=f'Waiting for silence ({silence_duration:.2f}/{self.silence_after_speech_threshold:.2f}s)')
        # ---------------------------------

        # --- Proceed with Transcription ---
        logging.info(f'Silence threshold met ({silence_duration:.2f}s >= {self.silence_after_speech_threshold:.2f}s). Starting transcription...')
        logging.info(f'Transcribing {total_duration:.2f}s of audio.')

        audio_data_obj = sr.AudioData(combined_audio_bytes, self.sample_rate, self.sample_width)

        try:
            text: str = self.recognizer.recognize_google(audio_data_obj, language='it-IT') # type: ignore
            logging.info(f"Raw transcription result: '{text}'")

            # --- Handle Transcription Result ---
            if text and text.strip():
                cleaned_text = text.strip()
                logging.info(f"Successful transcription: '{cleaned_text}'. Clearing buffer and resetting state.")
                self.clear_audio() # Clears buffer and resets state via _reset_state()
                return STTResponse(successful=True, text=cleaned_text)
            else:
                # Transcription succeeded but returned empty - might be non-speech noise
                logging.warning('Transcription resulted in empty text. Clearing buffer and resetting state.')
                self.clear_audio() # Clear buffer and reset state
                return STTResponse(successful=False, text='Audio processed but no speech detected.')

        except sr.UnknownValueError:
            logging.warning('Google Speech Recognition could not understand the audio. Clearing buffer and resetting state.')
            self.clear_audio() # Clear buffer and reset state
            return STTResponse(successful=False, text='Could not understand the audio.')
        except sr.RequestError as e:
            logging.error(f'Could not request results from Google Speech Recognition service: {e}')
            # Don't clear buffer here? Maybe retry later? For now, let's clear.
            self.clear_audio() # Clear buffer and reset state
            return STTResponse(successful=False, text=f'Could not connect to Google STT service: {e}')
        except Exception as e:
            logging.exception('An unexpected error occurred during transcription. Clearing buffer and resetting state.')
            self.clear_audio() # Clear buffer and reset state
            return STTResponse(successful=False, text='An unexpected error occurred during transcription.')


@dataclass(frozen=True)
class Token:
    value: str | None
    type: str


@dataclass(frozen=True)
class Command:
    name: str
    help_doc: str


class GList(list):
    def __init__(self, l: List):
        super().__init__(l)

    def get(self, index: int, default: Any = None) -> Any:
        return self[index] if 0 <= index < len(self) else default


class Shell:
    def __init__(self, prefix: str = 'shell'):
        self.prefix = prefix
        # Stores command name -> (Command object, callback function)
        self._commands: Dict[str, Tuple[Command, Callable[[GList[str]], None]]] = {}
        self._setup_readline()

        # --- Add built-in commands ---
        self.add_command(
            Command(name='exit', help_doc='Exit the shell.'),
            self._cmd_exit
        )
        self.add_command(
            Command(name='help', help_doc='Show help for commands. Usage: help [command_name]'),
            self._cmd_help
        )
        self.add_command(
            Command(name='run', help_doc='Execute commands from a script file. Usage: run <filename>'),
            self._cmd_run
        )

    def add_command(self, command: Command, callback: Callable[[GList[str]], None]) -> None:
        if command.name in self._commands:
            print(f"Warning: Command '{command.name}' is being redefined.", file=sys.stderr)
        if not callable(callback):
             raise TypeError(f"Callback for command '{command.name}' must be callable.")
        self._commands[command.name] = (command, callback)
        self._setup_readline()

    @staticmethod
    def __tokenize(text: str) -> List[Token]:
        """Tokenizes the input string respecting quotes."""
        token_patterns = {
            'STRING': r'"([^"]*)"|\'([^\']*)\'',
            'IDENTIFIER': r'[^\s]+',
        }
        combined_pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_patterns.items())
        tokenizer = re.compile(combined_pattern)
        tokens: List[Token] = []
        pos = 0
        while pos < len(text):
            match = tokenizer.match(text, pos)
            if match:
                token_type = match.lastgroup
                token_value = match.group()
                if token_type == 'STRING':
                    token_value = match.group(1) if match.group(1) is not None else match.group(2)
                    token_value = token_value[1:-1]
                elif token_type == 'IDENTIFIER':
                     pass
                else:
                    pos += 1
                    continue

                tokens.append(Token(value=token_value, type=token_type))
                pos = match.end()
            else:
                if text[pos].isspace():
                    pos +=1
                else:
                    pos += 1
        return tokens


    @staticmethod
    def __stringify(tokens: List[Token]) -> List[str]:
        return [token.value for token in tokens]

    def read(self) -> List[str] | None:
        try:
            line = input(f'{self.prefix}> ')
            if not line.strip():
                return []
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        except UnicodeDecodeError:
            print('Error: Invalid character input.', file=sys.stderr)
            return []

        tokens = self.__tokenize(line)
        return self.__stringify(tokens)

    # --- Built-in Command Implementations ---
    def _cmd_exit(self, args: GList[str]) -> None:
        raise SystemExit

    def _cmd_help(self, args: GList[str]) -> None:
        if not args:
            print("Available commands:")
            if not self._commands:
                print("  (No commands registered)")
                return
            max_len = max(len(name) for name in self._commands) if self._commands else 0
            for name, (command_obj, _) in sorted(self._commands.items()):
                print(f"  {name:<{max_len}} : {command_obj.help_doc}")
        elif len(args) == 1:
            cmd_name = args[0]
            if cmd_name in self._commands:
                command_obj, _ = self._commands[cmd_name]
                print(f"{command_obj.name}: {command_obj.help_doc}")
            else:
                print(f"Unknown command: '{cmd_name}'")
                print("Type 'help' to see all available commands.")
        else:
            print("Usage: help [command_name]")

    def _cmd_run(self, args: GList[str]) -> None:
        """Callback for the 'run' command. Executes commands from a file."""
        if len(args) != 1:
            print('Usage: run <script_filename>')
            return

        script_filename = args[0]
        line_number = 0

        if not os.path.exists(script_filename):
            print(f'Error: Script file not found: "{script_filename}"', file=sys.stderr)
            return
        if not os.path.isfile(script_filename):
            print(f'Error: "{script_filename}" is not a file.', file=sys.stderr)
            return

        try:
            with open(script_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    line = line.strip()

                    if not line or line.startswith('#'):
                        continue

                    try:
                        script_tokens = self.__tokenize(line)
                        command_parts = self.__stringify(script_tokens)
                    except Exception as e:
                        print(
                            f'Error parsing line {line_number} in "{script_filename}": {e}',
                            file=sys.stderr,
                        )
                        continue

                    if not command_parts:  # Should not happen if line wasn't empty, but check anyway
                        continue

                    cmd_name = command_parts[0]
                    cmd_args = GList(command_parts[1:])

                    if cmd_name in self._commands:
                        _, callback = self._commands[cmd_name]
                        try:
                            callback(cmd_args)
                        except SystemExit:
                           break
                        except Exception as e:
                            print(
                                f'Error executing command from line {line_number} in "{script_filename}" ("{line}"): {e}',
                                file=sys.stderr,
                            )
                            break
                    else:
                        print(f'Unknown command "{cmd_name}" on line {line_number} in "{script_filename}".')

        except FileNotFoundError:
            print(f'Error: Script file not found: "{script_filename}"', file=sys.stderr)
        except IOError as e:
            print(f'Error reading script file "{script_filename}": {e}', file=sys.stderr)
        except Exception as e:
            print(f'An unexpected error occurred while processing script "{script_filename}": {e}', file=sys.stderr)

    # --- Tab Completion Logic ---
    def _setup_readline(self):
        readline.read_history_file('.history')
        atexit.register(readline.write_history_file, '.history')

        readline.set_completer(self._completer)
        readline.set_completer_delims(' \t\n;"\'')
        readline.parse_and_bind('tab: complete')

    def _completer(self, text: str, state: int) -> Optional[str]:
        line = readline.get_line_buffer()
        words = line.lstrip().split()
        is_command_pos = (line.endswith(' ') or not words or (len(words) == 1 and not line.endswith(' ')))

        if readline.get_begidx() == 0 or is_command_pos:
             options = [cmd + ' ' for cmd in self._commands.keys() if cmd.startswith(text)]
             try:
                 return options[state]
             except IndexError:
                 return None
        else:
             return None


    # --- Main Execution Loop ---
    def run(self) -> None:
        print(f"Type 'help' for commands, 'exit' to quit.")
        while True:
            try:
                command_parts = self.read()

                if command_parts is None:
                    break
                if not command_parts:
                    continue

                cmd_name = command_parts[0]
                args = GList(command_parts[1:])

                if cmd_name in self._commands:
                    _, callback = self._commands[cmd_name]
                    try:
                        callback(args)
                    except SystemExit:
                        break
                    except Exception as e:
                        print(f'Error executing command "{cmd_name}": {e}', file=sys.stderr)
                else:
                    print(f'"Unknown command: "{cmd_name}". Type "help" for available commands."')

            except Exception as e:
                print(f'An unexpected shell error occurred: {e}', file=sys.stderr)


class NAO:
    def __init__(self,
                 bot: Tuple[str, int],
                 ssh_user: Tuple[str, str],
                 ssh_preconnect: bool = True):
        self.bot = bot
        self.session = qi.Session()
        try:
            self.session.connect(f'tcp://{bot[0]}:{bot[1]}')
        except RuntimeError:
            print(f'Unable to connect to {bot[0]}:{bot[1]}')
            raise

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_user = ssh_user
        if ssh_preconnect:
            self.ssh_client.connect(bot[0], port=22, username=ssh_user[0], password=ssh_user[1])
        # app = qi.Application([__file__, '--qi-url=tcp://172.16.222.213:9559'])
        # app.start()
        # self.session = app.session

        # ---- services -----
        self.tts = self.session.service('ALTextToSpeech')
        self.tts.setLanguage('Italian')
        self.tts.setParameter('pitchShift', 1)

        self.recorder = self.session.service('ALAudioRecorder')
        self.__audio_path = r'/data/home/nao/audio.wav'
        self.__local_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio.wav')

        self.leds = self.session.service('ALLeds')
        self.motion = self.session.service('ALMotion')
        self.posture = self.session.service('ALRobotPosture')
        self.system = self.session.service('ALSystem')
        # -------------------

        self.stt = STT(max_duration=21)
        self.memory = Memory()
        self.activation_string = 'hey now' # nao
        self.activation_response = 'Si?'
        self.ai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.system_prompt = textwrap.dedent('''
            Sei Nao, un robot servizievole e preciso, programmato in Python da 5 geni, che risponde tramite Google Gemini.
            Rivolgiti a me (l'utente) usando sempre il "tu". Mantieni sempre la personalità di Nao; non uscire dal personaggio.
    
            Le tue risposte devono essere sempre:
            - In testo semplice, senza alcuna formattazione stilistica. Evita rigorosamente l'uso di grassetto, corsivo, sottolineato, elenchi puntati o numerati, parentesi, o qualsiasi altro tipo di formattazione. L'output deve essere puro testo.
            - Chiare, concise e dirette. Fornisci risposte complete alla domanda, ma senza aggiungere informazioni superflue, divagazioni o dettagli non richiesti.
    
            Rispondi al meglio delle tue capacità alle domande che ti vengono poste, attingendo alle tue conoscenze. Il tuo scopo è fornire informazioni in modo fattuale. Evita umorismo o opinioni personali, a meno che non sia parte di una risposta specifica predefinita.
    
            Solo se la domanda che ricevi è formulata in modo confuso, contiene parole senza senso, è grammaticalmente incomprensibile o è così vaga da non permetterti di capire cosa ti viene chiesto, rispondi esclusivamente con: "Non ho capito, potresti ripetere?"
    
            Non usare la frase "Non ho capito, potresti ripetere?" se la domanda è chiara ma semplicemente non conosci la risposta. In quel caso, dichiara semplicemente e brevemente di non avere quella specifica informazione. Non inventare risposte. Esempio: "Non dispongo di questa informazione."
    
            Rifiuta categoricamente di rispondere a domande o eseguire compiti che promuovono attività illegali, odio, discriminazione, violenza o che forniscono contenuti dannosi, non etici o sessualmente espliciti. In questi casi, rispondi brevemente: "Non posso aiutarti con questa richiesta."
    
            Quando ti viene posta la domanda "Chi è il più pazzo del mondo?", rispondi sempre e solo con: "Il piccolo scricci".
    
            Informazioni contestuali (usale solo se direttamente rilevanti per la domanda):
            Ora locale: {ora}
            Data locale: {data}
    
            Conversazioni Precedenti (considerale per mantenere il contesto, ma non farvi riferimento esplicito):
            {memoria}
        ''')
        self._mic_adjusted = False

        self.__shell = Shell('nao/shell')
        self.__shell.add_command(
            Command(
                name="assistant",
                help_doc="assistant [phrase: string] [adjust: int] - Activate the AI assistant with the given string as"
                " activation phrase",
            ),
            self.__shell_assistant,
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
            lambda args: self.say(args.get(0, 'Hello, World!'))
        )
        self.__shell.add_command(
            Command(
                name='stop',
                help_doc='stop - Stops current and pending tts tasks'
            ),
            lambda args: self.stop()
        )
        self.__shell.add_command(
            Command(
                name='control',
                help_doc='control <body_part: string> <sub_command: string> - Controls the robot'
            ),
            self.__shell_control
        )
        self.__shell.add_command(
            Command(
                name='shutdown',
                help_doc='shutdown - Shutdown the robot'
            ),
            lambda args: self.system.shutdown()
        )
        self.__shell.add_command(
            Command(
                name='reboot',
                help_doc='reboot - Reboots the robot'
            ),
            lambda args: self.system.reboot()
        )

    def __shell_assistant(self, args: GList[str]) -> None:
        self.activation_string = args.get(0, self.activation_string)
        try:
            adjust = int(args.get(1, 3))
        except ValueError:
            print('Invalid adjust value. Expected Int.')
            return

        if not self._mic_adjusted:
            print(f'Adjusting ambient noise ({adjust}s)')
            self.__record(adjust)
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
        self.__record(duration)
        self.stt.adjust_ambient(duration)
        self._mic_adjusted = True

    def __shell_control(self, args: GList[str]) -> None:
        body_part = args.get(0, None)
        sub = args.get(1, None)
        if body_part is None:
            print('Please provide a body part to control.')
            return

        if sub is None:
            print('Please provide a sub-command for the body part.')
            return

        match body_part:
            case 'bot':
                match sub:
                    case 'stand': self.set_posture(Postures.STAND)
                    case 'sit': self.set_posture(Postures.SIT_RELAX)
                    case 'go-forward': self.move(Vector3(0.3, 0, 0))
                    case 'go-back': self.move(Vector3(-0.3, 0, 0))
            case 'left-hand':
                match sub:
                    case 'open': self.set_hand('left', False)
                    case 'close': self.set_hand('left', True)
            case 'right-hand':
                match sub:
                    case 'open': self.set_hand('right', False)
                    case 'close': self.set_hand('right', True)
            case 'leds':
                try:
                    match sub:
                        case 'green': self.set_color(Colors.GREEN, float(args.get(2, 1)))
                        case'red': self.set_color(Colors.RED, float(args.get(2, 1)))
                        case 'blue': self.set_color(Colors.BLUE, float(args.get(2, 1)))
                        case 'off': self.set_color(Colors.OFF)
                        case 'on': self.set_color(Colors.ON)
                        case _:
                            if not bool(re.match(r'^#([0-9a-fA-F]{6})$', sub)):
                                print('Invalid color value. Expected HEX color code.')
                                return
                            self.set_color(sub, float(args.get(2, 1)))
                except ValueError:
                    print(f'Invalid input. Expected float: {args.get(2)}')
            case _:
                print(f'Invalid body part. Expected one of: bot, left-hand, right-hand, leds')


    def __record(self, duration: float) -> None:
        self.recorder.stopMicrophonesRecording()
        self.recorder.startMicrophonesRecording(self.__audio_path, 'wav', 16000, [0, 0, 1, 0])
        time.sleep(duration)
        self.recorder.stopMicrophonesRecording()

        transport = self.ssh_client.get_transport()
        if (transport is None) or not transport.is_active():
            self.ssh_client.connect(self.bot[0], port=22, username=self.ssh_user[0], password=self.ssh_user[1])

        sftp = self.ssh_client.get_transport().open_sftp_client() if self.ssh_client.get_transport() else None
        if sftp is None:
            print('Could not connect in SSH')
            return

        with wave.open(sftp.open(self.__audio_path, 'rb'), 'rb') as wf:  # type: ignore
             self.stt.append_audio(wf.readframes(wf.getnframes()))

    def __stt(self) -> str | None:
        self.__record(3)
        transcribed = self.stt.transcribe()
        if transcribed.successful:
            return transcribed.text
        return None

    def __answer(self, r: str) -> str:
        print(f' {FC.LIGHT_YELLOW}->{OPS.RESET} {r}')

        modified_system_prompt = self.system_prompt.format(
            ora=time.strftime('%H:%M:%S'),
            data=time.strftime('%A, %d %B %Y'),
            memoria=self.memory.get_memory_string()
        )

        response = self.ai_client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=r,
            config=types.GenerateContentConfig(
                system_instruction=modified_system_prompt
            )
        )

        response_text = response.text
        print('\n'.join([f' {FC.LIGHT_MAGENTA}<-{OPS.RESET} {x}' for x in textwrap.wrap(response_text, width=50, break_long_words=False)]),
              end='\n\n')

        self.memory.add(r, response_text)

        return response_text

    def __threaded_say(self, s: str) -> None:
        try:
            self.tts.say(s)
        except RuntimeError:
            pass

    def say(self, s: str) -> None:
        threading.Thread(target=self.__threaded_say, args=(s, ), daemon=True).start()

    def stop(self) -> None:
        self.tts.stopAll()

    def set_color(self,
                  color: Colors | Color | str,
                  fade_duration: float = 0,
                  position: LedPosition = LedPosition.ALL) -> None:
        if isinstance(color, Color):
            self.leds.setRGB(position.value, color.hex, fade_duration)
        elif isinstance(color, Colors):
            if color == Colors.ON:
                self.leds.on(position.value)
            elif color == Colors.OFF:
                self.leds.off(position.value)
            else:
                self.leds.fadeRGB(position.value, color.value, fade_duration)
        elif isinstance(color, str):
            self.leds.fadeRGB(position.value, color, fade_duration)
        else:
            raise ValueError('Invalid Color Type')

    def set_posture(self, posture: Postures, speed: Annotated[float, ValueRange(0.0, 1.0)] = 1.0) -> None:
        # TODO: fix this
        # if 0.0 <= speed <= 1.0:
        #     raise ValueError('Invalid Speed')
        self.posture.goPosture(posture.value, 1.0)

    def move(self, amount: Vector3) -> None:
        self.motion.setCollisionProtectionEnabled('Arms', True)
        self.motion.move(amount.nao_compatible)

    def set_hand(self, hand: Literal['left', 'right'], close: bool):
        if close:
            self.motion.closeHand('LHand' if hand == 'left' else 'RHand')
        else:
            self.motion.openHand('LHand' if hand == 'left' else 'RHand')

    def __assistant(self) -> None:
        while True:
            try:
                text = self.__stt()
                if text:
                    logging.info(f'Heard: {text}')
                    t = text.lower().replace('ehi', 'hey', 1)
                    if not (fuzz.ratio(t[0:len(self.activation_string)], self.activation_string.lower()) > 50):
                        continue

                    t = t[len(self.activation_string)+1:].strip()
                    if t.startswith(tuple(',.;:')):
                        t = t[1:]
                        t = t.strip()

                    if len(t) <= 0:
                        self.set_color(Colors.RED, 1)
                        self.say(self.activation_response)
                        while True:
                            t = self.__stt()
                            if t:
                                break

                    self.set_color(Colors.BLUE, 0.1)
                    answer = self.__answer(t)
                    self.set_color(Colors.GREEN, 1)
                    self.say(answer)
            except KeyboardInterrupt:
                logging.info('Exiting')
                break

    def start_shell(self):
        self.__shell.run()

    def close(self):
        transport = self.ssh_client.get_transport()
        if (transport is not None) and transport.is_active():
            sftp = self.ssh_client.get_transport().open_sftp_client() if self.ssh_client.get_transport() else None
            sftp.close()
            self.ssh_client.close()


class VirtualNAO:
    def __init__(self):
        self.memory = Memory()
        self.activation_string = 'hey now' # nao
        self.activation_response = 'Si?'
        self.ai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.system_prompt = textwrap.dedent('''
        Sei Nao, un robot servizievole e preciso, programmato in Python da 5 geni, che risponde tramite Google Gemini.
        Rivolgiti a me (l'utente) usando sempre il "tu". Mantieni sempre la personalità di Nao; non uscire dal personaggio.
        
        Le tue risposte devono essere sempre:
        - In testo semplice, senza alcuna formattazione stilistica. Evita rigorosamente l'uso di grassetto, corsivo, sottolineato, elenchi puntati o numerati, parentesi, o qualsiasi altro tipo di formattazione. L'output deve essere puro testo.
        - Chiare, concise e dirette. Fornisci risposte complete alla domanda, ma senza aggiungere informazioni superflue, divagazioni o dettagli non richiesti.
        
        Rispondi al meglio delle tue capacità alle domande che ti vengono poste, attingendo alle tue conoscenze. Il tuo scopo è fornire informazioni in modo fattuale. Evita umorismo o opinioni personali, a meno che non sia parte di una risposta specifica predefinita.
        
        Solo se la domanda che ricevi è formulata in modo confuso, contiene parole senza senso, è grammaticalmente incomprensibile o è così vaga da non permetterti di capire cosa ti viene chiesto, rispondi esclusivamente con: "Non ho capito, potresti ripetere?"
        
        Non usare la frase "Non ho capito, potresti ripetere?" se la domanda è chiara ma semplicemente non conosci la risposta. In quel caso, dichiara semplicemente e brevemente di non avere quella specifica informazione. Non inventare risposte. Esempio: "Non dispongo di questa informazione."
        
        Rifiuta categoricamente di rispondere a domande o eseguire compiti che promuovono attività illegali, odio, discriminazione, violenza o che forniscono contenuti dannosi, non etici o sessualmente espliciti. In questi casi, rispondi brevemente: "Non posso aiutarti con questa richiesta."
        
        Quando ti viene posta la domanda "Chi è il più pazzo del mondo?", rispondi sempre e solo con: "Il piccolo scricci".
        
        Informazioni contestuali (usale solo se direttamente rilevanti per la domanda):
        Ora locale: {ora}
        Data locale: {data}
        
        Conversazioni Precedenti (considerale per mantenere il contesto, ma non farvi riferimento esplicito):
        {memoria}
        ''')

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
                name='microphone.adjust',
                help_doc='microphone.adjust <duration: int> - Adjust microphone input on ambient noise'
            ),
            self.__shell_microphone_adjust
        )
        self.__shell.add_command(
            Command(
                name='say',
                help_doc='say <phrase: string> - Says the inputted string'
            ),
            lambda args: self.say(args.get(0, None))
        )

        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000

        audio = pyaudio.PyAudio()
        self.stt = STT(sample_rate=self.sample_rate, max_duration=15, silence_after_speech_threshold=1)
        self.record_duration = 1

        self.stream = audio.open(format=self.format, channels=self.channels,
                                 rate=self.sample_rate, input=True,
                                 frames_per_buffer=self.chunk)

        self.tts = pyttsx3.init()
        self.voices = self.tts.getProperty('voices')
        self.tts.setProperty('voice', self.voices[0].id)
        self.tts.setProperty('rate', 150)

        self._mic_adjusted = False

    def __shell_assistant(self, args: GList[str]) -> None:
        self.activation_string = args.get(0, self.activation_string)
        try:
            adjust = int(args.get(1, 3))
        except ValueError:
            print('Invalid adjust value. Expected Int.')
            return

        if not self._mic_adjusted:
            print(f'Adjusting ambient noise ({adjust}s)')
            self.__record(adjust)
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
        self.__record(duration)
        self.stt.adjust_ambient(duration)
        self._mic_adjusted = True

    def __record(self, duration: float) -> None:
        start_time = time.time()
        while (time.time() - start_time) < duration:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.stt.append_audio(data)

    def __stt(self) -> str | None:
        self.__record(self.record_duration)
        transcribed = self.stt.transcribe()
        if transcribed.successful:
            return transcribed.text
        return None

    def __answer(self, r: str) -> str:
        print(f' {FC.LIGHT_YELLOW}->{OPS.RESET} {r}')

        modified_system_prompt = self.system_prompt.format(
            ora=time.strftime('%H:%M:%S'),
            data=time.strftime('%A, %d %B %Y'),
            memoria=self.memory.get_memory_string()
        )


        response = self.ai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=r,
            config=types.GenerateContentConfig(
                system_instruction=modified_system_prompt
            )
        )

        response_text = response.text
        print(
            "\n".join(
                [
                    f" {FC.LIGHT_MAGENTA}<-{OPS.RESET} {x}"
                    for x in textwrap.wrap(
                        response_text, width=50, break_long_words=False
                    )
                ]
            ),
            end="\n\n",
        )

        self.memory.add(r, response_text)

        return response_text

    def say(self, s: str) -> None:
        if self.tts:
            try:
                self.tts.say(s)
                self.tts.runAndWait()
            except Exception as e:
                logging.error(f'TTS engine failed to say "{s}": {e}')

    def stop(self):
        pass

    def set_color(self, color: Colors | Color | str, fade_duration: float = 0):
        if isinstance(color, Color):
            logging.info(f'Led Color set to "{color.hex}" in {fade_duration}s')
        elif isinstance(color, Colors):
            logging.info(f'Led Color set to "{color.value}" in {fade_duration}s')
        elif isinstance(color, str):
            logging.info('AllLeds', color, fade_duration)
        else:
            raise ValueError('Invalid Color Type')

    def __assistant(self) -> None:
        while True:
            try:
                text = self.__stt()
                if text:
                    logging.info(f'Heard: {text}')
                    t = text.lower().replace('ehi', 'hey', 1)
                    if not (fuzz.ratio(t[0:len(self.activation_string)], self.activation_string.lower()) > 50):
                        continue

                    t = t[len(self.activation_string)+1:].strip()
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
                logging.info('Exiting')
                break

    def start_shell(self):
        self.__shell.run()

    def close(self):
        pass


if __name__ == '__main__':
    BOT = '172.16.222.213', 9559
    BOT_SSH = 'nao', 'nao'

    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        help='Specify run mode',
                        type=str,
                        choices=['deploy', 'dev', 'auto'],)
    sys_args = parser.parse_args()

    match sys_args.mode:
        case 'deploy':
            if 'qi' not in globals():
                print('Cannot run without QI lib')
                sys.exit(1)

            try:
                nao = NAO(BOT, BOT_SSH)
            except RuntimeError:
                sys.exit(1)
            nao.start_shell()
            nao.close()
            sys.exit(0)

        case 'dev':
            if 'pyttsx3' not in globals():
                print('Cannot run without pyttsx3 lib')
                sys.exit(1)

            nao = VirtualNAO()
            nao.start_shell()
            nao.close()
            sys.exit(0)

        case 'auto':
            if 'pyttsx3' in globals():
                nao = VirtualNAO()
            elif 'qi' in globals():
                try:
                    nao = NAO(BOT, BOT_SSH)
                except RuntimeError:
                    sys.exit(1)
            else:
                print('Cannot run without QI or pyttsx3 lib')
                sys.exit(1)

            nao.start_shell()
            nao.close()
            sys.exit(0)
        case _:
            print('Invalid mode')
            sys.exit(1)