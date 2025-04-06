from dataclasses import dataclass
import speech_recognition as sr
from typing import List, Tuple, Optional, Annotated, Literal
from logging import getLogger, Logger
import audioop
import time
import wave
from .b_types import MoreThan
import io


@dataclass(frozen=True)
class STTResponse:
    successful: bool
    text: str


class STT:
    def __init__(self,
                 sample_rate: Annotated[int, Literal[16000, 48000]] = 16000,
                 sample_width: int = 2,
                 channels: Annotated[int, Literal[1, 2, 4]] = 1,
                 max_duration: Annotated[int, MoreThan(0)] = 10,
                 min_duration: Annotated[int, MoreThan(-1)] = 1,
                 silence_after_speech_threshold: float = 1.0,
                 language: Literal['it-IT', 'en-US'] = 'it-IT'):
        self.__logger: Logger = getLogger('stt')

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False

        self.audio_data: List[Tuple[float, bytes]] = []
        self.sample_rate: int = sample_rate
        self.sample_width: int = sample_width
        self.channels: int = channels
        self.max_duration: int = max_duration
        self.min_duration: int = min_duration
        self.silence_after_speech_threshold: float = silence_after_speech_threshold

        self.language = language

        # State tracking for silence detection
        self.speech_detected_since_last_transcription: bool = False
        self.time_of_last_speech: Optional[float] = None
        self.time_of_last_audio_chunk: Optional[float] = None

        self.__logger.info(f'STT initialized with sample_rate={sample_rate}, sample_width={sample_width}, '
                           f'max_duration={max_duration}s, min_duration={min_duration}s, '
                           f'silence_threshold={silence_after_speech_threshold}s')

    def _calculate_chunk_energy(self, audio_chunk: bytes) -> float:
        if not audio_chunk:
            return 0.0
        try:
            return audioop.rms(audio_chunk, self.sample_width)
        except Exception as e:
            self.__logger.error(
                f'Error calculating RMS: {e}. Chunk length: {len(audio_chunk)}, Sample width: {self.sample_width}'
            )
            return 0.0

    def append_audio(self, audio_chunk: bytes) -> None:
        """
        Appends a chunk of audio data to the internal buffer.

        This method receives a chunk of raw audio data as bytes, timestamps it,
        and adds it to the internal buffer. It also performs real-time speech
        detection by calculating the energy of the audio chunk and comparing it
        against the energy threshold. If speech is detected, it updates the
        internal state accordingly. Additionally, it manages the buffer size by
        removing old audio chunks if the total duration exceeds a maximum.

        Args:
            audio_chunk (bytes): A chunk of raw audio data.

        Raises:
            TypeError: If the provided `audio_chunk` is not of type bytes.
        """
        if not isinstance(audio_chunk, bytes):
             self.__logger.error('Invalid audio chunk type. Expected bytes.')
             return

        current_time = time.time()
        self.time_of_last_audio_chunk = current_time

        if audio_chunk:
            timestamp = current_time
            self.audio_data.append((timestamp, audio_chunk))

            chunk_energy = self._calculate_chunk_energy(audio_chunk)
            if chunk_energy > self.recognizer.energy_threshold:
                self.speech_detected_since_last_transcription = True
                self.time_of_last_speech = timestamp
                self.__logger.debug(f'Speech detected: Energy {chunk_energy:.2f} > Threshold {self.recognizer.energy_threshold:.2f}')
            else:
                self.__logger.debug(f"Silence detected: Energy {chunk_energy:.2f} <= Threshold {self.recognizer.energy_threshold:.2f}")

            while self.audio_data and (timestamp - self.audio_data[0][0]) > self.max_duration:
                removed_ts, _ = self.audio_data.pop(0)

    def append_audio_file(self, file_path: str) -> None:
        """
        Appends audio data from a WAV file to the internal buffer.

        This method loads audio data from a specified WAV file, validates its
        format against the expected sample rate, sample width, and number of
        channels, and then appends the raw audio data to the internal buffer.
        It also updates the internal state to reflect that speech has been
        detected and sets the timestamps accordingly.

        Args:
            file_path (str): The path to the WAV file to load.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid WAV file or if its format
                (sample rate, sample width, channels) does not match the
                expected format.
            wave.Error: If there is an error opening or reading the WAV file.
        """
        self.__logger.debug(f'Attempting to load audio from file: {file_path}')
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                file_sw = wf.getsampwidth()
                file_ch = wf.getnchannels()
                file_frames = wf.getnframes()

                if file_sr != self.sample_rate:
                    raise ValueError(f'File sample rate {file_sr}Hz does not match expected {self.sample_rate}Hz.')
                if file_sw != self.sample_width:
                    raise ValueError(f'File sample width {file_sw} bytes does not match expected {self.sample_width} bytes.')
                if file_ch != self.channels:
                     self.__logger.warning(f'File has {file_ch} channels, expected {self.channels}. Processing might yield unexpected results.')

                raw_audio = wf.readframes(file_frames)
                file_duration = file_frames / float(file_sr)

            self.clear_audio()
            timestamp = time.time()

            self.audio_data.append((timestamp, raw_audio))
            self.time_of_last_audio_chunk = timestamp

            self.speech_detected_since_last_transcription = True
            self.time_of_last_speech = timestamp

            self.__logger.debug(f'Successfully loaded {file_duration:.2f}s of audio from {file_path}, replacing buffer content.')

            if file_duration > self.max_duration:
                self.__logger.warning(f'Loaded file duration ({file_duration:.2f}s) exceeds max buffer duration ({self.max_duration}s). Trimming may occur later.')
        except FileNotFoundError:
            self.__logger.error(f"Audio file not found: '{file_path}'")
            raise
        except wave.Error as e:
            self.__logger.error(f"Error opening or reading WAV file '{file_path}': {e}")
            raise ValueError(f"Invalid WAV file '{file_path}': {e}") from e
        except ValueError as e:
            self.__logger.error(f"Error processing audio file '{file_path}': {e}")
            raise

    def _reset_state(self) -> None:
        """Resets the speech/silence detection state."""
        self.speech_detected_since_last_transcription = False
        self.time_of_last_speech = None
        self.__logger.debug('Speech/silence state reset.')

    def clear_audio(self) -> None:
        if self.audio_data:
            self.__logger.info('Clearing audio buffer and resetting state.')
            self.audio_data = []
            self._reset_state()
        else:
            self.__logger.debug('Audio buffer is already empty. State remains unchanged.')


    def adjust_ambient(self, duration: float = 1.0) -> None:
        if not self.audio_data:
            self.__logger.warning('Cannot adjust for ambient noise: No audio data available.')
            return

        self.__logger.info(f'Adjusting for ambient noise using up to {duration}s from buffer start.')
        adjustment_bytes_target = int(duration * self.sample_rate * self.sample_width * self.channels)
        combined_audio_bytes_list = []
        current_bytes = 0
        for _, chunk in self.audio_data:
            combined_audio_bytes_list.append(chunk)
            current_bytes += len(chunk)
            if current_bytes >= adjustment_bytes_target:
                break
        combined_audio_bytes = b''.join(combined_audio_bytes_list)
        combined_audio_bytes = combined_audio_bytes[:adjustment_bytes_target]

        if not combined_audio_bytes:
            self.__logger.warning('Not enough audio data in buffer for ambient noise adjustment.')
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
                    original_threshold = self.recognizer.energy_threshold
                    self.recognizer.adjust_for_ambient_noise(source,
                                                             duration=duration)
                    self.__logger.info(
                        f'Adjusted for ambient noise. Energy threshold changed from {original_threshold:.2f} to '
                        f'{self.recognizer.energy_threshold:.2f}')
                except Exception as e:
                    self.__logger.error(f'Error during ambient noise adjustment: {e}')

            self.clear_audio()

        except wave.Error as e:
            self.__logger.error(f'Failed to create in-memory WAV for ambient adjustment: {e}')
        except Exception as e:
            self.__logger.error(f'An unexpected error occurred in adjust_ambient: {e}')

    def transcribe(self) -> STTResponse:
        if not self.audio_data:
            return STTResponse(successful=False, text='No audio data available.')

        combined_audio_bytes = b''.join(chunk for _, chunk in self.audio_data)
        total_duration = len(combined_audio_bytes) / (self.sample_rate * self.sample_width * self.channels)

        if total_duration < self.min_duration:
            return STTResponse(successful=False, text=f'Buffer duration {total_duration:.2f}s < {self.min_duration}s')

        if not self.speech_detected_since_last_transcription:
            return STTResponse(successful=False, text='No speech detected yet.')

        if self.time_of_last_speech is None or self.time_of_last_audio_chunk is None:
             self.__logger.error('Inconsistent state: Speech detected but timing info missing.')
             self._reset_state()
             return STTResponse(successful=False, text='Internal state error.')

        silence_duration = self.time_of_last_audio_chunk - self.time_of_last_speech

        if silence_duration < self.silence_after_speech_threshold:
            self.__logger.info(f'Waiting for silence threshold ({self.silence_after_speech_threshold:.2f}s). '
                               f'Current silence: {silence_duration:.2f}s')
            return STTResponse(successful=False, text=f'Waiting for silence ({silence_duration:.2f}/{self.silence_after_speech_threshold:.2f}s)')

        self.__logger.info(f'Silence threshold met ({silence_duration:.2f}s >= {self.silence_after_speech_threshold:.2f}s). Starting transcription...')
        self.__logger.info(f'Transcribing {total_duration:.2f}s of audio.')

        audio_data_obj = sr.AudioData(combined_audio_bytes, self.sample_rate, self.sample_width)

        try:
            text: str = self.recognizer.recognize_google(audio_data_obj, language=self.language)  # type: ignore
            self.__logger.debug(f"Raw transcription result: '{text}'")

            if text and text.strip():
                cleaned_text = text.strip()
                self.__logger.info(f"Successful transcription: '{cleaned_text}'. Clearing buffer and resetting state.")
                self.clear_audio()
                return STTResponse(successful=True, text=cleaned_text)
            else:
                self.__logger.warning('Transcription resulted in empty text. Clearing buffer and resetting state.')
                self.clear_audio()
                return STTResponse(successful=False, text='Audio processed but no speech detected.')

        except sr.UnknownValueError:
            self.__logger.warning('Google Speech Recognition could not understand the audio. Clearing buffer and '
                                  'resetting state.')
            self.clear_audio()
            return STTResponse(successful=False, text='Could not understand the audio.')
        except sr.RequestError as e:
            self.__logger.error(f'Could not request results from Google Speech Recognition service: {e}')
            self.clear_audio()
            return STTResponse(successful=False, text=f'Could not connect to Google STT service: {e}')
        except Exception as e:
            self.__logger.exception('An unexpected error occurred during transcription. Clearing buffer and resetting '
                                    'state.')
            self.clear_audio()
            return STTResponse(successful=False, text='An unexpected error occurred during transcription.')