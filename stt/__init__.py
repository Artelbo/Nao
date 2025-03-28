import wave
from speech_recognition import AudioSource
from .base import SpeechToText, Service, Languages, convert_language_code
from enum import Enum
import speech_recognition as sr
from typing import Any, Generator
from vosk import Model, KaldiRecognizer
import orjson
import whisper
import faster_whisper
import io
import numpy as np
import torch
import webrtcvad
import sounddevice as sd
from tqdm import tqdm
import os
from collections import deque
import time

os.environ['CT2_VERBOSE'] = '-3'


class Services(Enum):
    GoogleSpeechRecognition = Service(name='Google Speech Recognition',
                                      requires_api_key=False,
                                      online=True)
    OpenAIWhisper = Service(name='OpenAI Whisper API',
                            requires_api_key=True,
                            online=True)
    MicrosoftAzureSpeech = Service(name='Microsoft Azure Speech',
                                   requires_api_key=True,
                                   online=True)
    VoskOffline = Service(name='Vosk',
                          requires_api_key=False,
                          online=False,
                          model_path='vosk-model-it-0.22')
    OpenAIWhisperOffline = Service(name='OpenAI Whisper',
                                   requires_api_key=False,
                                   online=False)
    OpenAIFasterWhisperOffline = Service(name='OpenAI Faster Unhallucinated Whisper',
                                         requires_api_key=False,
                                         online=False)
    FasterWhisperVoskHybrid = Service(name='Vosk (VAD) + OpenAI Faster Unhallucinated Whisper (STT)',
                                      requires_api_key=False,
                                      online=False,
                                      model_path='vosk-model-small-it-0.22') # Vosk model / the fastest and smallest

    Tests = Service(name='Some tests',
                    requires_api_key=False,
                    online=False,
                    model_path='vosk-model-small-it-0.22')


class SimpleService:
    BestOnline = Services.GoogleSpeechRecognition
    BestOffline = Services.FasterWhisperVoskHybrid


class STTException(Exception):
    """Speech To Text exception"""
    pass


def contains_speech(audio_data, sample_rate, aggressiveness=3, frame_duration_ms=30) -> tuple[bool, float]:
    """
    Checks if a NumPy array of int16 audio data contains speech using WebRTC VAD.

    Args:
        audio_data (np.ndarray): NumPy array of int16 audio data.
        sample_rate (int): Sample rate of the audio data.
        aggressiveness (int): WebRTC VAD aggressiveness level (0-3).
        frame_duration_ms (int): Duration of each frame in milliseconds (10, 20, or 30).

    Returns:
        bool: True if speech is detected, False otherwise.
        float: The proportion of frames detected as speech.  Returns -1 if an
               error occurs.

    Raises:
        ValueError: If invalid arguments are provided.
        TypeError: If audio_data is not a NumPy array or not int16.
        RuntimeError: If WebRTC VAD fails.
    """

    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError("Aggressiveness must be between 0 and 3.")
    if frame_duration_ms not in [10, 20, 30]:
        raise ValueError("Frame duration must be 10, 20, or 30 ms.")
    if not isinstance(audio_data, np.ndarray):
        raise TypeError("audio_data must be a NumPy array.")
    if audio_data.dtype != np.int16:
        raise TypeError("audio_data must be of type np.int16.")


    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(aggressiveness)

    # Calculate frame size in samples
    frame_size = int(sample_rate * frame_duration_ms / 1000)


    if len(audio_data) < frame_size:
        return False, 0.0 if len(audio_data) > 0 else -1.0 #handle empty audio

    # Iterate through audio frames
    speech_frames = 0
    total_frames = 0

    try:
        for i in range(0, len(audio_data) - frame_size + 1, frame_size):
            frame = audio_data[i:i + frame_size].tobytes()
            if vad.is_speech(frame, sample_rate):
                speech_frames += 1
            total_frames += 1
    except Exception as e:
        raise RuntimeError(f'Error processing with WebRTC VAD: {e}')

    if total_frames == 0:
        return False, -1.0

    speech_proportion = speech_frames / total_frames
    return speech_proportion > 0.1, speech_proportion


def calculate_rms(chunk):
    squared_mean = np.mean(chunk ** 2)
    if squared_mean < 0:
        if abs(squared_mean) < 1e-10:
            squared_mean = 0
        else:
            return 0
    rms = np.sqrt(squared_mean)
    return rms


class STT(SpeechToText):
    def __init__(self,
                 source: sr.AudioSource | None,
                 service: Services,
                 language: Languages = Languages.ITALIAN_ITALY,
                 api_key: None | str = None) -> None:
        self.recognizer: sr.Recognizer = sr.Recognizer()

        if not isinstance(source, AudioSource) or source is not None:
            raise ValueError('Source must be an instance of sr.AudioSource.')
        self.source: sr.AudioSource | None = source

        if not isinstance(service, Services):
            raise ValueError('Service must be an instance of Services.')
        self.service: Services = service

        if not isinstance(api_key, str) and api_key is not None:
            raise ValueError('API key must be a string or None.')
        self.api_key: str | None = api_key

        if not isinstance(language, Languages):
            raise ValueError('Language must be an instance of Languages.')
        self.language = language

        self.rms_threshold = 50
        self.silence_threshold_ms = 750
        self.timeout_ms = 10000
        self.sample_rate = 16000

        if self.service == Services.VoskOffline:
            self.vosk_model = Model(self.service.value.model_path)
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)
            # self.audio_queue = queue.Queue()

        elif self.service == Services.OpenAIWhisperOffline:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = whisper.load_model('turbo', device=device)
            # self.vad = webrtcvad.Vad(3)

        elif self.service == Services.OpenAIFasterWhisperOffline:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = faster_whisper.WhisperModel('large-v3-turbo', device=device)
            self.vad = webrtcvad.Vad(3)

        elif self.service in (Services.FasterWhisperVoskHybrid, Services.Tests):
            self.vosk_model = Model(self.service.value.model_path)
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)
            self.vosk_recognizer.SetWords(True)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = faster_whisper.WhisperModel('large-v3', device=device,
                                                     compute_type='int8') #  Use large-v3, int8 for speed/memory
            self.last_audio_time = 0
            self.last_process_time = 0
            self.buffered_audio = np.array([], dtype=np.int16)
            self.last_audio_time = 0  # Initialize last_audio_time
            self.last_process_time = 0 # Initialize last_process_time


    def adjust_ambient(self, duration: int = 5) -> float | None:
        """Adjusts for ambient noise."""
        if self.source is None:
            raise ValueError('No valid source given')

        if self.service in (Services.VoskOffline, Services.OpenAIWhisperOffline,
                            Services.OpenAIFasterWhisperOffline, Services.FasterWhisperVoskHybrid,
                            Services.Tests):
            rmss = []
            for _ in tqdm(range(duration), desc=f'RMS Adjusting for {self.service}', unit='op', dynamic_ncols=True):
                audio = self.recognizer.record(self.source, duration=1)
                wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                audio_array = np.frombuffer(wav_data, dtype=np.int16)
                rmss.append(calculate_rms(audio_array))

            if rmss: #check if rmss is empty
                self.rms_threshold = (sum(rmss) / len(rmss)) * 1.5

            return self.rms_threshold
        else:
            return self.recognizer.adjust_for_ambient_noise(self.source, duration)

    def __process_buffered_audio(self, current_time):
        audio_float32 = self.buffered_audio.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float32,
            language=convert_language_code(self.language),
            unhallucinated=True
        )
        full_text = "".join(segment.text for segment in segments).strip()
        self.buffered_audio = np.array([], dtype=np.int16)
        if full_text:
            yield full_text
        self.last_process_time = current_time

    def transcribe(self, prompt: str | None = None) -> Generator[str, None, None]:
        """Transcribes speech based on the selected service."""
        if self.source is None:
            raise ValueError('No valid source given')

        while True:
            try:
                audio = self.recognizer.listen(self.source, phrase_time_limit=30)  # Limit phrase time
                current_time = time.time()
                match self.service:
                    case Services.GoogleSpeechRecognition:
                        yield self.recognizer.recognize_google(audio, language=self.language.value)

                    case Services.OpenAIWhisper:
                        if not self.api_key:
                            raise ValueError('API key is required for OpenAI Whisper.')
                        yield self.recognizer.recognize_whisper(audio, api_key=self.api_key, language=self.language.value)

                    case Services.MicrosoftAzureSpeech:
                        if not self.api_key:
                            raise ValueError('API key is required for Microsoft Azure Speech.')
                        yield self.recognizer.recognize_azure(audio, key=self.api_key, language=self.language.value)


                    case Services.VoskOffline:
                        wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                        if self.vosk_recognizer.AcceptWaveform(wav_data):
                            result = orjson.loads(self.vosk_recognizer.Result())
                            yield result.get('text', '')
                        # else: #removed else
                        #     result = orjson.loads(self.vosk_recognizer.PartialResult())
                        #     print("Partial result: ", result.get('partial', ''))

                    case Services.OpenAIWhisperOffline:
                        # Original Whisper (less efficient, kept for reference)
                        wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                        audio_array_int16 = np.frombuffer(wav_data, dtype=np.int16)
                        audio_float32 = audio_array_int16.astype(np.float32) / 32768.0

                        yield whisper.transcribe(self.model,  # type: whisper.Whisper
                                                 audio_float32,
                                                 language=convert_language_code(self.language),  # type: ignore
                                                 verbose=False,  #  Set to True for debugging
                                                 temperature=(0.0, 0.2),
                                                 logprob_threshold=-1.0,
                                                 no_speech_threshold=0.5,
                                                 hallucination_silence_threshold=2)['text']


                    case Services.OpenAIFasterWhisperOffline:
                        wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                        audio_array_int16 = np.frombuffer(wav_data, dtype=np.int16)
                        audio_float32 = audio_array_int16.astype(np.float32) / 32768.0

                        segments, _ = self.model.transcribe(
                            audio_float32,
                            language=convert_language_code(self.language),
                            unhallucinated=True,
                            initial_prompt=prompt,
                            best_of=5,
                        )
                        full_text = ''.join(segment.text for segment in segments).strip()
                        yield full_text



                    case Services.FasterWhisperVoskHybrid:
                        wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                        audio_array_int16 = np.frombuffer(wav_data, dtype=np.int16)
                        self.last_audio_time = current_time

                        if not hasattr(self, 'last_process_time'):
                            self.last_process_time = current_time
                        # --- Silence Detection ---
                        chunk_size = int(self.sample_rate * 0.1)
                        is_silent = True
                        for i in range(0, len(audio_array_int16), chunk_size):
                            chunk = audio_array_int16[i:i + chunk_size]
                            rms = calculate_rms(chunk)
                            if rms > self.rms_threshold:
                                is_silent = False
                                break

                        if not is_silent:
                            self.buffered_audio = np.concatenate((self.buffered_audio, audio_array_int16))

                            # Vosk Processing FIRST
                            if self.vosk_recognizer.AcceptWaveform(self.buffered_audio.tobytes()):
                                vosk_result = orjson.loads(self.vosk_recognizer.Result()).get('text', '')
                                # print(f'Vosk results: "{vosk_result}"') # Debug

                                if vosk_result: # If Vosk detects something, THEN use Faster Whisper
                                    audio_float32 = self.buffered_audio.astype(np.float32) / 32768.0
                                    segments, _ = self.model.transcribe(
                                        audio_float32,
                                        language=convert_language_code(self.language),
                                        unhallucinated=True
                                    )
                                    full_text = ''.join(segment.text for segment in segments).strip()
                                    self.buffered_audio = np.array([], dtype=np.int16)  # Clear buffer
                                    if full_text:
                                        yield full_text
                                    self.last_process_time = current_time

                        # --- Timeout and Silence Handling ---
                        elapsed_since_last_process = (current_time - self.last_process_time) * 1000
                        if len(self.buffered_audio) > 0 and elapsed_since_last_process:
                            if (current_time - self.last_process_time) * 1000 > self.silence_threshold_ms:
                                if self.vosk_recognizer.AcceptWaveform(self.buffered_audio.tobytes()):
                                    vosk_result = orjson.loads(self.vosk_recognizer.Result()).get('text', '')
                                    if vosk_result:
                                        audio_float32 = self.buffered_audio.astype(np.float32) / 32768.0
                                        segments, _ = self.model.transcribe(
                                            audio_float32,
                                            language=convert_language_code(self.language),
                                            unhallucinated=True
                                        )
                                        full_text = ''.join(segment.text for segment in segments).strip()
                                        self.buffered_audio = np.array([], dtype=np.int16)
                                        if full_text:
                                            yield full_text

                                        self.last_process_time = current_time
                        if (current_time - self.last_process_time) * 1000 > self.timeout_ms and len(self.buffered_audio) > 0:
                            if self.vosk_recognizer.AcceptWaveform(self.buffered_audio.tobytes()):
                                vosk_result = orjson.loads(self.vosk_recognizer.Result()).get('text', '')

                                if vosk_result:
                                    audio_float32 = self.buffered_audio.astype(np.float32) / 32768.0
                                    segments, _ = self.model.transcribe(
                                        audio_float32,
                                        language=convert_language_code(self.language),
                                        unhallucinated=True
                                    )
                                    full_text = "".join(segment.text for segment in segments).strip()
                                    self.buffered_audio = np.array([], dtype=np.int16)  # Clear buffer
                                    if full_text:
                                        yield full_text

                            self.last_process_time = current_time

                    case Services.Tests:
                        wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                        audio_array_int16 = np.frombuffer(wav_data, dtype=np.int16)
                        self.last_audio_time = current_time

                        if not hasattr(self, 'last_process_time'):
                            self.last_process_time = current_time

                        chunk_size = int(self.sample_rate * 0.1)
                        is_silent = True
                        for i in range(0, len(audio_array_int16), chunk_size):
                            chunk = audio_array_int16[i:i + chunk_size]
                            rms = calculate_rms(chunk)
                            if rms > self.rms_threshold:
                                is_silent = False
                                break

                        if not is_silent:
                            self.buffered_audio = np.concatenate((self.buffered_audio, audio_array_int16))
                            # Vosk Processing FIRST
                            if self.vosk_recognizer.AcceptWaveform(self.buffered_audio.tobytes()):
                                vosk_result = orjson.loads(self.vosk_recognizer.Result()).get('text', '')
                                if vosk_result:
                                    # print(f'[vosk] {vosk_result}')
                                    yield from self.__process_buffered_audio(current_time)
                                    self.buffered_audio = np.array([], dtype=np.int16)

                        # elapsed_since_last_process = (current_time - self.last_process_time) * 1000
                        # if len(self.buffered_audio) > 0 and elapsed_since_last_process:
                        #     if elapsed_since_last_process > self.silence_threshold_ms:
                        #         yield from self.__process_buffered_audio(current_time)
                        #         self.buffered_audio = np.array([], dtype=np.int16)
                        # if (current_time - self.last_process_time) * 1000 > self.timeout_ms and len(self.buffered_audio) > 0:
                        #     yield from self.__process_buffered_audio(current_time)
                        #     self.buffered_audio = np.array([], dtype=np.int16)

                    case _:
                        raise ValueError('Unsupported speech recognition service.')

            except sr.UnknownValueError:
                yield ''  #  Return empty string for no speech
            except sr.RequestError as e:
                raise STTException(f'Could not request results from the selected service: {e}')
            except RuntimeError as e: #  Catch runtime errors (e.g., from VAD)
                print(f"Runtime error: {e}")
                yield ''

    def transcribe_file(self, prompt: str | None = None) -> Generator[str, None, None]:
        """Transcribes speech from an audio FILE based on the selected service."""

        try:
            audio = self.recognizer.record(self.source)
        except Exception as e:
            raise STTException(f"Error reading audio file: {e}")


        try:
            match self.service:
                case Services.VoskOffline:
                    wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                    if self.vosk_recognizer.AcceptWaveform(wav_data):
                        result = orjson.loads(self.vosk_recognizer.Result())
                        yield result.get('text', '')
                    else:  # For Vosk, process any remaining partial result.
                        result = orjson.loads(self.vosk_recognizer.FinalResult())
                        yield result.get('text', '')

                case Services.FasterWhisperVoskHybrid:
                    wav_data = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
                    audio_array_int16 = np.frombuffer(wav_data, dtype=np.int16)
                    audio_float32 = audio_array_int16.astype(np.float32) / 32768.0

                    # Prioritize Faster Whisper
                    segments, _ = self.model.transcribe(
                        audio_float32,
                        language=convert_language_code(self.language),
                        unhallucinated=True,
                        initial_prompt=prompt,
                        best_of=5,
                    )
                    full_text = ''.join(segment.text for segment in segments).strip()

                    if full_text:
                        yield full_text
                    else:  # Fallback to Vosk if Faster Whisper returns nothing
                        if self.vosk_recognizer.AcceptWaveform(wav_data):
                            result = orjson.loads(self.vosk_recognizer.Result())
                            yield result.get('text', '')
                        else:
                            result = orjson.loads(self.vosk_recognizer.FinalResult())
                            yield result.get('text', '')


                case _:
                    raise ValueError('Unsupported speech recognition service.')

        except sr.UnknownValueError:
            yield ''  # Return empty string for no speech
        except sr.RequestError as e:
            raise STTException(f'Could not request results from the selected service: {e}')
        except Exception as e:
            raise STTException(f'An unexpected error occurred: {e}')