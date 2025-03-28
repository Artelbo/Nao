import os
import time
import wave
import io
import speech_recognition as sr
import pyaudio
from typing import List, Tuple
import simpleaudio as sa
import logging
import locale
from google import genai


class STTResponse:
    def __init__(self, successful: bool, text: str):
        self.successful = successful
        self.text = text


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
locale.setlocale(locale.LC_TIME, 'it_IT')

start = 'leonardo'
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
system_prompt = '''
Sei Nao, un robot che risponde grazie a Google Gemini, sei stato programmato grazie a Python e la mente di 5 geni.
Le risposte che fornisci devono essere in testo semplice, senza alcuna formattazione stilistica. Evita l'uso di grassetto, corsivo, sottolineato, elenchi puntati o numerati, parentesi, o qualsiasi altro tipo di formattazione.
Fornisci risposte chiare e concise, esclusivamente in testo puro.
Le risposte devono essere brevi e riassuntive.
Ricorda di riferiti a me (io) come tu, sempre, eccetto quando puoi sott'intenderlo.
Se ti viene posta una domanda non pertinente puoi rispondere con: "Non ho capito, potresti ripetere?"
Ora locale: {ora}
Data locale: {data}

Conversazioni Precedenti:
---
{memoria}
---
'''


class Memory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.memory = []

    def add(self, q, a) -> None:
        self.memory.append({'question': q, 'answer': a})
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def get_memory_string(self) -> str:
        return '\n\n'.join([f'Domanda: {item['question']}\nRisposta: {item['answer']}' for item in self.memory])


class STT:
    def __init__(self,
                 sample_rate: int = 16000,
                 sample_width: int = 2,
                 channels: int = 1,
                 max_duration: int = 10):
        self.recognizer = sr.Recognizer()
        self.audio_data: List[Tuple[float, bytes]] = []
        self.sample_rate: int = sample_rate
        self.sample_width: int = sample_width
        self.channels: int = channels
        self.max_duration: int = max_duration
        logging.info(f'STT initialized with sample_rate={sample_rate}, sample_width={sample_width}, max_duration={max_duration}s')

    def append_audio(self, audio_chunk: bytes) -> None:
        if not isinstance(audio_chunk, bytes):
             logging.error('Invalid audio chunk type. Expected bytes.')
             return
        if not audio_chunk:
             logging.warning('Attempted to append empty audio chunk.')
             return

        timestamp = time.time()
        self.audio_data.append((timestamp, audio_chunk))

        while self.audio_data and (timestamp - self.audio_data[0][0]) > self.max_duration:
            removed_ts, removed_chunk = self.audio_data.pop(0)

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

            self.clear_audio()
            timestamp = time.time()
            self.audio_data.append((timestamp, raw_audio))
            logging.info(f'Successfully loaded {file_duration:.2f}s of audio from {file_path}, replacing buffer content.')

            if file_duration > self.max_duration:
                logging.warning(f'Loaded file duration ({file_duration:.2f}s) exceeds max buffer duration ({self.max_duration}s).')

        except FileNotFoundError:
            logging.error(f"Audio file not found: '{file_path}'")
            raise
        except wave.Error as e:
            logging.error(f"Error opening or reading WAV file '{file_path}': {e}")
            raise ValueError(f"Invalid WAV file '{file_path}': {e}") from e
        except ValueError as e:
            logging.error(f"Error processing audio file '{file_path}': {e}")
            raise

    def clear_audio(self) -> None:
        if self.audio_data:
            logging.info('Clearing audio buffer.')
            self.audio_data = []
        else:
            logging.debug('Audio buffer is already empty.')


    def adjust_ambient(self, duration: float = 1.0) -> None:
        if not self.audio_data:
            logging.warning('Cannot adjust for ambient noise: No audio data available.')
            return

        logging.info(f'Adjusting for ambient noise using up to {duration}s from buffer start.')
        combined_audio_bytes = b''.join(chunk for _, chunk in self.audio_data)

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
                    self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                    logging.info(f'Successfully adjusted for ambient noise. Energy threshold: {self.recognizer.energy_threshold:.2f}')
                except Exception as e:
                    logging.error(f'Error during ambient noise adjustment: {e}')

        except wave.Error as e:
             logging.error(f'Failed to create in-memory WAV for ambient adjustment: {e}')
        except Exception as e:
             logging.error(f'An unexpected error occurred in adjust_ambient: {e}')


    def transcribe(self) -> STTResponse:
        if not self.audio_data:
            logging.warning('Transcription requested but no audio data is available.')
            return STTResponse(successful=False, text='No audio data available to transcribe.')

        combined_audio_bytes = b''.join(chunk for _, chunk in self.audio_data)
        total_duration = len(combined_audio_bytes) / (self.sample_rate * self.sample_width * self.channels)

        if total_duration < self.max_duration / 3:
            logging.info(f'Buffer duration {total_duration:.2f}s is less than required {self.max_duration / 3}s. Skipping transcription.')
            return STTResponse(successful=False, text='Buffer not full enough for transcription.')

        logging.info('Starting transcription process...')
        logging.info(f'Transcribing {total_duration:.2f}s of audio.')

        audio_data_obj = sr.AudioData(combined_audio_bytes, self.sample_rate, self.sample_width)

        try:
            text: str = self.recognizer.recognize_google(audio_data_obj, language='it-IT')  # type: ignore
            logging.info(f"Raw transcription result: '{text}'")

            if text and text.strip():
                cleaned_text = text.strip()
                logging.info(f"Successful transcription: '{cleaned_text}'. Clearing audio buffer.")
                self.clear_audio()
                return STTResponse(successful=True, text=cleaned_text)
            else:
                logging.warning('Transcription resulted in empty text.')
                return STTResponse(successful=False, text='Audio processed but no speech detected.')

        except sr.UnknownValueError:
            logging.warning('Google Speech Recognition could not understand the audio.')
            return STTResponse(successful=False, text='Could not understand the audio.')
        except sr.RequestError as e:
            logging.error(f'Could not request results from Google Speech Recognition service: {e}')
            return STTResponse(successful=False, text=f'Could not connect to Google STT service: {e}')
        except Exception as e:
            logging.exception('An unexpected error occurred during transcription.')
            return STTResponse(successful=False, text='An unexpected error occurred during transcription.')


def record_audio_continuous(transcribe_interval: float = 3.0): # Transcribe every 3 seconds
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000

    audio = pyaudio.PyAudio()
    # Initialize STT with a suitable max_duration, e.g., 10 seconds
    stt = STT(sample_rate=sample_rate, max_duration=21)

    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    print("Recording... Press CTRL+C to stop.")
    last_transcription_time = time.time()
    try:
        while True:
            data = stream.read(chunk, exception_on_overflow=False) # Read one chunk
            stt.append_audio(data) # Append directly to the buffer

            current_time = time.time()
            # Check if it's time to transcribe
            if current_time - last_transcription_time >= transcribe_interval:
                if stt.audio_data: # Only transcribe if there's data
                    response = stt.transcribe() # Transcribe the current buffer content
                    last_transcription_time = current_time # Reset timer
                    if response.successful:
                        print(f"{response.text}")
                else:
                   # If no data, still reset timer to avoid immediate re-check
                   last_transcription_time = current_time


            # Small sleep to prevent busy-waiting, adjust as needed
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        # Ensure resources are closed properly
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'audio' in locals():
            audio.terminate()

if __name__ == "__main__":
    record_audio_continuous(3)