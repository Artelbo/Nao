import time
from threading import Thread
from naoqi import ALProxy
import socket
from protolib.client2 import Session


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]


class NAOBot:
    __IP_ADDRESS = get_local_ip()
    __PORT = 9559

    __SERVER_IP = '127.0.0.1'
    __SERVER_PORT = 7942

    __ASSISTANT_AUDIO_PATH = '/home/nao/assistant_audio.wav'
    __CHUNK_TIME_LEN = 1

    def __init__(self):
        self.motion = ALProxy('ALMotion', self.__IP_ADDRESS, self.__PORT)
        self.posture = ALProxy('ALRobotPosture', self.__IP_ADDRESS, self.__PORT)
        self.recorder = ALProxy('ALAudioRecorder', self.__IP_ADDRESS, self.__PORT)
        self.player = ALProxy('ALAudioPlayer', self.__IP_ADDRESS, self.__PORT)
        self.tts = ALProxy('ALTextToSpeech', self.__IP_ADDRESS, self.__PORT)
        self.sr = ALProxy('ALSpeechRecognition', self.__IP_ADDRESS, self.__PORT)
        self.led = ALProxy('ALLeds', self.__IP_ADDRESS, self.__PORT)
        self.com_session = Session((self.__SERVER_IP, self.__SERVER_PORT))

    def __assistant(self):
        while True:
            try:
                self.recorder.stopMicrophonesRecording()
                self.recorder.startMicrophonesRecording(self.__ASSISTANT_AUDIO_PATH, 'wav', 16000, [0, 0 , 1, 0])
                time.sleep(self.__CHUNK_TIME_LEN)
                self.recorder.stopMicrophonesRecording()

            except Exception as e:
                print('Error: ' + str(e))
            except KeyboardInterrupt:
                break


    def run(self):
        try:
            self.tts.setLanguage('Italian')
            self.tts.setParameter('pitchShift', 1)
            self.led.fadeRGB('AllLeds', 'green', 0)
            while True:
                command = raw_input(': ').lower()

                if command == 'assistant':
                    self.com_session.stream(self.__assistant)

                if command == 'quit':
                    break

        except KeyboardInterrupt:
            print('\nExiting..')
                
    def __say(self, text):
        self.tts.say(text)
        
    def __move(self, x, y, z):
        self.motion.setCollisionProtectionEnabled('Arms', True)
        self.motion.moveTo(x, y, z)


if __name__ == '__main__':
    nao = NAOBot()
    nao.run()
