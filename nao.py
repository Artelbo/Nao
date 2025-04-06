from logging import Logger, getLogger
from typing import Tuple, Annotated, Literal, Dict
import paramiko
import os
from stt import STT
from builtin_shell import Shell, Command, GList
from ai import Memory
import textwrap
from google import genai
from google.genai import types
from b_types import Vector3, Colors, Color, ValueRange
from robot_specific import Postures, LedPosition
import re
from colors import FC, OPS
import time
import threading
from thefuzz import fuzz

try:
    import qi  # type: ignore
except ImportError:
    pass


class NAO:
    def __init__(self,
                 bot: Tuple[str, int],
                 ssh_user: Tuple[str, str],
                 locale_data: Dict,
                 ssh_preconnect: bool = True):
        self.__logger: Logger = getLogger('nao')

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
        self.stt = STT(max_duration=21)

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

        self.memory = Memory(capacity=5, locale=locale_data['memory'])
        self.activation_string = 'hey now'  # nao
        self.activation_response = 'Si?'
        self.ai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.system_prompt = textwrap.dedent(locale_data['prompt']).strip()
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
            lambda args: self.say(' '.join(args))
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
                    case 'stand':
                        self.set_posture(Postures.STAND)
                    case 'sit':
                        self.set_posture(Postures.SIT_RELAX)
                    case 'go-forward':
                        self.move(Vector3(0.3, 0, 0))
                    case 'go-back':
                        self.move(Vector3(-0.3, 0, 0))
            case 'left-hand':
                match sub:
                    case 'open':
                        self.set_hand('left', False)
                    case 'close':
                        self.set_hand('left', True)
            case 'right-hand':
                match sub:
                    case 'open':
                        self.set_hand('right', False)
                    case 'close':
                        self.set_hand('right', True)
            case 'leds':
                try:
                    match sub:
                        case 'green':
                            self.set_color(Colors.GREEN, float(args.get(2, 1)))
                        case 'red':
                            self.set_color(Colors.RED, float(args.get(2, 1)))
                        case 'blue':
                            self.set_color(Colors.BLUE, float(args.get(2, 1)))
                        case 'off':
                            self.set_color(Colors.OFF)
                        case 'on':
                            self.set_color(Colors.ON)
                        case _:
                            if not bool(re.match(r'^#([0-9a-fA-F]{6})$', sub)):
                                print('Invalid color value. Expected HEX color code.')
                                return
                            self.set_color(sub, float(args.get(2, 1)))
                except ValueError:
                    print(f'Invalid input. Expected float: {args.get(2)}')
            case _:
                print(f'Invalid body part. Expected one of: bot, left-hand, right-hand, leds')

    def __stt(self) -> str | None:
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
            model='gemini-2.0-flash-lite',
            contents=r,
            config=types.GenerateContentConfig(
                system_instruction=modified_system_prompt
            )
        )

        response_text = response.text
        print('\n'.join([f' {FC.LIGHT_MAGENTA}<-{OPS.RESET} {x}' for x in
                         textwrap.wrap(response_text, width=50, break_long_words=False)]),
              end='\n\n')

        self.memory.add(r, response_text)

        return response_text

    def __threaded_say(self, s: str) -> None:
        try:
            self.tts.say(s)
        except RuntimeError:
            pass

    def say(self, s: str) -> None:
        threading.Thread(target=self.__threaded_say, args=(s,), daemon=True).start()

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
        if not (0.0 <= speed <= 1.0):
            raise ValueError('Invalid Speed')
        self.posture.goPosture(posture.value, speed)

    def move(self, amount: Vector3) -> None:
        self.motion.setCollisionProtectionEnabled('Arms', True)
        self.motion.move(amount.nao_compatible)

    def set_hand(self, hand: Literal['left', 'right'], close: bool):
        if close:
            self.motion.closeHand('LHand' if hand == 'left' else 'RHand')
        else:
            self.motion.openHand('LHand' if hand == 'left' else 'RHand')

    def __assistant(self) -> None:
        self.set_color(Colors.GREEN, 1)
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
                self.__logger.info('Exiting')
                break

    def start_shell(self):
        self.__shell.run()

    def close(self):
        transport = self.ssh_client.get_transport()
        if (transport is not None) and transport.is_active():
            sftp = self.ssh_client.get_transport().open_sftp_client() if self.ssh_client.get_transport() else None
            sftp.close()
            self.ssh_client.close()