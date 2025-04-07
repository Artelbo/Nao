from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Tuple, Callable, Union
from logging import Logger, getLogger
import sys
import re
import os
import readline
import atexit
import time


@dataclass(frozen=True)
class Token:
    value: Optional[str]
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


class DefaultDict(dict):
    def __init__(self, v: Dict, default: Any):
        super().__init__(v)
        self.default = default

    def __getitem__(self, key: str) -> Any:
        return self.get(key, self.default)


class Shell:
    def __init__(self, prefix: str = 'shell'):
        self.__logger: Logger = getLogger('shell')

        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        self.prefix: str = prefix

        self.__commands: Dict[str, Tuple[Command, Callable[[GList[str]], None]]] = {}
        self.__setup_readline()

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
        self.add_command(
            Command(name='wait', help_doc='Wait for a specified number of seconds. Usage: wait <time>'),
            self._cmd_wait
        )

    def add_command(self, command: Command, callback: Callable[[GList[str]], None]) -> None:
        if command.name in self.__commands:
            self.__logger.warning(f"Command '{command.name}' is being redefined.")
        if not callable(callback):
             raise TypeError(f"Callback for command '{command.name}' must be callable.")
        self.__commands[command.name] = (command, callback)
        self.__setup_readline()

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
                pos += 1
        return tokens


    @staticmethod
    def __stringify(tokens: List[Token]) -> List[str]:
        return [token.value for token in tokens]

    def read(self) -> Optional[List[str]]:
        try:
            line = input(f'{self.prefix} > ')
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

    def _cmd_exit(self, args: GList[str]) -> None:
        raise SystemExit

    def _cmd_help(self, args: GList[str]) -> None:
        if not args:
            print('Available commands:')
            if not self.__commands:
                print('  (No commands registered)')
                return
            max_len = max(len(name) for name in self.__commands) if self.__commands else 0
            for name, (command_obj, _) in sorted(self.__commands.items()):
                print(f'  {name:<{max_len}} : {command_obj.help_doc}')
        elif len(args) == 1:
            cmd_name = args[0]
            if cmd_name in self.__commands:
                command_obj, _ = self.__commands[cmd_name]
                print(f'{command_obj.name}: {command_obj.help_doc}')
            else:
                print(f"Unknown command: '{cmd_name}'\n"
                      f"Type 'help' to see all available commands.")
        else:
            print('Usage: help [command_name]')

    def _cmd_run(self, args: GList[str]) -> None:
        if len(args) != 1:
            print('Usage: run <filename>')
            return

        script_filename = args[0]
        line_number = 0

        if not os.path.exists(script_filename):
            print(f"Error: Script file not found: '{script_filename}'", file=sys.stderr)
            return
        if not os.path.isfile(script_filename):
            print(f"Error: '{script_filename}' is not a file.", file=sys.stderr)
            return

        try:
            with open(script_filename, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line_number += 1
                    line = line.strip()

                    if not line or line.startswith('#'):
                        continue

                    try:
                        script_tokens = self.__tokenize(line)
                        command_parts = self.__stringify(script_tokens)
                    except Exception as e:
                        print(
                            f"Error parsing line {line_number} in '{script_filename}': {e}",
                            file=sys.stderr,
                        )
                        return

                    if not command_parts:
                        continue

                    cmd_name = command_parts[0]
                    cmd_args = GList(command_parts[1:])

                    if cmd_name in self.__commands:
                        _, callback = self.__commands[cmd_name]
                        try:
                            callback(cmd_args)
                        except SystemExit:
                           break
                        except Exception as e:
                            print(
                                f"Error executing command from line {line_number} in '{script_filename}' ('{line}'): {e}",
                                file=sys.stderr,
                            )
                            break
                    else:
                        print(f"Unknown command '{cmd_name}' on line {line_number} in '{script_filename}'.")

        except FileNotFoundError:
            print(f"Error: Script file not found: '{script_filename}'", file=sys.stderr)
        except IOError as e:
            print(f"Error reading script file '{script_filename}': {e}", file=sys.stderr)
        except Exception as e:
            self.__logger.error(f"An unexpected error occurred while processing script '{script_filename}': {e}")

    def _cmd_wait(self, args: GList[str]) -> None:
        if len(args) != 1:
            print('Usage: wait <time: float>')
            return

        try:
            wait_time = float(args[0])
            if wait_time < 0:
                print('Error: Wait time must be a non-negative number.')
                return
            time.sleep(wait_time)
        except ValueError:
            print('Error: Invalid wait time. Expected a float.')

    def __completer(self, text: str, state: int) -> Optional[str]:
        line = readline.get_line_buffer()
        words = line.lstrip().split()
        is_command_pos = (line.endswith(' ') or not words or (len(words) == 1 and not line.endswith(' ')))

        if readline.get_begidx() == 0 or is_command_pos:
             options = [cmd + ' ' for cmd in self.__commands.keys() if cmd.startswith(text)]
             try:
                 return options[state]
             except IndexError:
                 return None
        else:
             return None

    def __setup_readline(self):
        readline.read_history_file('../.history')
        atexit.register(readline.write_history_file, '../.history')

        readline.set_completer(self.__completer)
        readline.set_completer_delims(' \t\n;"\'')
        readline.parse_and_bind('tab: complete')

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

                if cmd_name in self.__commands:
                    _, callback = self.__commands[cmd_name]
                    try:
                        callback(args)
                    except SystemExit:
                        break
                    except Exception as e:
                        print(f"Error executing command '{cmd_name}': {e}", file=sys.stderr)
                else:
                    print(f"Unknown command: '{cmd_name}'. Type 'help' for available commands.")

            except Exception as e:
                self.__logger.error(f'An unexpected shell error occurred: {e}')