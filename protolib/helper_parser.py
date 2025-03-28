import re
from typing import List, Dict, Any, Callable, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import os
from io import StringIO, BytesIO, TextIOWrapper
import ast
import sys

@dataclass
class Token:
    value: Optional[str]
    type: str
    line: Optional[int] = None
    char: Optional[int] = None


class Instructions(Enum):
    PYTHON = 'python'
    COMMAND = 'cmd'


@dataclass
class Code:
    instruction: Instructions
    data: Any
    optional_hinting: Any = None


@dataclass
class ImportInstruction:
    path: str
    is_relative: bool

@dataclass
class DataDeclaration:
    name: str
    data: list[Code] | Any

@dataclass
class DataValueDeclaration:
    name: str
    data: list[Token] | Any


class ParsingError(Exception):
    pass


class ArrayType(list):
    def __init__(self, elements: List[Any]):
        super().__init__(elements)

class CodeBlockType(list):
    def __init__(self, elements: List[Any]):
        super().__init__(elements)

def find_in_path(filename: str) -> os.PathLike | str | None:
    for directory in os.getenv('PATH', '').split(os.pathsep):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):
            return full_path
    return None


def find_file(filename: str, directories: list[str | os.PathLike], also_path: bool = True) -> os.PathLike | str | None:
    for directory in directories:
        if not directory:
            continue
        full_path = os.path.abspath(os.path.join(directory, filename))
        if os.path.isfile(full_path):
            return full_path
    if also_path:
        r = find_in_path(filename)
        if r:
            return r
    return None



def tokenize_code(text: str) -> List[Token]:
    # Define token types and their patterns
    token_patterns = {
        'KEYWORD': r'\b('
                   r'python|cmd'
                   r')\b',                         # Keywords
        'VALUE': r'\b('
                 r'null|true|false'
                 r')\b',                           # Keywords like `true` or `false`
        'IDENTIFIER': r'\b[a-zA-Z_]\w*\b',         # Variables, function names (lowercase identifiers)
        'STRING': r'"[^"]*"|\'[^\']*\'',           # String literals
        'NUMBER': r'\b\d+(\.\d*)?|\.\d+\b',        # Integer and floating-point numbers
        'OPERATOR': r'=|\-\>|\+|\-|\*|\/|\^|\%',   # Operators
        'PATH': r'\<[a-zA-Z0-9\-_\.\*]+\>',        # Paths
        'NEWLINE': r'\n',                          # Newline separator
        'EOL': r';',                               # End of line
        'PUNCTUATION': r'[\.\,\:\;\(\)\[\]\{\}]',  # Punctuation
    }

    comment_patterns = {
        'SINGLE_LINE_COMMENT': r'//.*?$',
        'MULTI_LINE_COMMENT': r'/\*.*?\*/',
    }

    # Combine complex patterns first
    combined_pattern = ''
    combined_pattern += '|'.join(f'(?P<{name}>{pattern})' for name, pattern in comment_patterns.items()) + '|'
    combined_pattern += '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_patterns.items())

    # Compile the combined regex
    tokenizer = re.compile(combined_pattern, re.DOTALL | re.MULTILINE)

    tokens: List[Token] = []
    line_number = 1
    line_start = 0
    char_position = 1

    for match in tokenizer.finditer(text):
        token_type: str = match.lastgroup
        token_value: str = match.group()
        char_position = match.start() - line_start + 1

        if token_type == 'NEWLINE':
            tokens.append(Token(value='\n', type='NEWLINE', line=line_number, char=char_position))
            line_number += 1
            line_start = match.end()
            continue

        if token_type in comment_patterns:
            continue

        if token_type == 'PATH' or token_type == 'STRING':
            token_value = token_value[1:-1]

        tokens.append(Token(value=token_value, type=token_type, line=line_number, char=char_position))

    return tokens + [Token(value=None, type='EOF', line=line_number, char=char_position)]

def read_and_tokenize(file: os.PathLike | str) -> List[Token]:
    if not os.path.isfile(file):
        raise FileNotFoundError(f'file not found: {file}')

    with open(file, 'r') as f:
        text = f.read()

    return tokenize_code(text)


class OBJ:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        for key, value in values.items():
            if isinstance(value, dict):
                setattr(self, key, OBJ(value))
                continue
            setattr(self, key, value)

    @property
    def _dict(self) -> dict[str, Any]:
        return dict([(key, value) for key, value in vars(self).items() if not key.startswith('_')])

    def __repr__(self) -> str:
        return f'{self._dict}'

    def __getattr__(self, item):
        return None

class Parser:
    def __init__(self, tokens: List[Token], _extract_blocks: bool = True):
        self.tokens = tokens
        self.current_index = 0

        if _extract_blocks:
            self.tokens = self.extract_code_blocks()
            self.begin()

    def begin(self) -> None:
        self.current_index = 0

    def advance(self, n: int = 1) -> None:
        if not isinstance(n, int):
            raise ValueError('n must be an integer')
        self.current_index += n

    def peek(self, offset: int = 0, prefer_error: bool = True) -> Token:
        if not isinstance(offset, int):
            raise ValueError('offset must be an integer')
        if self.current_index + offset >= len(self.tokens):
            if prefer_error:
                raise ValueError('reached end of tokens')
            return Token(value=None, type='EOF')
        return self.tokens[self.current_index + offset]

    @property
    def token(self) -> Token | List[Token | list[...]]:
        try:
            return self.tokens[self.current_index]
        except IndexError:
            return Token(value=None, type='EOF')

    def extract_code_blocks(self) -> List[Any]:
        result = []

        def extract_block(opening: str, closing: str) -> Any:
            block = []
            self.advance()

            while self.token.type != 'EOF' and self.token.value != closing:
                if self.token.value == opening:
                    block.append(extract_block(opening, closing))
                elif self.token.value in ('{', '[', '('):
                    block.append(extract_block(self.token.value, {'{': '}', '[': ']', '(': ')'}[self.token.value]))
                else:
                    block.append(self.token)
                self.advance()

            if self.token.value == closing:
                return CodeBlockType(block) if opening == '{' else ArrayType(block)
            else:
                raise ValueError(f'ERROR {self.token.line}:{self.token.char} -> expected closing {closing}')

        while self.token.type != 'EOF':
            if self.token.value in ('{', '[', '('):
                result.append(extract_block(self.token.value, {'{': '}', '[': ']', '(': ')'}[self.token.value]))
            else:
                result.append(self.token)
            self.advance()

        return result + [Token(value=None, type='EOF')]

    @staticmethod
    def parse_expression(tokens: list[Token]) -> Token:
        # TODO: somestuff
        return tokens[0]

    @staticmethod
    def group_between_comas(tokens: list[Token | list[...]]) -> list[Token]:
        if not isinstance(tokens, list):
            return tokens

        result: List[list[list[Token]]] = []
        group: List[list[Token]] = []

        for token in tokens:
            if isinstance(token, Token) and token.type == 'NEWLINE':
                continue
            elif isinstance(token, Token) and token.type == 'PUNCTUATION' and token.value == ',':
                result.append(group)
                group = []
            else:
                group.append(token)

        if group:
            result.append(group)

        return [Parser.parse_expression(r) for r in result if r]


    def parse_code(self, parsing_file: os.PathLike | str = '') -> List[Code]:
        code: List[Code] = []

        while self.token.type != 'EOF':
            match self.token.type:
                case 'NEWLINE' | 'EOL':
                    self.advance()
                    continue

                case 'KEYWORD':
                    if self.token.value == 'python':
                        self.advance()
                        if isinstance(self.token, list):
                            version = self.token[0]
                        else:  # type: Token
                            version = self.token

                        if version.type == 'IDENTIFIER':
                            if version.value in ('current', 'any'):
                                version.type = 'STRING'
                                version.value = '.'.join(map(str, sys.version_info[0:3]))

                        if version.type != 'STRING':
                            raise SyntaxError('ERROR {version.line}:{version.char} -> expected version number as string')

                        if not any((c in set('0123456789.')) for c in version.value):
                            raise SyntaxError('ERROR {version.line}:{version.char} -> version number must contain only digits and periods')

                        self.advance(1)  # skip version
                        if self.peek().type == 'PUNCTUATION':
                            self.advance(1)  # skip some separator


                        if self.token.type not in ('STRING', 'PATH'):
                            raise SyntaxError('ERROR {self.token.line}:{self.token.char} -> expected string or path')

                        file = self.token

                        if not os.path.isfile(file.value):
                            raise FileNotFoundError(f'ERROR {self.token.line}:{self.token.char} -> file not found: {file.value}')

                        self.advance(1)  # skip file path
                        if isinstance(self.peek(), Token) and self.peek().type == 'PUNCTUATION':
                            self.advance(1)  # skip some separator

                        if not isinstance(self.token, list) and self.token.type not in ('STRING', 'IDENTIFIER'):
                            raise SyntaxError('ERROR {self.token.line}:{self.token.char} -> expected function/s identifier')

                        function_name = self.token

                        code.append(Code(
                            instruction=Instructions.PYTHON,
                            data={
                                'version': self.token_to_value(version),
                                'file': self.token_to_value(file),
                                'function_name': self.token_to_value(function_name),
                            }
                        ))

                    elif self.token.value == 'cmd':
                        self.advance(1)
                        if isinstance(self.peek(), Token) and self.peek().type == 'PUNCTUATION':
                            self.advance(1)
                        x = []
                        while self.token.type != 'EOF':
                            x.append(self.token)
                            self.advance()

                        if len(x) == 1:
                            if x[0].type == 'STRING':
                                x = x[0].value.split(' ')
                                x = [Token(value=s, type='STRING') for s in x]

                        code.append(Code(
                            instruction=Instructions.COMMAND,
                            data=self.token_to_value(x)
                        ))

                case _:
                    raise ValueError(f'ERROR {self.token.line}:{self.token.char} -> unexpected token: {self.token.value}')

            self.advance()

        return code

    @staticmethod
    def code_to_obj(code: list[Code]) -> OBJ:
        obj = OBJ({})
        for c in code:
            if c.instruction == Instructions.PYTHON:
                setattr(obj, 'instruction', 'PYTHON')
                setattr(obj, 'python_version', c.data['version'])
                setattr(obj, 'file', c.data['file'])
                setattr(obj, 'function_name', c.data['function_name'])
            elif c.instruction == Instructions.COMMAND:
                setattr(obj, 'instruction', 'COMMAND')
                setattr(obj, 'command', c.data)
        return obj

    @staticmethod
    def token_to_value(token: Token | List) -> int | float | str | bool | None | list:
        if isinstance(token, list):
            return [Parser.token_to_value(t) for t in token if t.type != 'PUNCTUATION']
        elif isinstance(token, str):
            return token
        elif token.type == 'NUMBER':
            return int(token.value) if token.value.isdigit() else float(token.value)
        elif token.type in ('STRING', 'PATH', 'IDENTIFIER'):
            return token.value
        elif token.type == 'VALUE':
            if token.value == 'null':
                return None
            elif token.value == 'true':
                return True
            elif token.value == 'false':
                return False
        elif token.type == 'ARRAY':
            return ast.literal_eval(token.value)
        elif token.type in ('NEWLINE', 'PUNCTUATION'):
            return None
        else:
            raise ValueError(f'ERROR {token.line}:{token.char} -> unexpected token type: {token.type}')


def read_file(file: os.PathLike | str | StringIO | BytesIO | TextIOWrapper, filename: Optional[str] = None) -> OBJ:
    if isinstance(file, (os.PathLike, str)):
        filename = filename or os.path.basename(file)
        with open(file, 'r') as f:
            file: str = f.read()
    elif isinstance(file, StringIO):
        filename = filename or '<internals>'
        file: str = file.getvalue()
    elif isinstance(file, BytesIO):
        filename = filename or '<internals>'
        file: str = file.getvalue().decode('utf-8')
    elif isinstance(file, TextIOWrapper):
        filename = filename or '<internals>'
        file: str | bytes = file.read()
        if isinstance(file, bytes):
            file: str = file.decode('utf-8')
    else:
        raise ValueError('Invalid file type. Expected a file path, a string, or a BytesIO object.')

    try:
        tokens = tokenize_code(file)
        parser = Parser(tokens)
        return Parser.code_to_obj(parser.parse_code(filename))
    except Exception as e:
        raise ParsingError(f'{e}')