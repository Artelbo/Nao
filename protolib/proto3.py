import inspect
import time
from enum import Enum
from dataclasses import dataclass, is_dataclass, fields
from typing import List, Union, TypeVar, Annotated, get_type_hints, Any, Tuple, Dict, Optional
import struct
import orjson
import gzip
import io

numberT = TypeVar('numberT', int, float)


def int_to_bytes(v: int, b: int) -> bytes:
    return v.to_bytes(b, byteorder='big', signed=False)

def float_to_bytes(v: float) -> bytes:
    return struct.pack('>f', v)

def precise_float_to_bytes(v: float) -> bytes:
    return struct.pack('>d', v)  # Uses double precision (64-bit)

def bytes_to_int(v: bytes) -> int:
    return int.from_bytes(v, byteorder='big', signed=False)

def bytes_to_float(v: bytes) -> float:
    return struct.unpack('>f', v)[0]

def bytes_to_precise_float(v: bytes) -> float:
    return struct.unpack('>d', v)[0]


def check_annotated(func):
    hints = get_type_hints(func, include_extras=True)
    spec = inspect.getfullargspec(func)

    def wrapper(*args, **kwargs):
        if 'self' in spec.args and len(args) > 0 and is_dataclass(args[0]):
            obj = args[0]
            start_index = 1
        else:
            obj = None
            start_index = 0

        for idx, arg_name in enumerate(spec.args[start_index:]):
            hint = hints.get(arg_name)
            validators = getattr(hint, '__metadata__', None)
            if not validators:
                continue

            if obj:
                value = getattr(obj, arg_name)
            else:
                value = args[idx + start_index]

            for validator in validators:
                if hasattr(validator, 'validate_value'):
                    validator.validate_value(value)

        # Check keyword arguments
        for arg_name, value in kwargs.items():
            hint = hints.get(arg_name)
            validators = getattr(hint, '__metadata__', None)
            if not validators:
                continue
            for validator in validators:
                if hasattr(validator, 'validate_value'):
                    validator.validate_value(value)

        return func(*args, **kwargs)
    return wrapper


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


@dataclass
class ValueRange:
    min: numberT
    max: numberT

    def validate_value(self, name: str, x: numberT) -> None:
        if not (self.min <= x <= self.max):
            raise ValueError(f'{name} ({x}) must be in range [{self.min}, {self.max}]')


@dataclass
class StrMaxLength:
    max_length: int

    def validate_value(self, name: str, value: str) -> None:
        if len(value) > self.max_length:
            raise ValueError(f'{name} ({value}) exceeds max length of {self.max_length}')

class Encodings(Enum):
    PLAIN: int = 0
    GZIP: int = 1
    SECURE_PLAIN: int = 2
    SECURE_GZIP: int = 3


@dataclass
class SpecialByte(metaclass=ValidatedDataclass):
    encoding: Encodings
    version: Annotated[int, ValueRange(0, 63)]

    def encode(self) -> bytes:
        return int_to_bytes((self.encoding.value << 6 | self.version), 1)

    @staticmethod
    def decode(b: bytes) -> 'SpecialByte':
        value = bytes_to_int(b)
        encoding = (value >> 6) & 0b11
        version = value & 0b111111
        return SpecialByte(Encodings(encoding), version)


class HeaderValueType(Enum):
    INT: int = 0
    FLOAT: int = 1
    DOUBLE: int = 6
    STRING: int = 2
    JSON: int = 3
    BYTES: int = 5


@dataclass
class Header:
    name: Annotated[str, StrMaxLength(255)]  # 1 byte
    value: Union[int, float, str, dict[str, Any], list[Any]]  # n bytes

    def __encode_value(self, precise: bool) -> Tuple[HeaderValueType, bytes]:
        if isinstance(self.value, int):
            return HeaderValueType.INT, int_to_bytes(self.value, 8)
        elif isinstance(self.value, float):
            if precise:
                return HeaderValueType.DOUBLE, precise_float_to_bytes(self.value)
            return HeaderValueType.FLOAT, float_to_bytes(self.value)
        elif isinstance(self.value, str):
            return HeaderValueType.STRING, self.value.encode('utf-8')
        elif isinstance(self.value, (list, dict, bool)):
            try:
                return HeaderValueType.JSON, orjson.dumps(self.value)
            except orjson.JSONEncodeError:
                raise ValueError('Invalid value format')
        elif isinstance(self.value, (bytes, bytearray)):
            return HeaderValueType.BYTES, bytes(self.value)
        raise ValueError('Invalid value format')

    def encode(self, precise_floats: bool = False) -> bytes:
        encoded_name = self.name.encode('utf-8')
        value_type, encoded_value = self.__encode_value(precise_floats)

        return (int_to_bytes(len(encoded_name), 1) + encoded_name +
                int_to_bytes(value_type.value, 1) + int_to_bytes(len(encoded_value), 2) + encoded_value)

    @staticmethod
    def decode(b: bytes) -> 'Header':
        name_length = bytes_to_int(b[:1])
        name = b[1:1+name_length].decode('utf-8')
        value_type = HeaderValueType(bytes_to_int(b[1+name_length:2+name_length]))
        value_length = bytes_to_int(b[2+name_length:4+name_length])
        value_bytes = b[4+name_length:4+name_length+value_length]

        if value_type == HeaderValueType.INT:
            value = bytes_to_int(value_bytes)
        elif value_type == HeaderValueType.FLOAT:
            value = bytes_to_float(value_bytes)
        elif value_type == HeaderValueType.DOUBLE:
            value = bytes_to_precise_float(value_bytes)
        elif value_type == HeaderValueType.STRING:
            value = value_bytes.decode('utf-8')
        elif value_type == HeaderValueType.JSON:
            try:
                value = orjson.loads(value_bytes)
            except orjson.JSONDecodeError:
                raise ValueError('Invalid JSON format')
        elif value_type == HeaderValueType.BYTES:
            value = value_bytes
        else:
            raise ValueError('Unknown value type')

        return Header(name, value)


def compress_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9, mtime=0) as f:
        f.write(data)
    return buf.getvalue()


def decompress_bytes(compressed_data: bytes) -> bytes:
    if not compressed_data:
        raise ValueError('Input data cannot be None or empty.')

    with io.BytesIO(compressed_data) as buf:
        with gzip.GzipFile(fileobj=buf, mode='rb') as f:
            try:
                return f.read()
            except gzip.BadGzipFile as e:
                raise gzip.BadGzipFile(f'Invalid gzip data: {e}') from e
            except EOFError as e:
                raise gzip.BadGzipFile(f'Incomplete gzip data (likely truncated): {e}') from e
            except OSError as e:
                if 'Not a gzipped file' in str(e):
                    raise gzip.BadGzipFile(f'Not a gzipped file: {e}') from e
                raise


class Connection(Enum):
    CLOSE: str = 'close'
    KEEP_ALIVE: str = 'keep-alive'
    DUPLEX: str = 'duplex'


def proto_encode(headers: Dict[str, Union[int, float, str, dict[str, Any], list[Any]]],
                 payload: bytes,
                 encoding: Encodings = Encodings.PLAIN,
                 precise_floats: bool = False) -> bytes:
    sb = SpecialByte(encoding=encoding,
                     version=1).encode()

    converted_headers = []
    for name, value in headers.items():
        converted_headers.append(Header(name=name, value=value))

    if len(converted_headers) > 255:
        raise ValueError('Too many headers')

    encoded_headers = b''.join([int_to_bytes(len(h), 4) + h for h in [h.encode(precise_floats) for h in converted_headers]])

    data = (int_to_bytes(len(converted_headers), 1) + int_to_bytes(len(encoded_headers), 4) + encoded_headers +
            int_to_bytes(len(payload), 4) + payload)

    if encoding in (Encodings.GZIP, ):
        data = compress_bytes(data)

    return sb + data


def proto_decode(encoded_data: bytes) -> Tuple[Dict[str, Any], bytes, Encodings]:
    if len(encoded_data.strip()) <= 0:
        raise ValueError('Input data cannot be None or empty.')

    sb = SpecialByte.decode(encoded_data[0:1])

    data = encoded_data[1:]

    if sb.encoding in (Encodings.GZIP, ):
        data = decompress_bytes(data)

    num_headers = bytes_to_int(data[0:1])
    headers_length = bytes_to_int(data[1:5])
    headers_end = 5 + headers_length

    headers_bytes = data[5:headers_end]
    headers = {}
    offset = 0

    for _ in range(num_headers):
        header_length = bytes_to_int(headers_bytes[offset:offset + 4])
        offset += 4
        header = Header.decode(headers_bytes[offset:offset + header_length])
        headers[header.name] = header.value
        offset += header_length

    payload_length_start = headers_end
    payload_length = bytes_to_int(data[payload_length_start:payload_length_start + 4])
    payload_start = payload_length_start + 4
    payload = data[payload_start:payload_start + payload_length]

    return headers, payload, sb.encoding

if __name__ == '__main__':
    request = proto_encode(
        headers={
            'Timestamp': int(time.time()),
        },
        payload=b'Lorem Ipsum dolor sit amet, consectetur adipiscing elit',
        encoding=Encodings.GZIP
    )

    print(request)

    print(proto_decode(request))