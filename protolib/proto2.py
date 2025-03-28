import gzip
import io
import json
import struct
import time


def int_to_bytes(v, b):
    return ('%%0%dx' % (b * 2) % v).decode('hex')

def float_to_bytes(v):
    return struct.pack('>f', v)

def precise_float_to_bytes(v):
    return struct.pack('>d', v)  # Uses double precision (64-bit)

def bytes_to_int(v):
    return int(v.encode('hex'), 16)

def bytes_to_float(v):
    return struct.unpack('>f', v)[0]

def bytes_to_precise_float(v):
    return struct.unpack('>d', v)[0]


class Encodings:
    PLAIN = 0
    GZIP = 1
    SECURE_PLAIN = 2
    SECURE_GZIP = 3


class SpecialByte:
    def __init__(self, encoding, version):
        self.encoding = encoding  # type: int
        self.version = version  # type: int

    def encode(self):
        return int_to_bytes(self.encoding << 6 | self.version, 1)

    @staticmethod
    def decode(b):
        value = bytes_to_int(b)
        encoding = (value >> 6) & 0b11
        version = value & 0b111111
        return SpecialByte(encoding, version)


class HeaderValueType:
    INT = 0
    FLOAT = 1
    DOUBLE = 6
    STRING = 2
    JSON = 3
    BYTES = 5


class Header:
    def __init__(self, name, value):
        self.name = name  # type: str
        self.value = value  # this type: int | float | str | dict | list

    def __encode_value(self, precise):
        if isinstance(self.value, int):
            return HeaderValueType.INT, int_to_bytes(self.value, 8)
        elif isinstance(self.value, float):
            if precise:
                return HeaderValueType.DOUBLE, precise_float_to_bytes(self.value)
            return HeaderValueType.FLOAT, float_to_bytes(self.value)
        elif isinstance(self.value, str):
            return HeaderValueType.STRING, bytes(self.value.encode('utf-8'))
        elif isinstance(self.value, (list, dict, bool)):
            return HeaderValueType.JSON, bytes(json.dumps(self.value).encode('utf-8'))
        elif isinstance(self.value, (bytes, bytearray)):
            return HeaderValueType.BYTES, bytes(self.value)

    def encode(self, precise_floats):
        encoded_name = self.name.encode('utf-8')
        value_type, encoded_value = self.__encode_value(precise_floats)

        return int_to_bytes(len(encoded_name), 1) + encoded_name + \
               int_to_bytes(value_type, 1) + int_to_bytes(len(encoded_value), 2) + encoded_value

    @staticmethod
    def decode(b):
        name_length = bytes_to_int(b[:1])
        name = b[1:1+name_length].decode('utf-8')
        value_type = bytes_to_int(b[1+name_length:2+name_length])
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
            value = json.loads(value_bytes.decode('utf-8'))
        elif value_type == HeaderValueType.BYTES:
            value = value_bytes
        else:
            raise ValueError('Unknown value type')

        return Header(name, value)


def compress_bytes(data):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9, mtime=0) as f:
        f.write(data)
    return buf.getvalue()


def decompress_bytes(compressed_data):
    if not compressed_data:
        raise ValueError('Input data cannot be None or empty.')

    with io.BytesIO(compressed_data) as buf:
        with gzip.GzipFile(fileobj=buf, mode='rb') as f:
            try:
                return f.read()
            except EOFError as e:
                raise Exception('Incomplete gzip data (likely truncated): ' + str(e))
            except OSError as e:
                if 'Not a gzipped file' in str(e):
                    raise Exception('Not a gzipped file: ' + str(e))
                raise

class Connection:
    CLOSE = 'close'
    KEEP_ALIVE = 'keep-alive'
    DUPLEX = 'duplex'


def proto_encode(headers, payload, encoding=Encodings.PLAIN, precise_floats=False):
    sb = SpecialByte(encoding=encoding, version=1).encode()

    converted_headers = []
    for name, value in headers.iteritems():  # Python 2 uses iteritems()
        converted_headers.append(Header(name=name, value=value))

    if len(converted_headers) > 255:
        raise ValueError('Too many headers')

    encoded_headers = b''.join([int_to_bytes(len(h), 4) + h for h in [h.encode(precise_floats) for h in converted_headers]])

    data = (int_to_bytes(len(converted_headers), 1) + int_to_bytes(len(encoded_headers), 4) + encoded_headers +
            int_to_bytes(len(payload), 4) + payload)

    if encoding in (Encodings.GZIP, ):
        data = compress_bytes(data)

    return sb + data

def proto_decode(encoded_data):
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
            'Connection': Connection.KEEP_ALIVE
        },
        payload=b'Lorem Ipsum dolor sit amet, consectetur adipiscing elit',
        encoding=Encodings.PLAIN
    )

    print request.encode('hex')  # Python 2 print syntax
    with open('ilpiccoloscricci.bin', 'wb') as f:
        f.write(request)
    print proto_decode(request)