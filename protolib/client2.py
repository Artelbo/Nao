import socket
from proto2 import proto_decode, proto_encode, Encodings, Connection
import hashlib
import time
from pprint import pprint


def make_request(payload,
                 endpoint,
                 headers=None,
                 mimetype='application/octet-stream',
                 encoding=Encodings.PLAIN):
    if headers is None:
        headers = {}

    headers['Timestamp'] = int(time.time())
    headers['Content-Type'] = headers.get('Content-Type') or mimetype
    headers['Content-Length'] = headers.get('Content-Length') or len(payload)
    headers['Content-Hash'] = headers.get('Content-Hash') or hashlib.md5(payload).hexdigest()
    headers['Endpoint'] = endpoint
    headers['Connection'] = headers.get('Connection', 'close')

    return {
        'headers': headers,
        'payload': payload,
        'encoding': encoding
    }


class ProtoSocketWrapper:
    def __init__(self, sock, endpoint, encoding):
        self.__sock = sock
        self.__endpoint = endpoint
        self.__encoding = encoding

    def send(self, payload, headers=None):
        if headers is None:
            headers = {}

        data = proto_encode(**make_request(payload, self.__endpoint, headers, encoding=self.__encoding))
        self.__sock.sendall(data)

    def receive(self):
        data = self.__sock.recv(1024)
        headers, payload, encoding = proto_decode(data)
        return payload, headers, encoding


class Session:
    def __init__(self, server, maintain_connection=False):
        self.server = server
        self.sock = None
        self.connected = False
        self.maintain_connection = maintain_connection

    def request(self,
                endpoint,
                payload=b'',
                headers=None,
                maintain_connection=None):
        if headers is None:
            headers = {}

        if maintain_connection is None:
            maintain_connection = self.maintain_connection

        if maintain_connection:
            headers['Connection'] = Connection.KEEP_ALIVE
        encoded_request = proto_encode(**make_request(payload, endpoint, headers,
                                                      encoding=Encodings.GZIP))

        if not self.connected:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(self.server)
            self.connected = True

        self.sock.sendall(encoded_request)

        response = self.sock.recv(1024)
        decoded_response = proto_decode(response)

        if not maintain_connection:
            self.connected = False
            self.sock.close()

        return decoded_response

    def stream(self,
               callback,
               endpoint, payload=b'', headers=None):
        if headers is None:
            headers = {}

        headers['Connection'] = Connection.DUPLEX
        encoded_request = proto_encode(**make_request(payload, endpoint, headers,
                                                      encoding=Encodings.GZIP))

        if not self.connected:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(self.server)
            self.connected = True

        self.sock.sendall(encoded_request)

        wrapper = ProtoSocketWrapper(self.sock, endpoint, Encodings.GZIP)
        callback(wrapper)

        self.connected = False
        self.sock.close()

    def close(self):
        if self.connected:
            self.sock.close()
            self.connected = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()


if __name__ == '__main__':
    with Session(('localhost', 7942), True) as session:
        response_headers, response_payload, _ = session.request('hello/world', b'')
        pprint(response_headers, indent=4)
        print(response_payload)

        response_headers, response_payload, _ = session.request('hello/world', b'')
        pprint(response_headers, indent=4)
        print(response_payload)

        def stream_callback(sock):
            while True:
                data = raw_input('chat: ').encode('utf-8')
                sock.send(data)
                if data == b'':
                    break

                response_payload, response_headers, _ = sock.receive()
                print('Received: {}'.format(response_payload.decode('utf-8')))

        session.stream(stream_callback, 'echo', b'')
