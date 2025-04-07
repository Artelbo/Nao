import socket
import logging
import threading
import time
from typing import Any, Tuple, Callable, Dict, Union, Optional
from .proto3 import proto_decode, proto_encode, Encodings, Connection
import hashlib


def make_response(payload: bytes,
                  response_status_code: int,
                  headers: Dict[str, Any] = None,
                  mimetype: str = 'application/octet-stream',
                  encoding: Encodings = Encodings.PLAIN) -> Dict[str, Any]:
    if headers is None:
        headers = {}

    headers['Timestamp'] = int(time.time())
    headers['Content-Type'] = headers.get('Content-Type') or mimetype
    headers['Content-Length'] = headers.get('Content-Length') or len(payload)
    headers['Content-Hash'] = headers.get('Content-Hash') or hashlib.md5(payload).hexdigest()
    headers['Status'] = headers.get('Status') or response_status_code
    if response_status_code == 0:
        del headers['Status']

    return {
        'headers': headers,
        'payload': payload,
        'encoding': encoding
    }


class ProtoSocketWrapper:
    def __init__(self, sock: socket.socket, encoding: Encodings, amount: int):
        self.__sock = sock
        self.__encoding = encoding
        self.__amount = amount

    def send(self, payload: bytes, headers: Dict[str, Any] = None) -> None:
        if headers is None:
            headers = {}

        data = proto_encode(**make_response(payload, 0, headers, encoding=self.__encoding))
        self.__sock.sendall(data)

    def receive(self) -> Tuple[bytes, Dict[str, Any], Encodings]:
        data = self.__sock.recv(self.__amount)
        headers, payload, encoding = proto_decode(data)
        return payload, headers, encoding


class SingleConnectionServer:
    def __init__(self, ip: str, port: int = 7942, recv_amount: int = 1024):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((ip, port))
        self.recv_amount = recv_amount
        self.routes: Dict[str, Callable] = {}

    def route(self, route: str):
        def decorator(
                func: Union[Callable[[Dict, bytes], Tuple[Dict, bytes, int]],
                            Callable[[Dict, bytes, ProtoSocketWrapper], None]]):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return {}, b'', 500
            self.routes[route] = wrapper
            return func
        return decorator

    def start(self):
        logging.info(f'Server started on {self.sock.getsockname()}')

        while True:
            self.sock.listen(1)

            try:
                client_sock, client_addr = self.sock.accept()
            except KeyboardInterrupt:
                break

            self.sock.listen(0)
            logging.info(f'Client connected: {client_addr}')

            keep_alive = True
            while keep_alive:
                try:
                    data = client_sock.recv(self.recv_amount)
                except (ConnectionAbortedError, ConnectionResetError):
                    logging.error(f'Client disconnected forcefully: {client_addr}')
                    break

                if len(data) == 0:
                    break

                headers, payload, _ = proto_decode(data)
                logging.debug(f'Received message size:{len(payload)} headers:{headers}')

                connection = Connection(headers.get('Connection', 'close'))
                if connection == connection.CLOSE:
                    keep_alive = False

                endpoint: Optional[str] = headers.get('Endpoint', None)
                if endpoint is None:
                    logging.warning('Missing endpoint in request headers')
                    response_data = proto_encode(**make_response(b'Missing Headers', 400))
                    client_sock.sendall(response_data)
                    continue

                if connection == connection.DUPLEX:
                    wrapper = ProtoSocketWrapper(client_sock, Encodings.PLAIN, self.recv_amount)
                    self.routes.get(endpoint)(headers, payload, wrapper)
                else:
                    response_headers, response_payload, status = self.routes.get(endpoint)(headers, payload)
                    response_data = proto_encode(**make_response(response_payload, status, response_headers,
                                                                 encoding=Encodings.GZIP))
                    client_sock.sendall(response_data)

            logging.info(f'Client disconnected: {client_addr}')
            client_sock.close()

        logging.info('Server stopped')
        self.sock.close()


class MultiConnectionServer:
    def __init__(self, ip: str, port: int = 7942, recv_amount: int = 1024):
        self.__logger = logging.getLogger('server')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((ip, port))
        self.recv_amount = recv_amount
        self.routes: Dict[str, Callable] = {}

    def route(self, route: str):
        def decorator(
                func: Union[Callable[[Dict, bytes], Tuple[Dict, bytes, int]],
                            Callable[[Dict, bytes, ProtoSocketWrapper], None]]):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.__logger.exception("Error in route handler:")
                    return {}, b'', 500
            self.routes[route] = wrapper
            return func
        return decorator

    def handle_client(self, client_sock, client_addr):
        self.__logger.info(f'Client connected: {client_addr}')

        keep_alive = True
        while keep_alive:
            try:
                data = client_sock.recv(self.recv_amount)
            except (ConnectionAbortedError, ConnectionResetError):
                self.__logger.error(f'Client disconnected forcefully: {client_addr}')
                break

            if len(data) == 0:
                break

            headers, payload, _ = proto_decode(data)
            self.__logger.debug(f'Received message (size: {len(payload)}, headers: {headers})')

            connection = Connection(headers.get('Connection', 'close'))
            if connection == Connection.CLOSE:
                keep_alive = False

            endpoint: Optional[str] = headers.get('Endpoint', None)
            if endpoint is None:
                self.__logger.warning('Missing endpoint in request headers')
                response_data = proto_encode(**make_response(b'Missing Headers', 400))
                client_sock.sendall(response_data)
                continue

            if connection == Connection.DUPLEX:
                wrapper = ProtoSocketWrapper(client_sock, Encodings.PLAIN, self.recv_amount)
                try:
                    self.routes.get(endpoint)(headers, payload, wrapper)
                except Exception as e:
                    self.__logger.exception(f'Error handling duplex connection: {e}')
            else:
                try:
                    response_headers, response_payload, status = self.routes.get(endpoint)(headers, payload)
                    response_data = proto_encode(**make_response(response_payload, status, response_headers, encoding=Encodings.GZIP))
                    client_sock.sendall(response_data)
                except Exception as e:
                    self.__logger.exception(f'Error handling request: {e}')

        self.__logger.info(f'Client disconnected: {client_addr}')
        client_sock.close()

    def start(self):
        self.__logger.info(f'Server started on {self.sock.getsockname()}')

        self.sock.listen(5)

        try:
            while True:
                client_sock, client_addr = self.sock.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, args=(client_sock, client_addr)
                )
                client_thread.start()

        except KeyboardInterrupt:
            self.__logger.info('Server stopped')
        finally:
            self.sock.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    server = MultiConnectionServer('127.0.0.1', 7942)

    @server.route('hello/world')
    def default_fallback(headers: Dict[str, Any], payload: bytes):
        return (
            {'Content-Type': 'application/json'},
            b'{"message": "Hello, World!"}',
            200
        )

    @server.route('echo')
    def default_fallback(headers: Dict[str, Any], payload: bytes, sock: ProtoSocketWrapper):
        while True:
            request_payload, request_headers, _ = sock.receive()
            if request_payload == b'':
                break
            sock.send(request_payload)

    server.start()