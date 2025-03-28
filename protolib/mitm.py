import socket
import threading
import logging
from typing import Tuple, Dict, Any
from proto3 import proto_decode, proto_encode, Connection, Encodings  # Assuming proto3.py exists and is accessible
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

CLIENT_PORT = 7942  # Original client port (where the client *thinks* it's connecting)
SERVER_PORT = 7943  # The *actual* server port
MITM_PORT = 7942 # MITM listens on the port the client uses
REAL_SERVER_ADDRESS = ('localhost', SERVER_PORT)


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

    return {
        'headers': headers,
        'payload': payload,
        'encoding': encoding
    }

def make_request(payload: bytes,
                 endpoint: str,
                 headers: Dict[str, Any] = None,
                 mimetype: str = 'application/octet-stream',
                 encoding: Encodings = Encodings.PLAIN) -> Dict[str, Any]:
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

def proxy_handler(client_socket: socket.socket, server_address: Tuple[str, int]):
    """Handles a single client connection, proxying data between client and server."""

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.connect(server_address)
        logging.info(f"Connected to real server at {server_address}")
    except ConnectionRefusedError:
        logging.error(f"Could not connect to real server at {server_address}")
        client_socket.close()
        return

    keep_alive = True
    while keep_alive:
        try:
            # Receive from client
            client_data = client_socket.recv(4096)
            if not client_data:
                logging.info("Client disconnected")
                break

            client_headers, client_payload, _ = proto_decode(client_data)
            logging.info(f"Received from client: Headers: {client_headers}, Payload size: {len(client_payload)}")

            logging.debug(f"full client data: {client_data}")

            # --- Modify Client Request (Optional) ---
            # Example: Change the endpoint
            # client_headers['Endpoint'] = 'modified/endpoint'
            # Example:  Add a header
            client_headers['X-MITM-Proxy'] = 'Active'

            # --- reconstruct client request --
            modified_client_request = proto_encode(**make_request(client_payload, client_headers['Endpoint'], client_headers))

            # Forward to server
            server_socket.sendall(modified_client_request)
            logging.info(f"Forwarded to server: {len(modified_client_request)} bytes")

            # Receive from server
            server_data = server_socket.recv(4096)
            if not server_data:
                logging.info("Server disconnected")
                break

            server_headers, server_payload, _ = proto_decode(server_data)

            logging.info(f"Received from server: Headers: {server_headers}, Payload Size: {len(server_payload)}")
            logging.debug(f"full server_data: {server_data}")
            # --- Modify Server Response (Optional) ---
            # Example:  Modify the payload
            # server_payload = b"Modified Server Response"
            server_headers['X-MITM-Modified'] = True  # Add a header indicating modification

            #reconstruct server response
            modified_server_response = proto_encode(**make_response(server_payload, server_headers.get('Status', 200), server_headers))
            # Forward to client
            client_socket.sendall(modified_server_response)
            logging.info(f"Forwarded to client: {len(modified_server_response)} bytes")

            connection = Connection(client_headers.get('Connection', 'close'))
            if connection == connection.CLOSE:
                keep_alive = False

        except Exception as e:
            logging.error(f"Error in proxy handler: {e}", exc_info=True)
            break


    server_socket.close()
    client_socket.close()
    logging.info("Proxy handler closed connections.")


def start_mitm_proxy(mitm_port: int, server_address: Tuple[str, int]):
    """Starts the MITM proxy server."""

    proxy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow rebinding to the port
    proxy_socket.bind(('localhost', mitm_port))
    proxy_socket.listen(5)
    logging.info(f"MITM proxy listening on port {mitm_port}, forwarding to {server_address}")

    try:
        while True:
            client_socket, client_address = proxy_socket.accept()
            logging.info(f"Accepted connection from {client_address}")
            # Handle each client connection in a separate thread.
            client_thread = threading.Thread(target=proxy_handler, args=(client_socket, server_address), daemon=True)
            client_thread.start()
    except KeyboardInterrupt:
        logging.info("MITM proxy shutting down.")
    finally:
        proxy_socket.close()



if __name__ == "__main__":
    start_mitm_proxy(MITM_PORT, REAL_SERVER_ADDRESS)