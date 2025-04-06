import sys
from typing import Tuple
import os
import logging
import locale
from dotenv import load_dotenv
import threading
from libs.config import setup_logging, load_config, load_locale
from libs.nao import NAO
from libs.virtualnao import VirtualNAO, TestClient


# Initial config
setup_logging('log.yaml')
logger = logging.getLogger('main')

# Config
config, parser = load_config('config.yaml')
# -- additional parser arguments
parser.add_argument('mode',
                    help='Specify run mode.',
                    type=str,
                    choices=['deploy', 'dev'])
parser.add_argument('-a', '--address',
                    help='Bot IPv4 address.',
                    type=str,
                    default='172.16.222.213')
parser.add_argument('-p', '--port',
                    help='Bot proto port.',
                    type=int,
                    default=9559)
parser.add_argument('-s', '--ssh',
                    help='Bot SSH user and password.',
                    type=str,
                    default='nao:nao')
# --
args = parser.parse_args()
config.update({k: v for k, v in vars(args).items() if v is not None})

try:
    locale.setlocale(locale.LC_TIME, config['locale'])
    locale_data = load_locale(os.path.join(os.getcwd(), 'locales', config['locale']+'.json'))
except Exception:
    logger.critical(f"Could not set locale to '{config['locale']}'.")
    exit(1)

try:
    load_dotenv('.env')
except FileNotFoundError:
    logger.warning('.env not found')

if not os.path.exists('.history'):
    with open('.history', 'w') as f:
        logger.warning('.history file not found, creating...')


if __name__ == '__main__':
    BOT: Tuple[str, int] = config['address'], config['port']
    if len(config['ssh'].split(':', 1)) <= 1:
        print('Invalid --ssh value')
        sys.exit(1)
    SSH_USER, SSH_PASS = config['ssh'].split(':', 1)
    BOT_SSH: Tuple[str, str] = SSH_USER, SSH_PASS
    del SSH_USER, SSH_PASS

    match config['mode']:
        case 'deploy':
            try:
                nao = NAO(BOT, BOT_SSH, locale_data=locale_data)
            except RuntimeError:
                sys.exit(1)
            nao.start_shell()
            nao.close()
            sys.exit(0)

        case 'dev':
            nao = VirtualNAO(locale_data=locale_data)

            client = TestClient()
            threading.Thread(target=client.start, daemon=True).start()

            nao.start_shell()
            nao.close()

            sys.exit(0)
        case _:
            print('Invalid mode')
            sys.exit(1)