import sys
import warnings

__PROTO_NAME__ = 'BeepBoopNet'

if sys.version_info < (3,):
    warnings.warn('Detected Python 2', Warning)
