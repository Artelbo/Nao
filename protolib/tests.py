import sys
import time
from typing import Callable, Any

from proto3 import SpecialByte, Encodings


def timeit(func: Callable) -> Callable:
    """Decorator that times the execution of a function using perf_counter."""
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f'Function "{func.__name__}" completed in {execution_time:.6f} seconds.')
        return result
    return wrapper


@timeit
def test_special_byte() -> bool:
    sb = SpecialByte(encoding=Encodings.PLAIN, version=1).encode()
    if sb == b'\x01':
        return True
    else:
        print(f"Test error: {sys._getframe().f_code.co_name}: Expected b'\\x01', got {sb}")
        return False


@timeit
def test_special_byte_different_version() -> bool:
    sb = SpecialByte(encoding=Encodings.PLAIN, version=2).encode()
    if sb == b'\x02':
        return True
    else:
        print(f"Test error: {sys._getframe().f_code.co_name}: Expected b'\\x02', got {sb}")
        return False


@timeit
def test_special_byte_different_encoding() -> bool:
    sb = SpecialByte(encoding=Encodings.GZIP, version=0).encode()
    if sb == b'@':
        return True
    else:
        print(f"Test error: {sys._getframe().f_code.co_name}: Expected b'@', got {sb}")
        return False


@timeit
def test_special_byte_different() -> bool:
    sb = SpecialByte(encoding=Encodings.GZIP, version=52).encode()
    if sb == b't':
        return True
    else:
        print(f"Test error: {sys._getframe().f_code.co_name}: Expected b't', got {sb}")
        return False


@timeit
def test_special_byte_wrong() -> bool:
    try:
        SpecialByte(encoding=Encodings.PLAIN, version=64).encode()
        SpecialByte(encoding=Encodings.PLAIN, version=-2).encode()
        print(f"Test error: {sys._getframe().f_code.co_name}: Expected ValueError('invalid input version')")
        return False
    except ValueError:
        return True
