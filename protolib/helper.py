import os
import sys
from helper_parser import read_file
import orjson
from typing import TypedDict, List, Dict, Callable
import io
import importlib.util
import time

def import_function_from_file(filename: str, function_name: str,
                              injected_attrs: list = None, raise_if_not_found: bool = False) -> Callable | None:
    if injected_attrs is None:
        injected_attrs = []

    if not os.path.isfile(filename):
        raise FileNotFoundError(f'File "{os.path.basename(filename)}" does not exist.')

    basename = filename.split(os.sep)[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(basename, filename)
    module = importlib.util.module_from_spec(spec)
    for attr, value in injected_attrs:
        setattr(module, attr, value)
    spec.loader.exec_module(module)
    func = getattr(module, function_name, None)
    if func is None and raise_if_not_found:
        raise AttributeError(f'Function "{function_name}" not found in {os.path.basename(filename)}.')
    return func


class PackageJson(TypedDict):
    scripts: Dict[str, str]


package = os.path.join(os.getcwd(), 'package.json')
CURRENT_PYTHON_VERSION = '.'.join(map(str, sys.version_info[0:3]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python helper.py [script_name]')
        exit(1)


    with open(package, 'rb') as f:
        try:
            data: PackageJson = orjson.loads(f.read())
        except orjson.JSONDecodeError:
            print(f'Error parsing {package}')
            exit(1)


    for name, script in data['scripts'].items():
        if sys.argv[1] != name:
            continue

        parsed = read_file(io.StringIO(script), os.path.basename(package) + f' -> scripts -> {name}')
        if parsed.instruction == 'PYTHON':
            funcs = parsed.function_name
            if isinstance(parsed.function_name, str):
                funcs = [parsed.function_name]

            for fun in funcs:
                func: Callable | None = import_function_from_file(parsed.file, fun)
                if func is None:
                    print(f'Warning: Function "{fun}" not found in {parsed.file}.')
                    continue

                print(f'Running function "{fun}" in Python {CURRENT_PYTHON_VERSION}...')
                match parsed.python_version:
                    case CURRENT_PYTHON_VERSION:
                        st = time.perf_counter()
                        func()
                        elapsed = time.perf_counter() - st
                        # print(f'Function "{fun}" completed in {elapsed:.6f} seconds.')