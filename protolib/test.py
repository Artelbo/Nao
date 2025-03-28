from typing import Callable, Any, List, Tuple
import importlib


def get_functions_from_module(module_name: str) -> List[Tuple[str, Callable[[], bool]]]:
  try:
      module = importlib.import_module(module_name)
      functions = [
          (attr_name, getattr(module, attr_name))
          for attr_name in dir(module)
          if callable(getattr(module, attr_name)) and attr_name.startswith('test_')
      ]
      return functions
  except ImportError:
      return []

def run_tests() -> bool:
    """Runs all tests and reports results."""
    tests: List[Tuple[str, Callable[[], bool]]] = get_functions_from_module('tests')

    passed = 0
    for name, test in tests:
        try:
            if not test():
                print(f'Test "{name}" failed.\n')
                continue
            print('')
            passed += 1
        except Exception as e:
            print(f'Test "{name}" failed with an error: {str(e)}\n')
            continue

    if passed == len(tests):
        print(f'All tests passed. ({len(tests)})')
        return True
    else:
        print(f'{passed}/{len(tests)} tests passed.')
        return False