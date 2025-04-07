import logging.config
import yaml
import os
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    TypeVar,
    TypedDict,
    get_type_hints,
    Type,
    Literal,
    get_origin,
    get_args,
    Union,
    Optional,
)
import argparse
import json
import sys


class Config(TypedDict):
    locale: str


class ArgsConfig(TypedDict, total=False):
    locale: str


T = TypeVar('T')
TD = TypeVar('TD', bound=TypedDict)


def setup_logging(log_config: Union[str, os.PathLike]) -> None:
    if not os.path.exists(log_config):
        raise FileNotFoundError(f'Log config file "{log_config}" not found')

    with open(log_config, 'r') as f:
        config = yaml.safe_load(f.read())

    if not config or 'handlers' not in config or 'file' not in config['handlers'] or 'filename' not in \
            config['handlers']['file']:
        logging.warning(
            f"Log configuration in '{log_config}' is missing expected structure for directory creation. Skipping.")
        log_dir = None
    else:
        log_dir = os.path.dirname(config['handlers']['file']['filename'])

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            logging.error(f'Could not create log directory {log_dir}: {e}')
            pass

    logging.config.dictConfig(config)


def is_valid_dict(data: Dict[str, Any], c: Type[TD]) -> bool:
    if not isinstance(data, dict):
        raise ValueError('Data is not a dictionary')

    try:
        hints = get_type_hints(c)
        required_keys = getattr(c, '__required_keys__', set())
        optional_keys = getattr(c, '__optional_keys__', set(hints.keys()))
        is_total = getattr(c, '__total__', True)
        if not is_total:
            required_keys = set()
            optional_keys = set(hints.keys())
        else:
            if not required_keys and not optional_keys and hints:
                required_keys = set(hints.keys())

        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise SyntaxError(f'Missing required keys: {missing_keys}')

        all_defined_keys = required_keys.union(optional_keys)

        for key, value in data.items():
            if key in hints:
                expected_type = hints[key]
                origin = get_origin(expected_type)
                args = get_args(expected_type)

                if origin is Union and type(None) in args:
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    if len(non_none_args) == 1:
                        actual_expected_type = non_none_args[0]
                    else:
                        actual_expected_type = tuple(non_none_args)
                else:
                    actual_expected_type = expected_type

                origin_actual = get_origin(actual_expected_type)
                args_actual = get_args(actual_expected_type)

                if origin_actual is list or origin_actual is List:
                    if args_actual and len(args_actual) == 1:
                        inner_type = args_actual[0]
                        if not isinstance(value, list) or not all(isinstance(item, inner_type) for item in value):
                            raise TypeError(f"Key '{key}' has incorrect type. Expected List[{inner_type.__name__}], got {type(value).__name__} or incorrect item types.")
                    elif not isinstance(value, list):
                        raise TypeError(f"Key '{key}' has incorrect type. Expected List, got {type(value).__name__}.")
                elif origin_actual is Literal:
                    if not value in args_actual:
                        raise TypeError(f"Key '{key}' has incorrect value. Expected one of {args_actual}, got {value!r}.")
                elif isinstance(actual_expected_type, tuple):
                    if not isinstance(value, actual_expected_type):
                        type_names = ', '.join(t.__name__ for t in actual_expected_type)
                        raise TypeError(f"Key '{key}' has incorrect type. Expected one of ({type_names}), got {type(value).__name__}.")
                elif not isinstance(value, actual_expected_type):
                    raise TypeError(f"Key '{key}' has incorrect type. Expected {getattr(actual_expected_type, '__name__', repr(actual_expected_type))}, got {type(value).__name__}.")

        extra_keys = data.keys() - all_defined_keys
        if extra_keys:
            if is_total:
                raise SyntaxError(f'Found unexpected keys: {extra_keys}')
    except Exception as e:
        raise
    return True


def create_parser_from_typeddict(desc: str, typeddict_cls: Type[TD]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    try:
        hints = get_type_hints(typeddict_cls)
        annotations = getattr(typeddict_cls, '__annotations__', {})
        is_total = getattr(typeddict_cls, '__total__', True)

        required_keys = set()
        optional_keys = set()

        if not is_total:
            optional_keys = set(hints.keys())
        else:
            for k, v_hint in hints.items():
                origin = get_origin(v_hint)
                args = get_args(v_hint)
                is_optional = (origin is Union and type(None) in args)
                if is_optional:
                    optional_keys.add(k)
                else:
                    required_keys.add(k)

        if hasattr(typeddict_cls, '__required_keys__'):
            required_keys = getattr(typeddict_cls, '__required_keys__', set())
        if hasattr(typeddict_cls, '__optional_keys__'):
            optional_keys = getattr(typeddict_cls, '__optional_keys__', set())
    except Exception as e:
        raise Exception(f'Error introspecting TypedDict {typeddict_cls.__name__}: {e}')

    for key, hint in hints.items():
        arg_name = f"--{key.replace('_', '-')}"
        is_required_arg = key in required_keys and key not in optional_keys

        origin = get_origin(hint)
        args = get_args(hint)
        base_hint = hint
        is_explicitly_optional = False

        if origin is Union and type(None) in args:
            is_explicitly_optional = True
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                base_hint = non_none_args[0]
                origin = get_origin(base_hint)
                args = get_args(base_hint)
            else:
                print(f"Warning: Skipping argument '{key}'. Cannot automatically handle complex Union type hint: {hint}", file=sys.stderr)
                continue

        kwargs: Dict[str, Any] = {'help': f'Specify the {key}.', 'dest': key}
        processed = False

        type_name = getattr(base_hint, '__name__', str(base_hint))

        if base_hint is str:
            kwargs['type'] = str
            kwargs['help'] += ' Type: string.'
            processed = True
        elif base_hint is int:
            kwargs['type'] = int
            kwargs['help'] += ' Type: integer.'
            processed = True
        elif base_hint is float:
            kwargs['type'] = float
            kwargs['help'] += ' Type: float.'
            processed = True
        elif base_hint is bool:
            kwargs['action'] = 'store_true'
            kwargs['help'] = f"Enable the '{key}' flag."
            if is_explicitly_optional:
                kwargs['default'] = False
                kwargs['required'] = False
            else:
                print(f"Warning: Required boolean '{key}' mapped to store_true flag. Consider alternative representation.", file=sys.stderr)
                kwargs['required'] = False
                kwargs['default'] = False
            processed = True
        elif origin is list or origin is List:
            if args and len(args) == 1 and args[0] in (str, int, float):
                inner_type = args[0]
                kwargs['type'] = inner_type
                kwargs['nargs'] = '*'
                kwargs['help'] += f' Type: list of {inner_type.__name__} (0 or more values).'
                processed = True
            else:
                print(f"Warning: Skipping argument '{key}'. Unsupported List type hint: {hint}", file=sys.stderr)
        elif origin is Literal:
            if args and all(isinstance(a, (str, int, float)) for a in args):
                choice_types = set(type(a) for a in args)
                if len(choice_types) == 1:
                    arg_type = choice_types.pop()
                    kwargs['choices'] = args
                    kwargs['type'] = arg_type
                    kwargs['help'] += f" Choices: {', '.join(map(str, args))}."
                    processed = True
                else:
                    print(f"Warning: Skipping argument '{key}'. Literal choices have mixed types: {hint}", file=sys.stderr)
            else:
                print(f"Warning: Skipping argument '{key}'. Literal choices must be str, int, or float: {hint}", file=sys.stderr)

        if 'action' not in kwargs:
            if is_required_arg and not is_explicitly_optional:
                kwargs['required'] = True
            else:
                kwargs['required'] = False
                kwargs['default'] = None

        if processed:
            parser.add_argument(arg_name, **kwargs)
        elif key in hints:
            print(f"Warning: Skipping argument '{key}'. Unsupported type hint: {hint}", file=sys.stderr)

    return parser


def load_config(config_path: Union[str, os.PathLike]) -> Tuple[Config, argparse.ArgumentParser]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file "{config_path}" not found')

    with open(config_path, 'r') as f:
        config_data: Dict[str, Any] | None = yaml.safe_load(f.read())

    if config_data is None:
        raise ValueError('Config file is empty or invalid YAML')

    if is_valid_dict(config_data, Config):
        validated_config = config_data
        parser = create_parser_from_typeddict('Override config options from command line', ArgsConfig)
        return validated_config, parser
    else:
        raise ValueError('Invalid config structure (this should not happen if is_valid_dict raises errors)')


def path_to_os_specific(path: str) -> str:
    return path.replace('/', os.sep).replace('\\', os.sep)


def load_locale(locale_path: Union[str, os.PathLike]) -> Dict:
    if not os.path.exists(locale_path):
        raise FileNotFoundError(f'Locale file "{locale_path}" not found')
    with open(locale_path, 'r', encoding='utf-8') as f:
        return json.load(f)
