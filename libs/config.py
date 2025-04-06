import logging.config
import yaml
import os
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    override,
    TypeVar,
    TypedDict,
    NotRequired,
    TypeGuard,
    get_type_hints,
    Type,
    Literal,
    get_origin,
    get_args,
    Union,
)
import argparse
import json


class Config(TypedDict):
    locale: str


class ArgsConfig(TypedDict):
    locale: NotRequired[str]


T = TypeVar('T')
TD = TypeVar('TD', bound=TypedDict)


def setup_logging(log_config: str | os.PathLike) -> None:
    if not os.path.exists(log_config):
        raise FileNotFoundError(f'Log config file "{log_config}" not found')

    with open(log_config, 'r') as f:
        config = yaml.safe_load(f.read())

    log_dir = os.path.dirname(config['handlers']['file']['filename'])
    try:
        os.makedirs(log_dir)
    except OSError:
        pass

    logging.config.dictConfig(config)


def is_valid_dict(data: Dict[str, Any], c: Type[TD]) -> TypeGuard[TD]:
    if not isinstance(data, dict):
        raise ValueError('Data is not a dictionary')

    try:
        hints = get_type_hints(c)
        required_keys = getattr(c, '__required_keys__', set(k for k, v in hints.items()
                                                            if not isinstance(v, type(NotRequired))))
        optional_keys = getattr(c, '__optional_keys__', set(k for k, v in hints.items()
                                                            if isinstance(v, type(NotRequired))))

        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise SyntaxError(f'Missing required keys: {missing_keys}')

        all_defined_keys = required_keys.union(optional_keys)
        for key, value in data.items():
            if key in hints:
                expected_type = hints[key]
                origin = getattr(expected_type, '__origin__', None)
                args = getattr(expected_type, '__args__', ())

                if origin is NotRequired and args:
                    actual_expected_type = args[0]
                else:
                    actual_expected_type = expected_type

                if not isinstance(value, actual_expected_type):
                    raise TypeError(f"Key '{key}' has incorrect type. Expected {actual_expected_type.__name__}, got {type(value).__name__}.")

        extra_keys = data.keys() - all_defined_keys
        if extra_keys:
            is_total = getattr(c, '__total__', True)
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
        hints = get_type_hints(typeddict_cls, include_extras=True)
        annotations = getattr(typeddict_cls, '__annotations__', {})
        required_keys = set(getattr(typeddict_cls, '__required_keys__', set()))
        optional_keys = set(getattr(typeddict_cls, '__optional_keys__', set()))
        if not required_keys and not optional_keys and annotations:
            is_total = getattr(typeddict_cls, '__total__', True)
            for k, v_hint in hints.items():
                origin = get_origin(v_hint)
                args = get_args(v_hint)
                if origin is NotRequired or \
                   (origin is Union and type(None) in args) or \
                   not is_total:
                    optional_keys.add(k)
                else:
                    required_keys.add(k)
        all_keys_from_hints = set(hints.keys())
        inferred_optional = all_keys_from_hints - required_keys
        optional_keys.update(inferred_optional)

    except Exception as e:
        raise Exception(f'Error introspecting TypedDict {typeddict_cls.__name__}: {e}')

    for key, hint in hints.items():
        arg_name = f"--{key.replace('_', '-')}"
        is_required = key in required_keys
        origin = get_origin(hint)
        args = get_args(hint)
        base_hint = hint
        is_optional_type = False
        if origin is NotRequired and args:
            base_hint = args[0]
            origin = get_origin(base_hint)
            args = get_args(base_hint)
        elif origin is Union and type(None) in args:
            is_optional_type = True
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                base_hint = non_none_args[0]
                origin = get_origin(base_hint)
                args = get_args(base_hint)
        kwargs: Dict[str, Any] = {'help': f'Specify the {key}. Type: {base_hint.__name__}.', 'dest': key}
        processed = False
        if base_hint is str:
            kwargs['type'] = str
            processed = True
        elif base_hint is int:
            kwargs['type'] = int
            processed = True
        elif base_hint is float:
            kwargs['type'] = float
            processed = True
        elif base_hint is bool:
            kwargs['action'] = 'store_true'
            kwargs['help'] = f"Enable the '{key}' flag."
            processed = True

        elif origin is list or origin is List:
            if args and len(args) == 1 and args[0] in (str, int, float):
                inner_type = args[0]
                kwargs['type'] = inner_type
                kwargs['nargs'] = '*' # Use '*' for zero or more, '+' for one or more
                kwargs['help'] = f'Specify {key} (0 or more {inner_type.__name__} values).'
                processed = True

        elif origin is Literal:
            if args and all(isinstance(a, (str, int, float)) for a in args):
                choice_types = set(type(a) for a in args)
                if len(choice_types) == 1:
                     arg_type = choice_types.pop()
                     if arg_type in (str, int, float):
                          kwargs['choices'] = args
                          kwargs['type'] = arg_type
                          kwargs['help'] = f"Specify {key}. Choices: {', '.join(map(str, args))}."
                          processed = True
        if 'action' not in kwargs:
            if is_required:
                kwargs['required'] = True
            else:
                kwargs['default'] = None
                kwargs['required'] = False
        if processed:
             parser.add_argument(arg_name, **kwargs)
        elif key in hints:
            # print(f"Warning: Skipping argument '{key}'. Unsupported type hint: {hint}", file=sys.stderr)
            pass

    return parser


def load_config(log_config: str | os.PathLike) -> Tuple[Config, argparse.ArgumentParser]:
    if not os.path.exists(log_config):
        raise FileNotFoundError(f'Log config file "{log_config}" not found')

    with open(log_config, 'r') as f:
        config: Dict[str, Any] | None = yaml.safe_load(f.read())

    if config is None:
        raise ValueError('Config is empty')

    if not is_valid_dict(config, Config):
        raise ValueError('Invalid config')

    return config, create_parser_from_typeddict('', ArgsConfig)


def path_to_os_specific(path: str) -> str:
    return path.replace('/', os.sep).replace('\\', os.sep)


def load_locale(locale: str | os.PathLike) -> Dict:
    if not os.path.exists(locale):
        raise FileNotFoundError(f'Locale file "{locale}" not found')
    with open(locale, 'r', encoding='utf-8') as f:
        return json.load(f)