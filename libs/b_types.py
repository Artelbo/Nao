from dataclasses import dataclass
from typing import TypeVar, get_type_hints, Annotated, Tuple
from enum import Enum

numberT = TypeVar('numberT', int, float)


@dataclass
class ValueRange:
    """
    Used in type hinting, represent a range of numbers.
    """
    min: numberT
    max: numberT

    def validate_value(self, name: str, x: numberT) -> None:
        """
        Validates if a given value falls within the defined range.

        Args:
            name: The name of the value being validated (used in error messages).
            x: The value to validate.

        Raises:
            ValueError: If the value `x` is not within the range [self.min, self.max].
        """
        if not (self.min <= x <= self.max):
            raise ValueError(f'{name} ({x}) must be in range [{self.min}, {self.max}]')


@dataclass
class MoreThan:
    """
    Used in type hinting, represent a range of numbers.
    """
    v: numberT

    def validate_value(self, name: str, x: numberT) -> None:
        """
        Validates if a given value falls within the defined range.

        Args:
            name: The name of the value being validated (used in error messages).
            x: The value to validate.

        Raises:
            ValueError: If the value `x` is less than self.v.
        """
        if not (x > self.v):
            raise ValueError(f'{name} ({x}) must be greater than {self.v}')


@dataclass
class LessThan:
    """
    Used in type hinting, represent a range of numbers.
    """
    v: numberT

    def validate_value(self, name: str, x: numberT) -> None:
        """
        Validates if a given value falls within the defined range.

        Args:
            name: The name of the value being validated (used in error messages).
            x: The value to validate.

        Raises:
            ValueError: If the value `x` is more than self.v.
        """
        if not (x < self.v):
            raise ValueError(f'{name} ({x}) must be less than {self.v}')


class ValidatedDataclass(type):
    """
    Metaclass for validating dataclass fields based on type hints and metadata.

    This metaclass automatically validates the fields of a dataclass instance
    against constraints specified in the type hints using `Annotated` and
    custom validator classes (e.g., `ValueRange`).

    .. code-block:: python
        @dataclass
        class MyData(metaclass=ValidatedDataclass):
            value: Annotated[int, ValueRange(0, 100)]

        # Valid instance creation
        data = MyData(value=50)

        # Invalid instance creation (raises ValueError)
        data = MyData(value=150)
    """
    def __call__(cls, *args, **kwargs):
        """
        Intercepts instance creation to perform validation.

        Args:
            *args: Positional arguments for the dataclass constructor.
            **kwargs: Keyword arguments for the dataclass constructor.

        Returns:
            The validated dataclass instance.

        Raises:
            ValueError: If any field value fails validation.
        """
        instance = super().__call__(*args, **kwargs)
        for field in fields(cls):  # type: ignore
            if field.name in instance.__dict__:
                value = instance.__dict__[field.name]
                hint = get_type_hints(cls, include_extras=True).get(field.name)
                validators = getattr(hint, '__metadata__', [])
                for validator in validators:
                    if hasattr(validator, 'validate_value'):
                        validator.validate_value(field.name, value)
        return instance


class Colors(Enum):
    """
    Predefined colors for LEDs.
    """
    OFF = 'off'
    ON = 'on'

    WHITE = 'white'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'
    MAGENTA = 'magenta'
    CYAN = 'cyan'

@dataclass
class Color(metaclass=ValidatedDataclass):
    """
    Represents a color in RGB format.
    """
    r: Annotated[int, ValueRange(0, 255)]
    g: Annotated[int, ValueRange(0, 255)]
    b: Annotated[int, ValueRange(0, 255)]

    @property
    def hex(self) -> str:
        """
        Returns the color in hexadecimal format.

        Returns:
            Color (str): The color in hexadecimal format.
        """
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'

    __str__ = hex


@dataclass
class Vector2:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def nao_compatible(self) -> Tuple[float, float, float]:
        return self.x, self.y, 0.0


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def nao_compatible(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z