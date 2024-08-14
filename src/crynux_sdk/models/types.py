from enum import Enum

from annotated_types import MinLen
from pydantic import Field, WithJsonSchema
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated


def fraction_int_to_float(fraction: int) -> float:

    if isinstance(fraction, int) or fraction > 1:
        return float(fraction) / 100.0
    else:
        return fraction


def float_to_fraction_int(fraction: float) -> int:
    return int(fraction * 100)


FloatFractionAsInt = Annotated[
    float,
    Field(validate_default=True, gt=0, le=100),
    AfterValidator(fraction_int_to_float),
    PlainSerializer(float_to_fraction_int, return_type=int, when_used="json"),
    WithJsonSchema(
        {"type": "integer", "exclusiveMinimum": 0, "maximum": 100}, mode="validation"
    ),
    WithJsonSchema(
        {"type": "integer", "exclusiveMinimum": 0, "maximum": 100}, mode="serialization"
    ),
]

NonEmptyString = Annotated[str, MinLen(1)]


class VersionString(str, Enum):
    V1_0_0 = "1.0.0"
    V2_0_0 = "2.0.0"
    V2_1_0 = "2.1.0"
