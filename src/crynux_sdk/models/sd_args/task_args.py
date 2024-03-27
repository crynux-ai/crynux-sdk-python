from __future__ import annotations

from typing import Any, Mapping, Optional, TypedDict, Union

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel
from typing_extensions import Annotated

from ..types import FloatFractionAsInt, NonEmptyString
from .controlnet_args import ControlnetArgs


class RefinerArgs(BaseModel):
    model: NonEmptyString
    denoising_cutoff: FloatFractionAsInt = 80  # Not used if controlnet is enabled
    steps: Annotated[int, Gt(0), Le(100)] = 20


class LoraArgs(BaseModel):
    model: NonEmptyString
    weight: FloatFractionAsInt = 100


class TaskConfig(BaseModel):
    # image width
    image_width: int = 512
    # image height
    image_height: int = 512
    # stable diffusion image generation steps, default to 25
    steps: Annotated[int, Gt(0), Le(100)] = 25
    # image generation seed
    seed: Annotated[int, Ge(0), Lt(2147483648)] = 0
    # number of images to generate
    num_images: Annotated[int, Gt(0), Le(10)] = 1
    # whether to enable safety checker
    safety_checker: bool = True
    # cfg of stable diffusion
    cfg: Annotated[int, Gt(0), Le(20)] = 5


class TaskArgs(BaseModel):
    # base model for image generation
    base_model: NonEmptyString
    # prompt for image generation
    prompt: NonEmptyString
    # negative prompt for image generation
    negative_prompt: str = ""
    # task config for image generation
    task_config: TaskConfig
    # lora config
    lora: Optional[LoraArgs] = None
    # controlnet config
    controlnet: Optional[ControlnetArgs] = None
    # custom vae model name
    vae: str = ""
    # refiner config
    refiner: Optional[RefinerArgs] = None
    # textual inversion model name
    textual_inversion: str = ""


class TaskOptionalArgs(TypedDict, total=False):
    # task config for image generation
    task_config: Union[TaskConfig, Mapping[str, Any]]
    # lora config
    lora: Union[LoraArgs, Mapping[str, Any]]
    # controlnet config
    controlnet: Union[ControlnetArgs, Mapping[str, Any]]
    # custom vae model name
    vae: str
    # refiner config
    refiner: Union[RefinerArgs, Mapping[str, Any]]
    # textual inversion model name
    textual_inversion: str
