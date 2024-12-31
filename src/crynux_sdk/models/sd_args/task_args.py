from typing import Any, Mapping, Optional, TypedDict, Union, List
from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ..types import FloatFractionAsInt, NonEmptyString
from .controlnet_args import ControlnetArgs
from .scheduler_args import LCM, DPMSolverMultistep, EulerAncestralDiscrete


class RefinerArgs(BaseModel):
    model: NonEmptyString
    variant: Optional[str] = "fp16"
    denoising_cutoff: FloatFractionAsInt = 80  # Not used if controlnet is enabled
    steps: Annotated[int, Gt(0), Le(100)] = 20


class LoraArgs(BaseModel):
    model: NonEmptyString
    weight_file_name: str = ""
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
    cfg: Annotated[int, Ge(0), Le(20)] = 5


class BaseModelArgs(BaseModel):
    name: NonEmptyString
    variant: Optional[str] = "fp16"


class TaskArgs(BaseModel):
    # base model for image generation
    base_model: Union[BaseModelArgs, NonEmptyString]
    # custom unet model name
    unet: str = ""
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
    # custom scheduler args
    scheduler: Union[DPMSolverMultistep, EulerAncestralDiscrete, LCM] = Field(
        discriminator="method", default=DPMSolverMultistep()
    )
    # custom vae model name
    vae: str = ""
    # refiner config
    refiner: Optional[RefinerArgs] = None
    # textual inversion model name
    textual_inversion: str = ""

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.base_model, str):
            self.base_model = BaseModelArgs(name=self.base_model)

    def generate_model_ids(self) -> List[str]:
        res = []
        if isinstance(self.base_model, str):
            base_model = self.base_model
        else:
            base_model = self.base_model.name
            if self.base_model.variant is not None:
                base_model += f"+{self.base_model.variant}"
        res.append(f"base:{base_model}")

        if self.lora is not None:
            res.append(f"lora:{self.lora.model}")

        if self.controlnet is not None:
            controlnet = self.controlnet.model
            if self.controlnet.variant is not None:
                controlnet += f"+{self.controlnet.variant}"
            res.append(f"controlnet:{controlnet}")

        return res


class TaskOptionalArgs(TypedDict, total=False):
    # custom unet model name
    unet: str
    # task config for image generation
    task_config: Union[TaskConfig, Mapping[str, Any]]
    # lora config
    lora: Union[LoraArgs, Mapping[str, Any]]
    # controlnet config
    controlnet: Union[ControlnetArgs, Mapping[str, Any]]
    # custom scheduler args
    scheduler: Union[DPMSolverMultistep, EulerAncestralDiscrete, LCM, Mapping[str, Any]]
    # custom vae model name
    vae: str
    # refiner config
    refiner: Union[RefinerArgs, Mapping[str, Any]]
    # textual inversion model name
    textual_inversion: str
