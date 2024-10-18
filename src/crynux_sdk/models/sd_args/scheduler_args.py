from typing import Literal, Optional
from pydantic import BaseModel


class CommonSchedulerArgs(BaseModel):
    num_train_timesteps: Optional[int] = None
    beta_start: Optional[float] = None
    beta_end: Optional[float] = None
    beta_schedule: Optional[str] = None
    prediction_type: Optional[str] = None
    timestep_spacing: Optional[str] = None
    steps_offset: Optional[int] = None
    rescale_betas_zero_snr: Optional[bool] = None


class EulerAncestralDiscrete(BaseModel):
    method: Literal['EulerAncestralDiscreteScheduler'] = 'EulerAncestralDiscreteScheduler'
    args: Optional[CommonSchedulerArgs] = None


class LCMArgs(CommonSchedulerArgs):
    original_inference_steps: Optional[int] = None
    clip_samples: Optional[int] = None
    clip_samples_range: Optional[float] = None
    set_alpha_to_one: Optional[bool] = None
    thresholding: Optional[bool] = None
    dynamic_thresholding_ratio: Optional[float] = None
    sample_max_value: Optional[float] = None
    timestep_scaling: Optional[float] = None


class LCM(BaseModel):
    method: Literal['LCMScheduler'] = 'LCMScheduler'
    args: Optional[LCMArgs] = None


class DPMSolverMultistepArgs(CommonSchedulerArgs):
    solver_order: Optional[int] = None
    thresholding: Optional[bool] = None
    dynamic_thresholding_ratio: Optional[float] = None
    sample_max_value: Optional[float] = None
    algorithm_type: Optional[str] = None
    solver_type: Optional[str] = None
    lower_order_final: Optional[bool] = None
    euler_at_final: Optional[bool] = None
    use_karras_sigmas: Optional[bool] = None
    use_lu_lambdas: Optional[bool] = None
    final_sigmas_type: Optional[str] = None
    lambda_min_clipped: Optional[float] = None
    variance_type: Optional[str] = None


class DPMSolverMultistep(BaseModel):
    method: Literal['DPMSolverMultistepScheduler'] = 'DPMSolverMultistepScheduler'
    args: Optional[DPMSolverMultistepArgs] = None
