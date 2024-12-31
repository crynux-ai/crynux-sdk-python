from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from ..types import FloatFractionAsInt, NonEmptyString


class CannyArgs(BaseModel):
    low_threshold: int = 100
    high_threshold: int = 200


class PreprocessMethodCanny(BaseModel):
    method: Literal["canny"] = "canny"
    args: Optional[CannyArgs] = None


class PreprocessMethodScribbleHED(BaseModel):
    method: Literal["scribble_hed"] = "scribble_hed"


class PreprocessMethodSoftEdgeHED(BaseModel):
    method: Literal["softedge_hed"] = "softedge_hed"


class PreprocessMethodScribbleHEDSafe(BaseModel):
    method: Literal["scribble_hedsafe"] = "scribble_hedsafe"


class PreprocessMethodSoftEdgeHEDSafe(BaseModel):
    method: Literal["softedge_hedsafe"] = "softedge_hedsafe"


class PreprocessMethodDepthMidas(BaseModel):
    method: Literal["depth_midas"] = "depth_midas"


class MLSDArgs(BaseModel):
    thr_v: FloatFractionAsInt = 10
    thr_d: FloatFractionAsInt = 10


class PreprocessMethodMLSD(BaseModel):
    method: Literal["mlsd"] = "mlsd"
    args: Optional[MLSDArgs] = None


class PreprocessMethodOpenPoseBodyOnly(BaseModel):
    method: Literal["openpose"] = "openpose"


class PreprocessMethodOpenPoseFaceAndBody(BaseModel):
    method: Literal["openpose_face"] = "openpose_face"


class PreprocessMethodOpenPoseFaceOnly(BaseModel):
    method: Literal["openpose_faceonly"] = "openpose_faceonly"


class PreprocessMethodOpenPoseFull(BaseModel):
    method: Literal["openpose_full"] = "openpose_full"


class PreprocessMethodOpenPoseHand(BaseModel):
    method: Literal["openpose_hand"] = "openpose_hand"


class PreprocessMethodDWPose(BaseModel):
    method: Literal["dwpose"] = "dwpose"


class PidiNetArgs(BaseModel):
    apply_filter: bool = False


class PreprocessMethodScribblePidiNet(BaseModel):
    method: Literal["scribble_pidinet"] = "scribble_pidinet"
    args: Optional[PidiNetArgs ] = None


class PreprocessMethodSoftEdgePidiNet(BaseModel):
    method: Literal["softedge_pidinet"] = "softedge_pidinet"
    args: Optional[PidiNetArgs ] = None


class PreprocessMethodScribblePidiNetSafe(BaseModel):
    method: Literal["scribble_pidisafe"] = "scribble_pidisafe"
    args: Optional[PidiNetArgs ] = None


class PreprocessMethodSoftEdgePidiNetSafe(BaseModel):
    method: Literal["softedge_pidisafe"] = "softedge_pidisafe"
    args: Optional[PidiNetArgs ] = None


class PreprocessMethodNormalBAE(BaseModel):
    method: Literal["normal_bae"] = "normal_bae"


class PreprocessMethodLineartCoarse(BaseModel):
    method: Literal["lineart_coarse"] = "lineart_coarse"


class PreprocessMethodLineartRealistic(BaseModel):
    method: Literal["lineart_realistic"] = "lineart_realistic"


class PreprocessMethodLineartAnime(BaseModel):
    method: Literal["lineart_anime"] = "lineart_anime"


class DepthZoeArgs(BaseModel):
    gamma_corrected: bool = False


class PreprocessMethodDepthZoe(BaseModel):
    method: Literal["depth_zoe"] = "depth_zoe"
    args: Optional[DepthZoeArgs] = None


class LeresArgs(BaseModel):
    thr_a: int = 0
    thr_b: int = 0


class PreprocessMethodDepthLeres(BaseModel):
    method: Literal["depth_leres"] = "depth_leres"
    args: Optional[LeresArgs] = None


class PreprocessMethodDepthLeresPP(BaseModel):
    method: Literal["depth_leres++"] = "depth_leres++"
    args: Optional[LeresArgs] = None


class ShuffleArgs(BaseModel):
    h: Optional[int] = None
    w: Optional[int] = None
    f: Optional[int] = None


class PreprocessMethodShuffle(BaseModel):
    method: Literal["shuffle"] = "shuffle"
    args: Optional[ShuffleArgs] = None


class MediapipeFaceArgs(BaseModel):
    max_faces: int = 1
    min_confidence: FloatFractionAsInt = 50


class PreprocessMethodMediapipeFace(BaseModel):
    method: Literal["mediapipe_face"] = "mediapipe_face"
    args: Optional[MediapipeFaceArgs] = None


class ControlnetArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: NonEmptyString
    variant: Optional[str] = "fp16"
    image_dataurl: str = ""
    weight: FloatFractionAsInt = 70
    preprocess: Union[
        PreprocessMethodCanny,
        PreprocessMethodScribbleHED,
        PreprocessMethodScribbleHEDSafe,
        PreprocessMethodSoftEdgeHEDSafe,
        PreprocessMethodDepthMidas,
        PreprocessMethodMLSD,
        PreprocessMethodOpenPoseBodyOnly,
        PreprocessMethodOpenPoseFaceAndBody,
        PreprocessMethodOpenPoseFaceOnly,
        PreprocessMethodOpenPoseFull,
        PreprocessMethodOpenPoseHand,
        PreprocessMethodScribblePidiNet,
        PreprocessMethodSoftEdgePidiNet,
        PreprocessMethodScribblePidiNetSafe,
        PreprocessMethodSoftEdgePidiNetSafe,
        PreprocessMethodNormalBAE,
        PreprocessMethodLineartCoarse,
        PreprocessMethodLineartRealistic,
        PreprocessMethodLineartAnime,
        PreprocessMethodDepthZoe,
        PreprocessMethodDepthLeres,
        PreprocessMethodDepthLeresPP,
        PreprocessMethodShuffle,
        PreprocessMethodMediapipeFace,
        None,
    ] = Field(discriminator="method", default=None)
