from typing import Literal, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    model_id: str
    dataset_id: str
    image_size: int = Field(224, ge=32, le=1024)
    normalization: Literal["imagenet", "zero_one"] = "imagenet"
    max_images: Optional[int] = Field(None, ge=2)   # evaluate a class-balanced subset to save time
    positive_class: Optional[str] = None            # binary datasets only; default is the second class
