from pydantic import BaseModel
from typing import Optional

class ProcessingSettings(BaseModel):
    threshold: int = 200
    padding_percent: int = 2
    dpi: int = 300
    aspect_lock: bool = True

class CropInfo(BaseModel):
    x: int
    y: int
    width: int
    height: int

class ProcessingResponse(BaseModel):
    cropped_image: bytes
    preview_image: bytes
    crop_info: CropInfo