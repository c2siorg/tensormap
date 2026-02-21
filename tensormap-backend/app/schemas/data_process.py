import uuid as uuid_pkg

from pydantic import BaseModel


class TargetAddRequest(BaseModel):
    target: str
    file_id: uuid_pkg.UUID


class TransformationItem(BaseModel):
    transformation: str
    feature: str


class PreprocessRequest(BaseModel):
    transformations: list[TransformationItem]


class ImagePropertiesRequest(BaseModel):
    fileId: uuid_pkg.UUID
    fileType: str
    image_size: int
    batch_size: int
    color_mode: str
    label_mode: str
