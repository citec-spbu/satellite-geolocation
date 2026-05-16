from pydantic import BaseModel, Field
from PIL.Image import Image as PILImage

class RetrievalResult(BaseModel):
    image: PILImage
    score: float = Field(..., ge=0.0, le=1.0)

    class Config:
        arbitrary_types_allowed = True