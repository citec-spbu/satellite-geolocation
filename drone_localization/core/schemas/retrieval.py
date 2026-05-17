from PIL.Image import Image as PILImage
from typing import Optional, Dict, Any

class RetrievalResult(BaseModel):
    image: PILImage
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
