from pydantic import BaseModel

class LocalizationRequest(BaseModel):
    drone_image: str

class LocalizationResponse(BaseModel):
    drone_image: str
    sattelite_image: str
    coordinates: dict
    confidence: float