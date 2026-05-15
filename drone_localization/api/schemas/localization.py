from pydantic import BaseModel


class LocalizationRequest(BaseModel):
    drone_image: str


class LocalizationResponse(BaseModel):
    # СТРОГО по схеме:
    drone_image: str
    satellite_image: str
    coordinates: dict
    confidence: float
