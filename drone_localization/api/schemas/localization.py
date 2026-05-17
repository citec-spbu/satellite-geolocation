from pydantic import BaseModel, Field

class Coordinates(BaseModel):
    """Координаты в формате WGS84."""
    lat: float = Field(..., description="Широта (latitude) в градусах, диапазон [-90, 90]", ge=-90.0, le=90.0)
    lon: float = Field(..., description="Долгота (longitude) в градусах, диапазон [-180, 180]", ge=-180.0, le=180.0)

class LocalizationRequest(BaseModel):
    drone_image: str

class LocalizationResponse(BaseModel):
    # СТРОГО по схеме:
    drone_image: str
    satellite_image: str
    coordinates: Coordinates  # Явно указываем что это {lat, lon}
    confidence: float