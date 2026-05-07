from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class Coordinates(BaseModel):
    """GPS координаты с валидацией"""
    lat: float = Field(..., ge=-90, le=90, description="Широта")
    lon: float = Field(..., ge=-180, le=180, description="Долгота")
    
    @field_validator('lat', 'lon')
    @classmethod
    def round_coordinates(cls, v):
        """Округляем до 6 знаков (~10 см точность)"""
        return round(v, 6)

class RetrievalCandidate(BaseModel):
    """Кандидат из поиска (top-K)"""
    image: str = Field(..., description="Base64 спутникового снимка")
    score: float = Field(..., ge=0, le=1, description="Похожесть")
    tile_id: Optional[str] = Field(None, description="ID тайла")

class LocalizationRequest(BaseModel):
    """Запрос на локализацию"""
    drone_image: str = Field(..., min_length=50, description="Base64 drone image")
    use_refinement: bool = Field(default=True, description="Использовать Model 2")
    top_k: int = Field(default=5, ge=1, le=20, description="Количество кандидатов")
    
    @field_validator('drone_image')
    @classmethod
    def validate_base64(cls, v):
        """Убираем data:image/png;base64, префикс если есть"""
        if ',' in v:
            v = v.split(',')[1]
        if len(v) < 50:
            raise ValueError('Image too small or invalid')
        return v

class LocalizationResponse(BaseModel):
    """Ответ API"""
    satellite_image: str = Field(..., description="Base64 найденного спутника")
    coordinates: Coordinates = Field(..., description="GPS координаты")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность модели")
    processing_time_ms: Optional[float] = Field(None, description="Время обработки (мс)")
    candidates: Optional[List[RetrievalCandidate]] = Field(None, description="Top-K кандидатов")
    
    class Config:
        json_schema_extra = {
            "example": {
                "satellite_image": "iVBORw0KGgoAAAANSUhEUgAA...",
                "coordinates": {"lat": 59.9343, "lon": 30.3351},
                "confidence": 0.94,
                "processing_time_ms": 245.3
            }
        }