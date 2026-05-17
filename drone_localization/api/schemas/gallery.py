from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    """Координаты в формате WGS84."""
    """Координаты прямоугольника в формате WGS84.

    tl - верхний левый угол (top-left)
    br - нижний правый угол (bottom-right)
    E - долгота (East)
    N - широта (North)
    """
    tl_E: float = Field(..., description="Долгота верхнего левого угла (longitude), диапазон [-180, 180]", ge=-180.0, le=180.0)
    tl_N: float = Field(..., description="Широта верхнего левого угла (latitude), диапазон [-90, 90]", ge=-90.0, le=90.0)
    br_E: float = Field(..., description="Долгота нижнего правого угла (longitude), диапазон [-180, 180]", ge=-180.0, le=180.0)
    br_N: float = Field(..., description="Широта нижнего правого угла (latitude), диапазон [-90, 90]", ge=-90.0, le=90.0)


class GalleryUploadRequest(BaseModel):
    """Запрос на загрузку спутникового изображения в галерею."""
    image: str = Field(..., description="Base64 кодированное изображение")
    coordinates: Optional[Coordinates] = Field(
        default=None,
        description="Координаты прямоугольника (tl_E, tl_N, br_E, br_N)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Дополнительные метаданные (filename, location и т.д.)"
    )


class GalleryUploadResponse(BaseModel):
    """Ответ после успешной загрузки изображения."""
    image_id: str = Field(..., description="Уникальный идентификатор загруженного изображения")
    status: str = Field(default="success", description="Статус операции")


class GalleryImageResponse(BaseModel):
    """Ответ с изображением из галереи."""
    image_id: str
    image: str = Field(..., description="Base64 кодированное изображение")
    coordinates: Optional[Coordinates] = None
    metadata: Optional[Dict[str, Any]] = None


class GallerySearchRequest(BaseModel):
    """Запрос на поиск похожих изображений."""
    image: str = Field(..., description="Base64 кодированное изображение для поиска похожих")
    top_k: int = Field(default=5, ge=1, le=100, description="Количество результатов")


class GallerySearchResult(BaseModel):
    """Результат поиска похожего изображения."""
    image_id: str
    image: str = Field(..., description="Base64 кодированное изображение")
    score: float = Field(..., ge=0.0, le=1.0, description="Оценка сходства")
    coordinates: Optional[Coordinates] = None
    metadata: Optional[Dict[str, Any]] = None


class GallerySearchResponse(BaseModel):
    """Ответ с результатами поиска."""
    results: list[GallerySearchResult]


class GalleryCountResponse(BaseModel):
    """Ответ с количеством изображений в галерее."""
    count: int


class GalleryDeleteResponse(BaseModel):
    """Ответ после удаления изображения."""
    image_id: str
    deleted: bool = Field(..., description="True если удаление успешно")


class GalleryClearResponse(BaseModel):
    """Ответ после очистки галереи."""
    status: str = Field(default="success", description="Статус операции")


class GalleryHealthResponse(BaseModel):
    """Ответ проверки здоровья галереи."""
    healthy: bool
    details: Optional[str] = None