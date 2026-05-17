from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class GalleryUploadRequest(BaseModel):
    """Запрос на загрузку спутникового изображения в галерею."""
    image: str = Field(..., description="Base64 кодированное изображение")
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