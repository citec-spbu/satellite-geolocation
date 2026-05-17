import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from drone_localization.api.schemas.gallery import (
    GalleryUploadRequest,
    GalleryUploadResponse,
    GalleryImageResponse,
    GallerySearchRequest,
    GallerySearchResponse,
    GalleryCountResponse,
    GalleryDeleteResponse,
    GalleryClearResponse,
    GalleryHealthResponse,
)
from drone_localization.api.utils.utils import base64_to_pil, pil_to_base64
from drone_localization.core.services.gallery import GalleryService
from drone_localization.core.services.inference import InferenceService
from drone_localization.infrastructure.gallery_repository_impl import GalleryRepositoryImpl

logger = logging.getLogger(__name__)

router = APIRouter()

# Инициализация сервисов
inference_service = InferenceService()
gallery_repository = GalleryRepositoryImpl()
gallery_service = GalleryService(inference_service, gallery_repository)


@router.post("/upload", response_model=GalleryUploadResponse)
async def upload_image(
    image: str,  # base64 строка
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Загрузить спутниковое изображение в галерею.

    Изображение будет обработано моделью для получения эмбеддинга,
    затем сохранено в MinIO, а эмбеддинг - в Qdrant для поиска.
    """
    try:
        # Конвертируем base64 в PIL Image
        pil_image = base64_to_pil(image)

        # Загружаем через сервис
        image_id = gallery_service.upload_image(pil_image, metadata)

        return GalleryUploadResponse(
            image_id=image_id,
            status="success"
        )
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.post("/upload-file", response_model=GalleryUploadResponse)
async def upload_image_file(
    image: UploadFile = File(..., description="Файл изображения (JPEG/PNG)"),
    metadata: Optional[str] = Form(None, description="JSON метаданные")
):
    """
    Загрузить спутниковое изображение в галерею через file upload.

    Альтернативный эндпоинт для загрузки через multipart/form-data.
    """
    import json

    try:
        # Читаем файл
        contents = await image.read()

        # Конвертируем в PIL Image
        from PIL import Image as PILImage
        import io
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")

        # Парсим метаданные если есть
        parsed_metadata = None
        if metadata:
            parsed_metadata = json.loads(metadata)

        # Загружаем через сервис
        image_id = gallery_service.upload_image(pil_image, parsed_metadata)

        return GalleryUploadResponse(
            image_id=image_id,
            status="success"
        )
    except Exception as e:
        logger.error(f"Failed to upload image file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.get("/image/{image_id}", response_model=GalleryImageResponse)
async def get_image(image_id: str):
    """
    Получить изображение из галереи по ID.
    """
    try:
        pil_image = gallery_service.get_image(image_id)
        image_b64 = pil_to_base64(pil_image)

        # Получаем метаданные из repository (если нужно)
        # Для простоты возвращаем только image_id и image
        return GalleryImageResponse(
            image_id=image_id,
            image=image_b64
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    except Exception as e:
        logger.error(f"Failed to get image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get image: {str(e)}")


@router.post("/search", response_model=GallerySearchResponse)
async def search_similar(request: GallerySearchRequest):
    """
    Поиск похожих изображений в галерее.

    Принимает изображение, находит top_k наиболее похожих спутниковых снимков.
    """
    try:
        # Конвертируем base64 в PIL Image
        pil_image = base64_to_pil(request.image)

        # Получаем эмбеддинг
        embedding = inference_service.extract_satellite_embedding(pil_image)

        # Ищем похожие
        hits = gallery_service.search_similar(embedding, request.top_k)

        # Формируем ответ
        results = []
        for img_id, score in hits:
            try:
                img = gallery_service.get_image(img_id)
                img_b64 = pil_to_base64(img)
                results.append({
                    "image_id": img_id,
                    "image": img_b64,
                    "score": score
                })
            except Exception as e:
                logger.warning(f"Failed to retrieve image {img_id}: {e}")
                continue

        return GallerySearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/count", response_model=GalleryCountResponse)
async def get_count():
    """
    Получить количество изображений в галерее.
    """
    try:
        count = gallery_repository.count()
        return GalleryCountResponse(count=count)
    except Exception as e:
        logger.error(f"Failed to get count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")


@router.delete("/image/{image_id}", response_model=GalleryDeleteResponse)
async def delete_image(image_id: str):
    """
    Удалить изображение из галереи по ID.
    """
    try:
        deleted = gallery_repository.delete_image(image_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        return GalleryDeleteResponse(image_id=image_id, deleted=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")


@router.delete("/clear", response_model=GalleryClearResponse)
async def clear_gallery():
    """
    Полностью очистить галерею (удалить все изображения).

    Внимание: Это действие необратимо!
    """
    try:
        gallery_repository.clear()
        return GalleryClearResponse(status="success")
    except Exception as e:
        logger.error(f"Failed to clear gallery: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear gallery: {str(e)}")


@router.get("/health", response_model=GalleryHealthResponse)
async def health_check():
    """
    Проверка работоспособности хранилища галереи.
    """
    try:
        healthy = gallery_repository.health_check()
        return GalleryHealthResponse(
            healthy=healthy,
            details="OK" if healthy else "Storage unavailable"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return GalleryHealthResponse(
            healthy=False,
            details=str(e)
        )