import logging
from typing import Optional, Dict, Any, List

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
    Coordinates,
    GallerySearchResult,
)
from drone_localization.api.utils.image import base64_to_pil, pil_to_base64
from drone_localization.core.services.gallery import GalleryService
from drone_localization.core.services.inference import InferenceService
from drone_localization.infrastructure.gallery_repository_impl import GalleryRepositoryImpl
from drone_localization.configs.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Инициализация сервисов
inference_service = InferenceService()
gallery_repository = GalleryRepositoryImpl()
gallery_service = GalleryService(inference_service, gallery_repository)


@router.post("/upload", response_model=GalleryUploadResponse)
async def upload_image(request: GalleryUploadRequest):
    """
    Загрузить спутниковое изображение в галерею.

    Изображение будет обработано моделью для получения эмбеддинга,
    затем сохранено в MinIO, а эмбеддинг - в Qdrant для поиска.

    Args:
        request: GalleryUploadRequest с изображением, координатами и метаданными
    """
    try:
        # Конвертируем base64 в PIL Image
        pil_image = base64_to_pil(request.image)

        # Собираем метаданные с координатами
        full_metadata = request.metadata or {}
        if request.coordinates:
            full_metadata["coordinates"] = {
                "tl_E": request.coordinates.tl_E,
                "tl_N": request.coordinates.tl_N,
                "br_E": request.coordinates.br_E,
                "br_N": request.coordinates.br_N
            }

        # Загружаем через сервис
        image_id = gallery_service.upload_image(pil_image, full_metadata)

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
    tl_E: Optional[float] = Form(None, description="Долгота верхнего левого угла (longitude)"),
    tl_N: Optional[float] = Form(None, description="Широта верхнего левого угла (latitude)"),
    br_E: Optional[float] = Form(None, description="Долгота нижнего правого угла (longitude)"),
    br_N: Optional[float] = Form(None, description="Широта нижнего правого угла (latitude)"),
    meta: Optional[str] = Form(None, description="JSON метаданные")
):
    """
    Загрузить спутниковое изображение в галерею через file upload.

    Альтернативный эндпоинт для загрузки через multipart/form-data.

    Args:
        image: Файл изображения
        lat: Широта места съемки
        lon: Долгота места съемки
        meta JSON метаданные
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
        parsed_metadata = {}
        if meta:
            parsed_metadata = json.loads(meta)

        # Добавляем координаты если указаны
        if all(v is not None for v in [tl_E, tl_N, br_E, br_N]):
            parsed_metadata["coordinates"] = {
                "tl_E": tl_E,
                "tl_N": tl_N,
                "br_E": br_E,
                "br_N": br_N
            }

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

    Возвращает изображение и связанные метаданные включая координаты.
    """
    try:
        pil_image = gallery_service.get_image(image_id)
        image_b64 = pil_to_base64(pil_image)

        # Получаем метаданные из Qdrant
        qdrant = gallery_repository.qdrant
        collection = gallery_repository.collection

        point = qdrant.retrieve(
            collection_name=collection,
            ids=[image_id],
            with_payload=True,
            with_vectors=False
        )

        coordinates = None
        metadata = {}

        if point and len(point) > 0:
            payload = point[0].payload
            coordinates_raw = payload.get("coordinates")
            if coordinates_raw:
                coordinates = Coordinates(
                    tl_E=coordinates_raw["tl_E"],
                    tl_N=coordinates_raw["tl_N"],
                    br_E=coordinates_raw["br_E"],
                    br_N=coordinates_raw["br_N"]
                )

            # Копируем остальные метаданные кроме image_id и coordinates
            for key, value in payload.items():
                if key not in ["image_id", "coordinates"]:
                    metadata[key] = value

        return GalleryImageResponse(
            image_id=image_id,
            image=image_b64,
            coordinates=coordinates,
            metadata=metadata if metadata else None
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
    Возвращает результаты с координатами и метаданными.
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

                # Получаем метаданные из Qdrant
                qdrant = gallery_repository.qdrant
                collection = gallery_repository.collection

                point = qdrant.retrieve(
                    collection_name=collection,
                    ids=[img_id],
                    with_payload=True,
                    with_vectors=False
                )

                coordinates = None
                metadata = {}

                if point and len(point) > 0:
                    payload = point[0].payload
                    coordinates_raw = payload.get("coordinates")
                    if coordinates_raw:
                        coordinates = Coordinates(
                            tl_E=coordinates_raw["tl_E"],
                            tl_N=coordinates_raw["tl_N"],
                            br_E=coordinates_raw["br_E"],
                            br_N=coordinates_raw["br_N"]
                        )

                    # Копируем остальные метаданные
                    for key, value in payload.items():
                        if key not in ["image_id", "coordinates"]:
                            metadata[key] = value

                results.append(GallerySearchResult(
                    image_id=img_id,
                    image=img_b64,
                    score=score,
                    coordinates=coordinates,
                    metadata=metadata if metadata else None
                ))
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