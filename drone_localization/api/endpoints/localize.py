import logging

from fastapi import APIRouter, HTTPException
from PIL import ImageDraw, Image
from drone_localization.api.schemas.localization import (
    LocalizationRequest,
    LocalizationResponse,
    Coordinates,
)
from drone_localization.api.utils.image import base64_to_pil, pil_to_base64
from drone_localization.core.services.refinement import RefinementService
from drone_localization.core.services.retrieval import RetrievalService
from drone_localization.api.utils.image import pixel_coords_to_jps

logger = logging.getLogger(__name__)
router = APIRouter()

retrieval_service = RetrievalService()
refinement_service = RefinementService()


from PIL import Image, ImageDraw

def draw_cross(image, x, y, size=10, color=(255, 0, 0), width=5):
    """
    Рисует крестик на PIL.Image по координатам (x, y),
    безопасно обрезая линии по границам изображения.
    """

    draw = ImageDraw.Draw(image)

    img_w, img_h = image.size
    left   = max(0, x - size)
    right  = min(img_w - 1, x + size)

    top    = max(0, y - size)
    bottom = min(img_h - 1, y + size)

    # Рисуем только если центр вообще попадает в изображение
    if 0 <= y < img_h:
        draw.line((left, y, right, y), fill=color, width=width)

    if 0 <= x < img_w:
        draw.line((x, top, x, bottom), fill=color, width=width)

    return image

@router.post("/localize", response_model=LocalizationResponse)
async def run_full_pipeline(request: LocalizationRequest):
    try:
        drone_pillow = base64_to_pil(request.drone_image)

        logger.info("Start 1st model")
        # 1. Поиск спутника (Model 1)
        satellite_result = retrieval_service.find_match(request.drone_image)

        logger.info("1st model's work is ended")
        '''logger.info("Treshhold check")
        # Проверка порога
        if satellite_result.score < 0.3:
            logger.info("Treshhold not passed")
            raise HTTPException(status_code=404, detail="Satellite image not found")'''

        logger.info("Start 2nd model")
        # 2. Уточнение позиции (Model 2) -> возвращает пиксельные координаты (x, y) на спутниковом изображении
        pixel_x, pixel_y = refinement_service.calculate_position(
            drone_image=drone_pillow,
            satellite_image=satellite_result.image,
        )
        logger.info("2nd model's work is ended")
        
        # Получаем метаданные спутникового снимка (tl_E, tl_N, br_E, br_N)
        if satellite_result.metadata is None:
            raise HTTPException(status_code=500, detail="Satellite metadata not found")

        logger.info("Satellit's metadata was found")
        # Определяем размеры спутникового изображения
        sat_width, sat_height = satellite_result.image.size

        logger.info("Satellite's width and height is known:", sat_width, sat_height)

        logger.info("2nd model's output is pixels now doing lat and lon")
        # Конвертируем пиксельные координаты в GPS (широта, долгота)
        latitude, longitude = pixel_coords_to_jps(
            satellite_meta=satellite_result.metadata,
            pixel_coords=(pixel_x, pixel_y),
            img_width=sat_width,
            img_height=sat_height
        )
        satellite_result.image=draw_cross(satellite_result.image,pixel_x,pixel_y)
        satellite_b64 = pil_to_base64(satellite_result.image)
        # 3. Сборка ответа СТРОГО по схеме jsons-talking.txt
        # coordinates - это широта и долгота (lat, lon) в виде прямоугольника
        # Для точечного результата используем одинаковые координаты для всех углов
        logger.info("Response")
        return LocalizationResponse(
            drone_image=request.drone_image,  # Как на схеме
            satellite_image=satellite_b64,  # Как на схеме
            coordinates=Coordinates(lat=latitude, lon=longitude),  # Точечные координаты (lat, lon)
            confidence=satellite_result.score,  # Как на схеме
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))