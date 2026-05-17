import logging

from fastapi import APIRouter, HTTPException

from drone_localization.api.schemas.localization import (
    LocalizationRequest,
    LocalizationResponse,
)
from drone_localization.api.utils.utils import base64_to_pil, pil_to_base64
from drone_localization.core.services.refinement import RefinementService
from drone_localization.core.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter()

retrieval_service = RetrievalService()
refinement_service = RefinementService()


@router.post("/localize", response_model=LocalizationResponse)
async def run_full_pipeline(request: LocalizationRequest):
    try:
        drone_pillow = base64_to_pil(request.drone_image)

        logger.info("Start 1st model")
        # 1. Поиск спутника (Model 1)
        satellite_result = retrieval_service.find_match(request.drone_image)

        logger.info("1st model's work is ended")
        logger.info("Treshhold check")
        # Проверка порога
        if satellite_result.score < 0.3:
            logger.info("Treshhold not passed")
            raise HTTPException(status_code=404, detail="Satellite image not found")

        logger.info("Start 2nd model")
        # 2. Уточнение позиции (Model 2) -> возвращает только (lat, lon)
        lat, lon = refinement_service.calculate_position(
            drone_image=drone_pillow,
            satellite_image=satellite_result.image,
        )
        logger.info("2nd model's work is ended")

        satellite_b64 = pil_to_base64(satellite_result.image)
        # 3. Сборка ответа СТРОГО по схеме jsons-talking.txt
        return LocalizationResponse(
            drone_image=request.drone_image,  # Как на схеме
            satellite_image=satellite_b64,  # Как на схеме
            coordinates={"lat": lat, "lon": lon},  # Как на схеме
            confidence=satellite_result.score,  # Как на схеме
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
