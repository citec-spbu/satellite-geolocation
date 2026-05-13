from fastapi import APIRouter, HTTPException
from ..schemas.localization import LocalizationRequest, LocalizationResponse
from core.services.retrieval import RetrievalService
from core.services.refinement import RefinementService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

retrieval_service = RetrievalService()
refinement_service = RefinementService()

@router.post("/localize", response_model=LocalizationResponse)
async def run_full_pipeline(request: LocalizationRequest):
    try:
        logger.info("Start 1st model")
        # 1. Поиск спутника (Model 1)
        satellite_result = retrieval_service.find_match(request.drone_image)
        
        logger.info("1st model's work is ended")
        logger.info("Treshhold check")
        # Проверка порога
        if satellite_result.score < 0.3:
            raise HTTPException(status_code=404, detail="Satellite image not found")
        
        logger.info("Start 2nd model")
        # 2. Уточнение позиции (Model 2) -> возвращает только (lat, lon)
        lat, lon = refinement_service.calculate_position(
            drone_image_b64=request.drone_image,
            satellite_image_b64=satellite_result.image
        )
        logger.info("2nd model's work is ended")

        # 3. Сборка ответа СТРОГО по схеме jsons-talking.txt
        return LocalizationResponse(
            drone_image=request.drone_image,           # Как на схеме
            satellite_image=satellite_result.image,    # Как на схеме
            coordinates={"lat": lat, "lon": lon},      # Как на схеме
            confidence=satellite_result.score          # Как на схеме
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))