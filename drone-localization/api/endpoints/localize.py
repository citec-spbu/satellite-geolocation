from fastapi import APIRouter, HTTPException, status
from api.schemas.localization import (
    LocalizationRequest, 
    LocalizationResponse, 
    Coordinates,
    RetrievalCandidate
)
from core.services.retrieval import RetrievalService
from core.services.refinement import RefinementService
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Инициализация сервисов (глобально, чтобы не загружать каждый запрос)
retrieval_service = RetrievalService()
refinement_service = RefinementService()

@router.post("/localize", response_model=LocalizationResponse)
async def run_full_pipeline(request: LocalizationRequest):
    """
    Полный пайплайн локализации дрона
    
    1. Находит ближайший спутниковый тайл (Model 1)
    2. Уточняет координаты (Model 2)
    3. Возвращает результат
    """
    start_time = time.time()
    logger.info(f"Received localization request")
    
    try:
        # 1. Поиск спутника (Model 1)
        satellite_result = retrieval_service.find_match(
            drone_image_b64=request.drone_image,
            top_k=request.top_k
        )
        
        # Проверка порога похожести
        if satellite_result.score < 0.3:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Satellite image not found or too dissimilar"
            )
        
        # 2. Уточнение позиции (Model 2)
        lat, lon, confidence = refinement_service.calculate_position(
            drone_image_b64=request.drone_image,
            satellite_image_b64=satellite_result.image
        )
        
        # Считаем время обработки
        processing_time = (time.time() - start_time) * 1000
        
        # 3. Сборка ответа
        return LocalizationResponse(
            satellite_image=satellite_result.image,
            coordinates=Coordinates(lat=lat, lon=lon),
            confidence=confidence,
            processing_time_ms=round(processing_time, 2),
            candidates=None  # Можно добавить top-K если нужно
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )