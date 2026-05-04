from fastapi import APIRouter, HTTPException
from api.schemas.localization import LocalizationRequest, LocalizationResponse
from core.services.retrieval import RetrievalService
from core.services.refinement import RefinementService

router = APIRouter()
retrieval_service = RetrievalService()
refinement_service = RefinementService()

@router.post("/localize", response_model=LocalizationResponse)
async def run_full_pipeline(request: LocalizationRequest):
    try:
        # 1. Поиск спутника
        satellite_result = retrieval_service.find_match(request.drone_image)
        
        # Проверка на валидность результата (по порогу похожести)
        if satellite_result.score < 0.3:
            raise HTTPException(status_code=404, detail="Satellite image not found or too dissimilar")
        
        # 2. Уточнение позиции
        lat, lon = refinement_service.calculate_position(
            drone_image_b64=request.drone_image,
            satellite_image_b64=satellite_result.image
        )

        # 3. Сборка ответа
        return LocalizationResponse(
            drone_image=request.drone_image,
            satellite_image=satellite_result.image, 
            coordinates={"lat": lat, "lon": lon},  # приводим к dict
            confidence=satellite_result.score
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))