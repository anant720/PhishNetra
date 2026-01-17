"""
Health check routes for RiskAnalyzer AI
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from ..dependencies import get_analysis_service
from ...core.logging import get_logger

router = APIRouter()
logger = get_logger("api.health")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "PhishNetra",
        "version": "1.0.0",
        "message": "Service is running and ready to analyze texts"
    }


@router.get("/health/detailed")
async def detailed_health_check(
    analysis_service: Any = Depends(get_analysis_service)
) -> Dict[str, Any]:
    """Detailed health check with component status"""

    try:
        # Check model availability
        models_status = {}
        all_healthy = True

        for model_name, model in analysis_service.models.items():
            try:
                if hasattr(model, 'get_model_info'):
                    info = model.get_model_info()
                    models_status[model_name] = {
                        "status": "healthy",
                        "info": info
                    }
                else:
                    models_status[model_name] = {
                        "status": "unknown",
                        "info": "No status method available"
                    }
            except Exception as e:
                models_status[model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                all_healthy = False

        # Check preprocessing
        try:
            test_result = analysis_service.preprocessor.preprocess_text("test")
            preprocessing_status = "healthy"
        except Exception as e:
            preprocessing_status = "unhealthy"
            all_healthy = False

        # Check explainability
        try:
            test_explanation = analysis_service.reasoning.generate_explanation(
                "test", {}, {"risk_score": 50, "threat_category": "test"}
            )
            explainability_status = "healthy"
        except Exception as e:
            explainability_status = "unhealthy"
            all_healthy = False

        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "PhishNetra",
            "version": "1.0.0",
            "components": {
                "models": models_status,
                "preprocessing": preprocessing_status,
                "explainability": explainability_status,
                "decision_fusion": "healthy"  # Would add detailed check in production
            },
            "overall_health": all_healthy
        }

    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "PhishNetra",
            "error": str(e)
        }