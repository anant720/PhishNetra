"""
Analysis API routes for RiskAnalyzer AI
REST endpoints for text analysis
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ..dependencies import (
    get_analysis_service
)
from ...core.logging import get_logger
from ...core.config import settings

# Create router
router = APIRouter()
logger = get_logger("api.analyze")


class AnalysisRequest(BaseModel):
    """Request model for text analysis"""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze for scam detection")
    include_explainability: Optional[bool] = Field(True, description="Include detailed explanations")
    format_type: Optional[str] = Field("full", description="Response format: full, basic, minimal")

    class Config:
        schema_extra = {
            "example": {
                "text": "URGENT: Your account has been suspended. Click here to verify: http://fakebank.com/verify",
                "include_explainability": True,
                "format_type": "full"
            }
        }


class ThreatCategoryItem(BaseModel):
    """Multi-label threat category with confidence"""
    category: str = Field(..., description="Threat category name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this category")


class URLAnalysisResult(BaseModel):
    """URL analysis result"""
    url: str = Field(..., description="Analyzed URL")
    risk_score: float = Field(..., ge=0, le=100, description="URL risk score")
    verdict: str = Field(..., description="URL authenticity verdict (Likely Legitimate / Suspicious / High Risk)")
    signals: List[str] = Field(default_factory=list, description="Detected risk signals")
    reasoning: List[str] = Field(default_factory=list, description="URL analysis reasoning")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed analysis breakdown")
    contribution_to_scam_score: Optional[float] = Field(None, description="Contribution to final scam score")


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""

    risk_score: float = Field(..., ge=0, le=100, description="Risk score from 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    threat_category: str = Field(..., description="Primary detected threat category")
    threat_categories: Optional[List[ThreatCategoryItem]] = Field(
        None, description="All applicable threat categories (multi-label)"
    )
    reasoning: str = Field(..., description="Human-readable reasoning")
    model_confidence_breakdown: Optional[Dict[str, float]] = Field(
        None, description="Confidence scores for each model"
    )
    url_analysis: Optional[Dict[str, Any]] = Field(
        None, description="URL analysis results if URLs were detected. Includes url_count, max_risk_score, has_urls, and detailed analysis for each URL"
    )

    class Config:
        schema_extra = {
            "example": {
                "risk_score": 85.5,
                "confidence": 0.82,
                "threat_category": "phishing_redirection",
                "threat_categories": [
                    {"category": "phishing_redirection", "confidence": 1.0},
                    {"category": "identity_verification_abuse", "confidence": 0.75},
                    {"category": "financial_manipulation", "confidence": 0.45}
                ],
                "reasoning": "High-risk phishing attempt detected with suspicious URL and urgent language",
                "model_confidence_breakdown": {
                    "fasttext": 0.75,
                    "sentence_transformer": 0.80,
                    "distilbert": 0.85,
                    "similarity": 0.70
                },
                "url_analysis": {
                    "url_count": 1,
                    "max_risk_score": 75.0,
                    "has_urls": True
                }
            }
        }


@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Test endpoint working"}

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    phishnetra_service: Any = Depends(get_analysis_service)
):
    """
    Analyze text for scam detection

    This endpoint performs comprehensive scam analysis using multiple AI models
    and provides detailed risk assessment with explainability features.
    """

    logger.info("=== ANALYZE ROUTE CALLED ===")
    try:
        logger.info("Processing analysis request")
        logger.info(f"PhishNetra service available: {phishnetra_service is not None}")

        # Get validated data from Pydantic model
        text = request.text.strip()
        logger.info(f"Text to analyze: '{text[:100]}...'")
        options = {
            "include_explainability": request.include_explainability,
            "format_type": request.format_type
        }

        # Validate text is not empty
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Perform analysis
        logger.info(f"About to call analyze_text service")
        result = await phishnetra_service.analyze_text(
            text=text,
            include_explainability=options["include_explainability"]
        )
        logger.info(f"Service call completed, result has keys: {list(result.keys())}")

        # FORCE: Add basic URL detection regardless
        import re
        basic_url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)
        basic_urls = basic_url_pattern.findall(text)
        logger.info(f"Basic URL detection found: {basic_urls}")

        if basic_urls:
            result['url_analysis'] = {
                "urls": [{"url": url, "risk_score": 50.0, "verdict": "Suspicious", "signals": ["basic_detection"]} for url in basic_urls],
                "url_count": len(basic_urls),
                "max_risk_score": 50.0,
                "has_urls": True
            }
            logger.info("Added URL analysis to result")
        else:
            result['url_analysis'] = {
                "urls": [],
                "url_count": 0,
                "max_risk_score": 0.0,
                "has_urls": False
            }
            logger.info("No URLs found, added empty URL analysis")

        # Format response based on requested format
        format_type = options["format_type"]

        if format_type == "basic":
            response = {
                "risk_score": result["risk_score"],
                "confidence": result["confidence"],
                "threat_category": result["threat_category"],
                "reasoning": result["reasoning"]
            }
        elif format_type == "minimal":
            response = {
                "risk_score": result["risk_score"],
                "threat_category": result["threat_category"]
            }
        else:  # full
            response = result

        # Add request ID for tracking (would be implemented in production)
        response["request_id"] = "generated_request_id"

        logger.info(f"Analysis completed: risk_score={result['risk_score']:.1f}")

        return response

    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch")
async def analyze_batch(
    texts: list[str],
    background_tasks: BackgroundTasks,
    include_explainability: Optional[bool] = True,
    phishnetra_service: Any = Depends(get_analysis_service)
):
    """
    Analyze multiple texts for scam detection

    Accepts a list of texts and returns analysis for each.
    """

    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    if len(texts) > 50:  # Reasonable batch limit
        raise HTTPException(status_code=400, detail="Too many texts (max 50)")

    try:
        logger.info(f"Processing batch analysis request with {len(texts)} texts")

        results = []
        for i, text in enumerate(texts):
            try:
                # Validate each text
                if not isinstance(text, str) or len(text.strip()) == 0:
                    results.append({
                        "index": i,
                        "error": "Invalid text input",
                        "risk_score": 0,
                        "threat_category": "error"
                    })
                    continue

                # Analyze text
                result = await phishnetra_service.analyze_text(
                    text=text.strip(),
                    include_explainability=include_explainability
                )

                result["index"] = i
                results.append(result)

            except Exception as e:
                logger.error(f"Error analyzing text {i}: {e}")
                results.append({
                    "index": i,
                    "error": str(e),
                    "risk_score": 0,
                    "threat_category": "error"
                })

        logger.info(f"Batch analysis completed: {len(results)} results")

        return {
            "results": results,
            "total_processed": len([r for r in results if "error" not in r]),
            "total_errors": len([r for r in results if "error" in r])
        }

    except Exception as e:
        logger.error(f"Batch analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RiskAnalyzer AI",
        "version": "1.0.0"
    }


@router.get("/models/info")
async def get_models_info(phishnetra_service: Any = Depends(get_analysis_service)):
    """Get information about loaded models"""

    try:
        models_info = {}

        for model_name, model in phishnetra_service.models.items():
            if hasattr(model, 'get_model_info'):
                models_info[model_name] = model.get_model_info()

        # Add fusion info
        models_info["decision_fusion"] = phishnetra_service.fusion.get_fusion_info()

        return {
            "models": models_info,
            "total_models": len(models_info)
        }

    except Exception as e:
        logger.error(f"Models info endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")