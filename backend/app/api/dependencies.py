"""
API dependencies for RiskAnalyzer AI
FastAPI dependency injection and validation
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..core.config import settings
from ..core.preprocessing import preprocessor
from ..core.url_analyzer_simple import simple_url_analyzer as url_analyzer
from ..models.fasttext_model import fasttext_model
from ..models.sentence_transformer_simple import sentence_transformer
from ..models.distilbert_simple import distilbert_classifier
from ..models.similarity_simple import similarity_engine
from ..models.decision_fusion import decision_fusion
from ..explainability.reasoning import reasoning_engine
from ..explainability.highlighting import text_highlighter
from ..core.logging import get_logger

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Logger
api_logger = get_logger("api")


class PhishNetraService:
    """Service for text analysis operations"""

    def __init__(self):
        self.logger = api_logger
        self.preprocessor = preprocessor
        self.logger.info("Initializing URL analyzer...")
        try:
            self.url_analyzer = url_analyzer
            self.logger.info("URL analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize URL analyzer: {e}")
            self.url_analyzer = None
        self.models = {
            'fasttext': fasttext_model,
            'sentence_transformer': sentence_transformer,
            'distilbert': distilbert_classifier,
            'similarity': similarity_engine
        }
        self.fusion = decision_fusion
        self.reasoning = reasoning_engine
        self.highlighter = text_highlighter

    async def analyze_text(self, text: str, include_explainability: bool = True) -> Dict[str, Any]:
        """
        Complete text analysis pipeline

        Args:
            text: Input text to analyze
            include_explainability: Whether to include detailed explanations

        Returns:
            Complete analysis result
        """

        self.logger.info("ENTERING analyze_text method")
        self.logger.info(f"Starting analysis for text: {text[:50]}...")

        # Debug: Check if URL analyzer is available
        self.logger.info(f"URL analyzer available: {self.url_analyzer is not None}")

        try:
            self.logger.info("Entering try block")
            # Step 1: Preprocess text
            processed_data = self.preprocessor.preprocess_text(text)
            processed_text = processed_data["processed_text"]

            self.logger.debug("Text preprocessing completed")

            # Step 2: Get model predictions
            model_outputs = {}

            # FastText embeddings - Calculate similarity against known scam patterns
            import numpy as np
            
            fasttext_embedding = self.models['fasttext'].get_sentence_vector(processed_text)
            text_lower = processed_text.lower()
            
            # Enhanced keyword detection for better category recognition
            scam_keywords = {
                # Financial manipulation keywords
                "urgent": 0.15, "immediate": 0.15, "transfer": 0.20, "money": 0.18,
                "send money": 0.25, "send funds": 0.25, "wire transfer": 0.30, "payment": 0.20,
                "account": 0.15, "bank": 0.18, "suspended": 0.12, "frozen": 0.15,
                "debit card": 0.20, "credit card": 0.20, "card number": 0.25,
                "balance": 0.12, "transaction": 0.15, "deposit": 0.18, "withdrawal": 0.18,
                "refund": 0.20, "fee": 0.12, "charge": 0.12, "amount": 0.10,
                "dollar": 0.08, "usd": 0.08, "rupees": 0.08, "inr": 0.08,

                # Phishing/credential keywords
                "verify": 0.18, "verification": 0.20, "click": 0.15, "login": 0.20,
                "password": 0.25, "username": 0.15, "credentials": 0.25,
                "security": 0.12, "alert": 0.12, "access": 0.15,

                # Authority impersonation
                "official": 0.18, "government": 0.20, "police": 0.20, "irs": 0.25,
                "fbi": 0.25, "court": 0.18, "legal": 0.15, "notice": 0.12,
                "department": 0.12, "agency": 0.12, "authority": 0.15,

                # Social engineering
                "help": 0.08, "support": 0.10, "assistance": 0.10, "emergency": 0.20,
                "problem": 0.08, "issue": 0.08, "concern": 0.08,

                # Investment/job scams
                "investment": 0.20, "profit": 0.18, "earn": 0.15, "income": 0.15,
                "job": 0.12, "work": 0.10, "opportunity": 0.12, "bonus": 0.15,
                "commission": 0.18, "salary": 0.12,

                # Lottery/reward
                "winner": 0.20, "lottery": 0.25, "prize": 0.20, "reward": 0.18,
                "claim": 0.15, "congratulations": 0.18,

                # Delivery/courier
                "delivery": 0.15, "package": 0.12, "shipment": 0.12, "courier": 0.15,
                "tracking": 0.10,

                # Tech support
                "virus": 0.18, "malware": 0.20, "hacked": 0.20, "compromised": 0.20,
                "device": 0.10, "computer": 0.08, "phone": 0.08,

                # Urgency/emotional manipulation
                "action": 0.12, "now": 0.08, "today": 0.10, "deadline": 0.15,
                "limited": 0.12, "time": 0.08, "quick": 0.08, "fast": 0.08,
                "family": 0.12, "relative": 0.12, "friend": 0.08
            }
            
            keyword_score = 0.0
            matched_keywords = []
            for keyword, weight in scam_keywords.items():
                if keyword in text_lower:
                    keyword_score += weight
                    matched_keywords.append(keyword)
            
            # Cap keyword score at 0.85
            keyword_score = min(keyword_score, 0.85)
            
            # Add embedding-based similarity with scam-like vectors
            scam_phrases = [
                "urgent transfer money account verify",
                "account suspended click verify",
                "official notice payment required",
                "security alert verify identity"
            ]
            
            max_embedding_sim = 0.0
            for phrase in scam_phrases:
                scam_vector = self.models['fasttext'].get_sentence_vector(phrase)
                embedding_sim = np.dot(fasttext_embedding, scam_vector) / (
                    np.linalg.norm(fasttext_embedding) * np.linalg.norm(scam_vector) + 1e-8
                )
                max_embedding_sim = max(max_embedding_sim, max(0, embedding_sim))
            
            # Combine keyword and embedding similarity
            fasttext_similarity = keyword_score * 0.7 + max_embedding_sim * 0.3
            fasttext_similarity = min(fasttext_similarity, 0.95)  # Cap at 0.95
            
            # Dynamic confidence based on similarity strength
            fasttext_confidence = 0.65 + (fasttext_similarity * 0.3)
            
            model_outputs['fasttext'] = {
                'embedding': fasttext_embedding.tolist() if isinstance(fasttext_embedding, np.ndarray) else fasttext_embedding,
                'similarity_score': float(fasttext_similarity),
                'confidence': float(fasttext_confidence),
                'patterns': matched_keywords[:5] if matched_keywords else []
            }

            # Sentence Transformer analysis
            st_result = self.models['sentence_transformer'].analyze_semantic_patterns(processed_text)
            model_outputs['sentence_transformer'] = st_result

            # DistilBERT classification
            bert_result = self.models['distilbert'].analyze_text_patterns(processed_text)
            # scam_probability, confidence_score, and label are now directly in bert_result
            model_outputs['distilbert'] = bert_result

            # Similarity search
            sim_result = self.models['similarity'].find_scam_variants(processed_text)
            model_outputs['similarity'] = sim_result

            # URL Analysis (if URLs are present) - Use original text, not processed text
            if self.url_analyzer is None:
                self.logger.error("URL analyzer is None!")
                urls = []
            else:
                try:
                    self.logger.info(f"About to extract URLs from original text: '{text[:100]}...'")
                    urls = self.url_analyzer.extract_urls(text)  # Use original text for URL detection
                    self.logger.info(f"URL extraction completed: found {len(urls)} URLs: {urls}")
                except Exception as e:
                    self.logger.error(f"URL extraction failed: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    urls = []

            if urls:
                self.logger.info(f"Found {len(urls)} URL(s) in text, analyzing...")
                url_analyses = []
                max_url_risk = 0.0
                for url in urls:
                    try:
                        url_result = await self.url_analyzer.analyze_url(url)
                        url_analyses.append(url_result)
                        max_url_risk = max(max_url_risk, url_result.get('risk_score', 0))
                    except Exception as e:
                        self.logger.warning(f"URL analysis failed for {url}: {e}")
                        # Still include URL with basic risk
                        url_analyses.append({
                            "url": url,
                            "risk_score": 50.0,  # Default suspicious
                            "verdict": "Suspicious",
                            "signals": ["analysis_failed"],
                            "details": {}
                        })
                
                # Use highest risk URL for model output
                model_outputs['url_analysis'] = {
                    "urls": url_analyses,
                    "url_count": len(urls),
                    "max_risk_score": max_url_risk,
                    "has_urls": True
                }
                
                # If URL risk is high, boost overall risk
                if max_url_risk > 60:
                    # Boost FastText similarity for URL-based scams
                    if 'fasttext' in model_outputs:
                        model_outputs['fasttext']['similarity_score'] = min(
                            model_outputs['fasttext']['similarity_score'] + 0.2, 0.95
                        )
            else:
                model_outputs['url_analysis'] = {
                    "urls": [],
                    "url_count": 0,
                    "max_risk_score": 0.0,
                    "has_urls": False
                }

            self.logger.debug("Model predictions completed")

            # Step 3: Decision fusion
            fused_result = self.fusion.fuse_predictions(model_outputs, text)

            self.logger.debug("Decision fusion completed")

            # Step 4: Explainability (if requested)
            if include_explainability:
                explanation = self.reasoning.generate_explanation(
                    text, model_outputs, fused_result
                )
                fused_result["explanation"] = explanation

                # Add highlighted text
                highlights = explanation.get("highlighted_phrases", [])
                fused_result["highlighted_text_html"] = self.highlighter.highlight_text(
                    text, highlights, format_type="html"
                )

            self.logger.info(f"Analysis completed with risk score: {fused_result.get('risk_score', 0):.1f}")

            # Add URL analysis to result if present
            if 'url_analysis' in model_outputs:
                fused_result["url_analysis"] = model_outputs['url_analysis']

            # Add metadata
            fused_result["metadata"] = {
                "text_length": len(text),
                "processed_text_length": len(processed_text),
                "models_used": list(self.models.keys()),
                "processing_time": "calculated_server_side"  # Would be measured in production
            }

            return fused_result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Global analysis service instance
try:
    api_logger.info("Creating PhishNetraService instance...")
    analysis_service = PhishNetraService()
    api_logger.info("PhishNetraService created successfully")
except Exception as e:
    api_logger.error(f"Failed to create PhishNetraService: {e}")
    import traceback
    api_logger.error(f"Traceback: {traceback.format_exc()}")
    analysis_service = None


async def get_analysis_service() -> PhishNetraService:
    """Dependency to get analysis service"""
    api_logger.info(f"get_analysis_service called, analysis_service is: {analysis_service}")
    if analysis_service is None:
        api_logger.error("analysis_service is None!")
        raise HTTPException(status_code=500, detail="Analysis service not available")
    return analysis_service


def validate_text_length(text: str) -> str:
    """Validate and sanitize input text"""
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Text input is required")

    text = text.strip()

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > 10000:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")

    return text


class RequestValidator:
    """Request validation utilities"""

    @staticmethod
    def validate_analysis_request(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate analysis request parameters"""

        # Validate text
        validated_text = validate_text_length(text)

        # Validate options
        if options is None:
            options = {}

        validated_options = {
            "include_explainability": options.get("include_explainability", True),
            "format_type": options.get("format_type", "full"),
            "language": options.get("language", "auto")
        }

        # Validate format_type
        allowed_formats = ["full", "basic", "minimal"]
        if validated_options["format_type"] not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format_type. Allowed: {allowed_formats}"
            )

        return {
            "text": validated_text,
            "options": validated_options
        }


# Request validation dependency
async def validate_request(
    request: Request,
    text: str,
    include_explainability: Optional[bool] = True,
    format_type: Optional[str] = "full"
) -> Dict[str, Any]:
    """Validate incoming request"""

    # Log request
    api_logger.info(f"Processing request from {get_remote_address(request)}")

    # Validate request
    validator = RequestValidator()
    return validator.validate_analysis_request(
        text,
        {
            "include_explainability": include_explainability,
            "format_type": format_type
        }
    )