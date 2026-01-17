"""
Simple DistilBERT classifier for PhishNetra
Works without transformers/torch dependencies
"""

import re
from typing import List, Dict, Any, Optional

from ..core.logging import LoggerMixin


class SimpleDistilBERTClassifier(LoggerMixin):
    """
    Simple classifier without ML dependencies
    Uses rule-based and pattern matching for scam detection
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or "./models/simple_classifier"
        self.device = device or "cpu"
        self.max_seq_length = 512

        self.logger.info(f"Initializing Simple DistilBERT Classifier")

        # Scam detection patterns
        self.scam_patterns = {
            "phishing_urls": re.compile(r'https?://[^\s]+\.(xyz|top|win|bid|loan|club|online|site|click|link)', re.IGNORECASE),
            "urgent_keywords": re.compile(r'\b(urgent|immediate|action required|act now|time sensitive|limited time)\b', re.IGNORECASE),
            "authority_claims": re.compile(r'\b(irs|fbi|police|government|official|court|legal|department)\b', re.IGNORECASE),
            "financial_terms": re.compile(r'\b(transfer|send money|payment|account|bank|credit|debit|loan|investment)\b', re.IGNORECASE),
            "social_engineering": re.compile(r'\b(help|emergency|family|friend|trouble|support|assist|problem)\b', re.IGNORECASE),
            "technical_deception": re.compile(r'\b(verify|confirm|security|alert|click|link|download|attachment|login|password)\b', re.IGNORECASE)
        }

        # Scam scoring weights
        self.pattern_weights = {
            "phishing_urls": 0.9,
            "urgent_keywords": 0.7,
            "authority_claims": 0.8,
            "financial_terms": 0.6,
            "social_engineering": 0.5,
            "technical_deception": 0.7
        }

    def predict(self, texts: List[str], return_probabilities: bool = True) -> Dict[str, Any]:
        """Predict scam probability for texts"""
        if not texts:
            return {"predictions": [], "probabilities": [], "labels": []}

        self.logger.debug(f"Classifying {len(texts)} texts with Simple Classifier")

        predictions = []
        probabilities = []
        labels = []
        confidence_scores = []

        for text in texts:
            result = self._classify_single(text)
            predictions.append(result["prediction"])
            probabilities.append(result["probabilities"])
            labels.append(result["label"])
            confidence_scores.append(result["confidence_score"])

        return {
            "predictions": predictions,
            "probabilities": probabilities if return_probabilities else [],
            "labels": labels,
            "confidence_scores": confidence_scores,
            "scam_probabilities": [p[1] for p in probabilities] if return_probabilities else []
        }

    def _classify_single(self, text: str) -> Dict[str, Any]:
        """Classify single text"""
        if not text or not isinstance(text, str):
            return {
                "prediction": 0,
                "probabilities": [1.0, 0.0],
                "label": "legitimate",
                "confidence_score": 0.5
            }

        text_lower = text.lower()

        # Calculate scam score based on patterns
        scam_score = 0.0
        pattern_matches = {}

        for pattern_name, pattern in self.scam_patterns.items():
            matches = len(pattern.findall(text_lower))
            if matches > 0:
                weight = self.pattern_weights[pattern_name]
                scam_score += weight * min(matches, 3)  # Cap at 3 matches per pattern
                pattern_matches[pattern_name] = matches

        # Additional scoring factors
        additional_score = self._additional_scoring(text)
        scam_score += additional_score

        # Normalize score to 0-1 with better scaling
        # Use sigmoid-like normalization for smoother distribution
        max_possible_score = 5.0  # Reasonable maximum
        normalized_score = scam_score / max_possible_score
        scam_probability = min(normalized_score / (1 + normalized_score * 0.5), 1.0)  # Smooth curve
        
        # Ensure minimum variation - if score is very low, still give some probability
        if scam_score > 0.1:
            scam_probability = max(scam_probability, 0.15)  # Minimum 15% if any indicators
        
        legitimate_probability = 1.0 - scam_probability

        # Determine prediction
        prediction = 1 if scam_probability > 0.5 else 0
        label = "scam" if prediction == 1 else "legitimate"

        # Calculate confidence (distance from 0.5 threshold, with minimum confidence)
        base_confidence = abs(scam_probability - 0.5) * 2
        # Add confidence boost if multiple patterns match
        pattern_boost = min(len(pattern_matches) * 0.1, 0.2)
        confidence_score = min(base_confidence + pattern_boost, 0.95)

        return {
            "prediction": prediction,
            "probabilities": [legitimate_probability, scam_probability],
            "label": label,
            "confidence_score": confidence_score,
            "pattern_matches": pattern_matches
        }

    def _additional_scoring(self, text: str) -> float:
        """Additional scoring factors"""
        score = 0.0
        text_lower = text.lower()

        # URL presence
        if 'http' in text_lower or 'www' in text_lower:
            score += 0.3

        # Email presence
        if '@' in text:
            score += 0.2

        # Phone numbers
        import re
        if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text):
            score += 0.2

        # Money mentions
        if '$' in text or 'rs' in text_lower or 'rupees' in text_lower:
            score += 0.3

        # Excessive punctuation
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            score += min(exclamation_count * 0.1, 0.3)

        # ALL CAPS words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        if caps_words > 1:
            score += min(caps_words * 0.1, 0.2)

        # Length factors
        word_count = len(words)
        if word_count < 5:
            score -= 0.1  # Very short messages less likely to be scams
        elif word_count > 50:
            score += 0.1  # Longer messages can be more suspicious

        return score

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict for single text"""
        results = self.predict([text])
        return {
            "prediction": results["predictions"][0],
            "probability": results["probabilities"][0] if results["probabilities"] else [],
            "label": results["labels"][0],
            "confidence_score": results["confidence_scores"][0],
            "scam_probability": results["scam_probabilities"][0] if results["scam_probabilities"] else 0.0
        }

    def get_attention_weights(self, text: str) -> Dict[str, Any]:
        """Get simple attention weights"""
        if not text:
            return {"tokens": [], "attention_weights": {}, "important_tokens": []}

        words = text.split()
        attention_weights = {}

        # Simple importance scoring
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            score = 0.1  # Base score

            # Check against scam patterns
            for pattern_name, pattern in self.scam_patterns.items():
                if pattern.search(word_lower):
                    score += self.pattern_weights[pattern_name] * 0.5

            # Position bias
            if i == 0:
                score += 0.2  # First word
            elif i < len(words) // 3:
                score += 0.1  # Early words

            attention_weights[word] = min(score, 1.0)

        # Sort by importance
        important_tokens = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "tokens": words,
            "attention_weights": attention_weights,
            "important_tokens": important_tokens
        }

    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns"""
        prediction = self.predict_single(text)
        attention = self.get_attention_weights(text)

        # Simple suspicious phrases extraction
        suspicious_phrases = []
        text_lower = text.lower()

        scam_indicators = [
            "urgent action", "verify account", "security alert", "official notice",
            "transfer money", "send payment", "click here", "download now",
            "government agency", "police department", "court order"
        ]

        for indicator in scam_indicators:
            if indicator in text_lower:
                suspicious_phrases.append(indicator)

        # Extract values from prediction dict for easier access
        scam_probability = prediction.get('scam_probability', 0.0) if isinstance(prediction, dict) else 0.0
        confidence_score = prediction.get('confidence_score', 0.5) if isinstance(prediction, dict) else 0.5
        label = prediction.get('label', 'unknown') if isinstance(prediction, dict) else 'unknown'

        return {
            "prediction": prediction,
            "scam_probability": float(scam_probability),
            "confidence_score": float(confidence_score),
            "label": label,
            "attention_analysis": attention,
            "suspicious_phrases": suspicious_phrases,
            "text_length": len(text),
            "word_count": len(text.split())
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": "SimpleDistilBERTClassifier",
            "model_path": self.model_path,
            "device": str(self.device),
            "max_seq_length": self.max_seq_length,
            "num_classes": 2,
            "has_model": True
        }


# Create global instance
distilbert_classifier = SimpleDistilBERTClassifier()