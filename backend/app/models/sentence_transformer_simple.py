"""
Simplified Sentence Transformer model for PhishNetra
Works without torch/transformers dependencies
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..core.logging import LoggerMixin


class SimpleSentenceTransformer(LoggerMixin):
    """
    Simple sentence transformer without ML dependencies
    Uses basic text analysis for semantic understanding
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or "simple-transformer"
        self.device = device or "cpu"
        self.max_seq_length = 512

        self.logger.info(f"Initializing Simple Sentence Transformer")

        # Simple vocabulary for scam detection
        self.scam_keywords = {
            # Financial
            "transfer": 0.8, "money": 0.7, "account": 0.6, "bank": 0.6, "pay": 0.5,
            "credit": 0.5, "debit": 0.5, "loan": 0.6, "investment": 0.7, "profit": 0.6,

            # Urgency
            "urgent": 0.9, "immediate": 0.8, "now": 0.6, "quick": 0.5, "fast": 0.5,
            "deadline": 0.7, "expires": 0.6, "limited": 0.5, "time": 0.4,

            # Authority
            "official": 0.8, "government": 0.9, "police": 0.8, "irs": 0.9, "fbi": 0.9,
            "court": 0.7, "legal": 0.6, "department": 0.5, "agency": 0.6,

            # Social engineering
            "help": 0.5, "emergency": 0.7, "family": 0.6, "friend": 0.5, "trouble": 0.6,
            "support": 0.4, "assist": 0.4, "problem": 0.4,

            # Technical
            "verify": 0.6, "confirm": 0.5, "security": 0.7, "alert": 0.6, "click": 0.7,
            "link": 0.6, "download": 0.5, "attachment": 0.5, "login": 0.5, "password": 0.6
        }

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to simple embeddings"""
        if not texts:
            return np.array([])

        embeddings = []
        for text in texts:
            embedding = self._encode_single(text)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        if not text:
            return np.zeros(384, dtype=np.float32)  # Match standard embedding size

        text_lower = text.lower()

        # Create simple embedding based on keyword matches
        embedding = np.zeros(384, dtype=np.float32)

        # Base features
        word_count = len(text.split())
        char_count = len(text)
        has_exclamation = '!' in text
        has_question = '?' in text
        has_money = '$' in text or 'rs' in text_lower or 'rupees' in text_lower

        # Set basic features
        embedding[0] = min(word_count / 100, 1.0)  # Normalized word count
        embedding[1] = min(char_count / 1000, 1.0)  # Normalized char count
        embedding[2] = 1.0 if has_exclamation else 0.0
        embedding[3] = 1.0 if has_question else 0.0
        embedding[4] = 1.0 if has_money else 0.0

        # Keyword-based features
        keyword_scores = []
        for keyword, weight in self.scam_keywords.items():
            if keyword in text_lower:
                keyword_scores.append(weight)

        if keyword_scores:
            avg_keyword_score = np.mean(keyword_scores)
            embedding[5] = avg_keyword_score
            embedding[6] = max(keyword_scores)
            embedding[7] = len(keyword_scores) / 10  # Normalized count

        # Pattern-based features
        embedding[8] = 1.0 if 'http' in text_lower or 'www' in text_lower else 0.0
        embedding[9] = 1.0 if '@' in text else 0.0
        embedding[10] = 1.0 if any(char.isdigit() for char in text) else 0.0

        return embedding

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self._encode_single(text)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity"""
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_intent_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive intent features for all scam types
        Uses INTENT-BASED reasoning, not just keyword matching
        """
        text_lower = text.lower()
        
        # Count occurrences for weighted scoring
        def count_intent(phrases):
            return sum(1 for phrase in phrases if phrase in text_lower) / max(len(phrases), 1)

        features = {
            # 1. FINANCIAL MANIPULATION - Intent to influence money transfer
            "financial_transfer": min(count_intent([
                "transfer", "send money", "payment", "pay now", "wire", "refund",
                "debit", "credit", "withdraw", "deposit", "money moved", "approve payment"
            ]) * 2.0, 1.0),
            
            "account_access": min(count_intent([
                "account", "login", "password", "verify", "access", "credentials",
                "update account", "account suspended", "account locked"
            ]) * 1.5, 1.0),

            # 2. PHISHING & REDIRECTION - Intent to redirect/harvest
            "phishing_indicators": min(count_intent([
                "click link", "click here", "download", "attachment", "open link",
                "visit", "go to", "redirect", "follow link", "scan qr"
            ]) * 2.0, 1.0),
            
            "url_presence": 1.0 if ("http" in text_lower or "www" in text_lower or ".com" in text_lower or ".net" in text_lower) else 0.0,

            # 3. IDENTITY VERIFICATION ABUSE - Intent to extract personal info
            "personal_data_request": min(count_intent([
                "information", "details", "personal", "ssn", "social security",
                "kyc", "verify identity", "confirm details", "update information",
                "provide", "submit", "enter your"
            ]) * 2.0, 1.0),

            # 4. AUTHORITY IMPERSONATION - Intent to pretend to be institution
            "authority_claim": min(count_intent([
                "official", "government", "police", "irs", "fbi", "court",
                "department", "agency", "authority", "federal", "state"
            ]) * 1.8, 1.0),
            
            "legal_threat": min(count_intent([
                "legal", "court", "lawsuit", "arrest", "warrant", "charge",
                "violation", "penalty", "fine", "action required"
            ]) * 1.5, 1.0),

            # 5. SOCIAL ENGINEERING - Intent to manipulate trust
            "trust_building": min(count_intent([
                "secure", "safe", "protect", "guarantee", "trusted", "verified",
                "legitimate", "authorized", "certified", "official partner"
            ]) * 1.2, 1.0),
            
            "polite_tone": 1.0 if any(word in text_lower for word in ["please", "kindly", "thank you", "sir", "madam"]) else 0.0,

            # 6. EMOTIONAL MANIPULATION - Intent to trigger emotions
            "emotional_manipulation": min(count_intent([
                "help", "emergency", "family", "trouble", "urgent help",
                "desperate", "please help", "in need", "suffering"
            ]) * 1.5, 1.0),
            
            "high_urgency": min(count_intent([
                "urgent", "immediate", "now", "quick", "asap", "right away",
                "immediately", "without delay", "act now"
            ]) * 2.0, 1.0),
            
            "time_pressure": min(count_intent([
                "deadline", "expires", "limited time", "today only", "expiring soon",
                "last chance", "final notice", "within 24 hours"
            ]) * 1.8, 1.0),
            
            "fear_appeal": min(count_intent([
                "suspended", "closed", "terminated", "blocked", "frozen",
                "compromised", "hacked", "breach", "unauthorized"
            ]) * 1.5, 1.0),

            # 7. JOB/WORK/INVESTMENT SCAM - Intent to extract money under opportunity
            "job_opportunity": min(count_intent([
                "job", "work from home", "employment", "position", "hiring",
                "opportunity", "earn money", "make money", "income"
            ]) * 1.5, 1.0),
            
            "investment_promise": min(count_intent([
                "investment", "profit", "return", "guaranteed return", "high yield",
                "crypto", "bitcoin", "stock", "trading", "earn"
            ]) * 1.8, 1.0),
            
            "registration_fee": min(count_intent([
                "registration fee", "processing fee", "admin fee", "activation",
                "one-time payment", "small fee", "deposit"
            ]) * 2.0, 1.0),

            # 8. TECH SUPPORT & MALWARE - Intent to convince device compromised
            "tech_support": min(count_intent([
                "support", "virus", "malware", "computer", "device", "system",
                "infected", "compromised", "call us", "tech team"
            ]) * 1.5, 1.0),
            
            "download_request": min(count_intent([
                "download", "install", "update", "patch", "fix", "software",
                "program", "application", "tool"
            ]) * 1.8, 1.0),

            # 9. DELIVERY/COURIER SCAM - Intent to exploit package expectation
            "delivery_mention": min(count_intent([
                "delivery", "package", "courier", "shipment", "tracking",
                "parcel", "order", "dispatch", "out for delivery"
            ]) * 1.8, 1.0),
            
            "delivery_action": min(count_intent([
                "pay delivery", "customs fee", "redelivery", "reschedule",
                "confirm address", "update delivery"
            ]) * 2.0, 1.0),

            # 10. LOTTERY/REWARD SCAM - Intent to promise reward
            "lottery_mention": min(count_intent([
                "lottery", "winner", "prize", "reward", "claim", "congratulations",
                "you won", "selected", "lucky", "jackpot"
            ]) * 2.0, 1.0),
            
            "claim_action": min(count_intent([
                "claim now", "claim prize", "claim reward", "claim money",
                "processing fee", "tax payment", "claim fee"
            ]) * 2.0, 1.0),

            # 11. HYBRID/EVOLVING - Multiple intents combined
            "vague_reference": 1.0 if any(phrase in text_lower for phrase in [
                "your account", "your payment", "your order", "your request",
                "recent activity", "previous transaction"
            ]) and not any(specific in text_lower for specific in [
                "account number", "order number", "transaction id"
            ]) else 0.0,
            
            "action_required": 1.0 if any(phrase in text_lower for phrase in [
                "action required", "immediate action", "take action", "respond",
                "reply", "confirm", "verify"
            ]) else 0.0
        }

        return features

    def analyze_semantic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze semantic patterns"""
        embedding = self.encode_single(text)
        intent_features = self.get_intent_features(text)

        # Calculate pattern scores
        pattern_scores = {
            "financial_manipulation": (
                intent_features["financial_transfer"] * 0.4 +
                intent_features["account_access"] * 0.6
            ),
            "urgency_pressure": (
                intent_features["high_urgency"] * 0.5 +
                intent_features["time_pressure"] * 0.5
            ),
            "authority_deception": (
                intent_features["authority_claim"] * 0.6 +
                intent_features["legal_threat"] * 0.4
            ),
            "social_engineering": (
                intent_features["emotional_manipulation"] * 0.5 +
                intent_features["trust_building"] * 0.3 +
                intent_features["personal_data_request"] * 0.2
            ),
            "technical_deception": (
                intent_features["phishing_indicators"] * 0.7 +
                intent_features["account_access"] * 0.3
            )
        }

        dominant_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        
        # Calculate dynamic confidence based on pattern strength and feature count
        max_pattern_score = dominant_pattern[1]
        feature_count = sum(1 for v in intent_features.values() if v > 0.5)
        dynamic_confidence = min(0.6 + (max_pattern_score * 0.3) + (min(feature_count, 5) * 0.02), 0.95)

        return {
            "embedding": embedding,
            "intent_features": intent_features,
            "pattern_scores": pattern_scores,
            "dominant_pattern": dominant_pattern,
            "embedding_confidence": float(dynamic_confidence)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": "SimpleSentenceTransformer",
            "model_name": self.model_name,
            "embedding_dimension": 384,
            "max_seq_length": self.max_seq_length,
            "device": str(self.device),
            "has_model": True
        }


# Create global instance
sentence_transformer = SimpleSentenceTransformer()