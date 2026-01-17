"""
Sentence Transformer model for PhishNetra
Captures semantic meaning, intent, and context
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
from pathlib import Path

from ..core.logging import LoggerMixin
from ..core.config import settings


class SentenceTransformerModel(LoggerMixin):
    """
    Sentence Transformer for semantic understanding
    Uses MiniLM or other efficient transformer models
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.sentence_transformer_model
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu")
        self.max_seq_length = settings.max_sequence_length

        self.logger.info(f"Initializing Sentence Transformer model: {self.model_name}")

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load Sentence Transformer model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading SentenceTransformer: {self.model_name}")

            # Load model with optimizations
            model_kwargs = {
                "device": self.device,
                "cache_folder": settings.model_cache_dir
            }

            self.model = SentenceTransformer(self.model_name, **model_kwargs)

            # Set max sequence length
            self.model.max_seq_length = self.max_seq_length

            self.logger.info(f"Loaded SentenceTransformer on {self.device}")
            self.logger.info(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")

        except ImportError:
            self.logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            self._create_fallback_model()
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer: {e}")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Create fallback embedding model"""
        self.logger.warning("Using fallback sentence embedding model")

        # Simple TF-IDF style fallback
        self.model = None
        self.fallback_vocab = self._build_fallback_vocab()

    def _build_fallback_vocab(self) -> Dict[str, np.ndarray]:
        """Build vocabulary for fallback model"""
        # Key scam-related terms and their semantic vectors
        vocab = {
            # Financial manipulation
            "transfer money": np.array([0.8, 0.2, 0.1, -0.3, 0.5]),
            "send payment": np.array([0.7, 0.3, 0.0, -0.2, 0.6]),
            "bank account": np.array([0.1, 0.8, 0.4, -0.1, 0.2]),
            "credit card": np.array([0.2, 0.7, 0.5, 0.0, 0.1]),
            "investment opportunity": np.array([0.6, 0.4, -0.2, 0.8, 0.3]),

            # Urgency and pressure
            "urgent action": np.array([0.9, -0.1, 0.8, 0.2, -0.3]),
            "immediate response": np.array([0.8, 0.0, 0.7, 0.1, -0.2]),
            "time sensitive": np.array([0.7, -0.2, 0.6, 0.3, -0.1]),
            "act now": np.array([0.8, -0.1, 0.5, 0.4, 0.0]),

            # Authority deception
            "official notice": np.array([-0.2, 0.9, 0.3, -0.5, 0.4]),
            "government agency": np.array([-0.1, 0.8, 0.4, -0.6, 0.3]),
            "legal action": np.array([0.1, 0.7, 0.2, -0.4, 0.5]),
            "court order": np.array([0.0, 0.8, 0.1, -0.3, 0.6]),

            # Social engineering
            "help needed": np.array([0.4, 0.3, -0.6, 0.5, 0.2]),
            "family emergency": np.array([0.5, 0.2, -0.7, 0.3, 0.1]),
            "friend in trouble": np.array([0.3, 0.4, -0.5, 0.2, 0.3]),
            "personal information": np.array([-0.3, 0.6, 0.4, 0.1, -0.2]),

            # Technical deception
            "verify account": np.array([0.1, 0.5, 0.3, -0.2, 0.7]),
            "security alert": np.array([-0.1, 0.4, 0.8, 0.2, 0.3]),
            "click link": np.array([0.3, 0.2, 0.6, 0.4, 0.1]),
            "download attachment": np.array([0.4, 0.1, 0.5, 0.3, 0.2]),
        }

        return vocab

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts to sentence embeddings

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        self.logger.debug(f"Encoding {len(texts)} texts with SentenceTransformer")

        if self.model is not None:
            try:
                # Use SentenceTransformer
                embeddings = self.model.encode(
                    texts,
                    batch_size=min(batch_size, len(texts)),
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                return embeddings.astype(np.float32)

            except Exception as e:
                self.logger.error(f"SentenceTransformer encoding failed: {e}")
                return self._fallback_encode(texts)

        else:
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding method"""
        self.logger.debug("Using fallback sentence encoding")

        embeddings = []
        for text in texts:
            embedding = self._get_fallback_embedding(text)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def _get_fallback_embedding(self, text: str) -> np.ndarray:
        """Get fallback embedding for a single text"""
        if not text:
            return np.zeros(5, dtype=np.float32)  # Match vocab dimension

        text = text.lower()
        embedding_dim = 5

        # Check for matching phrases in vocab
        embedding = np.zeros(embedding_dim, dtype=np.float32)
        match_count = 0

        for phrase, vector in self.fallback_vocab.items():
            if phrase in text:
                embedding += vector
                match_count += 1

        # If no matches, create simple bag-of-words style embedding
        if match_count == 0:
            # Simple features: length, punctuation, caps
            features = [
                min(len(text) / 100, 1.0),  # Normalized length
                text.count('!') / 10,       # Exclamation marks
                text.count('?') / 10,       # Question marks
                sum(1 for c in text if c.isupper()) / len(text) if text else 0,  # Uppercase ratio
                text.count('$') / 5         # Money symbols
            ]
            embedding = np.array(features, dtype=np.float32)

        return embedding

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(max(0, min(1, similarity)))

    def find_similar_texts(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (text, similarity) tuples
        """
        if not candidates:
            return []

        query_emb = self.encode_single(query)
        candidate_embs = self.encode(candidates)

        similarities = []
        for i, emb in enumerate(candidate_embs):
            similarity = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append((candidates[i], float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_intent_features(self, text: str) -> Dict[str, float]:
        """
        Extract intent-related features from text

        Args:
            text: Input text

        Returns:
            Dictionary of intent features
        """
        text_lower = text.lower()

        features = {
            # Financial intent
            "financial_transfer": float(any(word in text_lower for word in ["transfer", "send", "pay", "money"])),
            "account_access": float(any(word in text_lower for word in ["account", "login", "password", "verify"])),

            # Urgency indicators
            "high_urgency": float(any(word in text_lower for word in ["urgent", "immediate", "now", "quick", "fast"])),
            "time_pressure": float(any(word in text_lower for word in ["deadline", "expires", "limited", "time"])),

            # Authority signals
            "authority_claim": float(any(word in text_lower for word in ["official", "government", "police", "irs", "fbi"])),
            "legal_threat": float(any(word in text_lower for word in ["legal", "court", "lawsuit", "action"])),

            # Social engineering
            "emotional_manipulation": float(any(word in text_lower for word in ["help", "emergency", "family", "friend", "trouble"])),
            "trust_building": float(any(word in text_lower for word in ["secure", "safe", "protect", "guarantee"])),

            # Technical deception
            "phishing_indicators": float(any(word in text_lower for word in ["click", "link", "download", "attachment"])),
            "personal_data_request": float(any(word in text_lower for word in ["information", "details", "data", "personal"]))
        }

        return features

    def analyze_semantic_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze semantic patterns in text

        Args:
            text: Input text

        Returns:
            Analysis results
        """
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

        return {
            "embedding": embedding,
            "intent_features": intent_features,
            "pattern_scores": pattern_scores,
            "dominant_pattern": max(pattern_scores.items(), key=lambda x: x[1])
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is not None:
            return {
                "model_type": "SentenceTransformer",
                "model_name": self.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "max_seq_length": self.max_seq_length,
                "device": str(self.device),
                "has_model": True
            }
        else:
            return {
                "model_type": "SentenceTransformer",
                "model_name": "fallback",
                "embedding_dimension": 5,
                "max_seq_length": self.max_seq_length,
                "device": "cpu",
                "has_model": False
            }


# Global Sentence Transformer instance
sentence_transformer = SentenceTransformerModel()