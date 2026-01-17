"""
FastText embedding model for PhishNetra
Handles spelling errors, slang, and multilingual text
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path

from ..core.logging import LoggerMixin
from ..core.config import settings


class FastTextModel(LoggerMixin):
    """
    FastText model for robust text embeddings
    Handles noisy text with spelling variations and slang
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.fasttext_model_path
        self.model = None
        self.word_vectors = {}
        self.vector_dim = 300  # Default FastText dimension

        self.logger.info("Initializing FastText model")

        # Load or create model
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            self.logger.warning(f"FastText model not found at {self.model_path}. Using fallback embeddings.")
            self._create_fallback_model()

    def _load_model(self):
        """Load FastText model from disk"""
        try:
            import fasttext

            self.logger.info(f"Loading FastText model from {self.model_path}")
            self.model = fasttext.load_model(self.model_path)

            # Get vocabulary and vectors
            self.word_vectors = {}
            words = self.model.get_words()
            for word in words[:10000]:  # Limit vocabulary for memory
                self.word_vectors[word] = self.model.get_word_vector(word)

            self.vector_dim = self.model.get_dimension()
            self.logger.info(f"Loaded FastText model with {len(words)} words, dimension {self.vector_dim}")

        except ImportError:
            self.logger.error("FastText library not available. Install with: pip install fasttext-wheel")
            self._create_fallback_model()
        except Exception as e:
            self.logger.error(f"Error loading FastText model: {e}")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Create a simple fallback embedding model"""
        self.logger.info("Creating fallback embedding model")

        # Simple vocabulary for common scam-related terms
        scam_vocab = [
            # Financial terms
            "money", "bank", "account", "transfer", "payment", "credit", "debit", "loan",
            "investment", "profit", "loss", "balance", "transaction", "fee", "charge",

            # Urgency terms
            "urgent", "immediate", "quick", "fast", "now", "today", "immediately",
            "emergency", "important", "critical", "action", "required",

            # Authority terms
            "irs", "fbi", "police", "government", "official", "authority", "legal",
            "court", "law", "compliance", "regulation", "department",

            # Social engineering
            "friend", "family", "relative", "help", "support", "assist", "problem",
            "issue", "trouble", "difficulty", "situation", "condition",

            # Common scam patterns
            "verify", "confirm", "validate", "authenticate", "security", "safe",
            "protect", "secure", "click", "link", "website", "email", "message",

            # Hinglish terms
            "rupees", "account", "message", "please", "okay", "friend", "amount",
            "transaction", "number", "code", "password", "login", "send"
        ]

        self.vector_dim = 100  # Smaller dimension for fallback

        # Create random but deterministic vectors
        np.random.seed(42)
        for word in scam_vocab:
            self.word_vectors[word] = np.random.randn(self.vector_dim).astype(np.float32)

        # Add common variations
        variations = {
            "acc": self.word_vectors.get("account", np.random.randn(self.vector_dim)),
            "amt": self.word_vectors.get("amount", np.random.randn(self.vector_dim)),
            "msg": self.word_vectors.get("message", np.random.randn(self.vector_dim)),
            "pls": self.word_vectors.get("please", np.random.randn(self.vector_dim)),
            "frnd": self.word_vectors.get("friend", np.random.randn(self.vector_dim)),
            "txn": self.word_vectors.get("transaction", np.random.randn(self.vector_dim)),
            "rs": self.word_vectors.get("rupees", np.random.randn(self.vector_dim)),
        }

        self.word_vectors.update(variations)
        self.logger.info(f"Created fallback model with {len(self.word_vectors)} word vectors")

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get vector for a single word"""
        word = word.lower().strip()

        # Direct lookup
        if word in self.word_vectors:
            return self.word_vectors[word]

        # Try subword lookup if FastText model available
        if self.model and hasattr(self.model, 'get_word_vector'):
            try:
                return self.model.get_word_vector(word)
            except:
                pass

        # Fallback: average of character n-grams
        return self._get_subword_vector(word)

    def _get_subword_vector(self, word: str) -> np.ndarray:
        """Get subword-based vector for OOV words"""
        if not word:
            return np.zeros(self.vector_dim, dtype=np.float32)

        # Generate character n-grams (3-6 characters)
        ngrams = set()
        word = f"<{word}>"

        for n in range(3, min(7, len(word) + 1)):
            for i in range(len(word) - n + 1):
                ngrams.add(word[i:i+n])

        # Average vectors of known n-grams
        vectors = []
        for ngram in ngrams:
            if ngram in self.word_vectors:
                vectors.append(self.word_vectors[ngram])

        if vectors:
            return np.mean(vectors, axis=0).astype(np.float32)

        # Ultimate fallback: random vector based on word hash
        np.random.seed(hash(word) % 2**32)
        return np.random.randn(self.vector_dim).astype(np.float32)

    def get_sentence_vector(self, text: str, method: str = "average") -> np.ndarray:
        """
        Get sentence-level embedding

        Args:
            text: Input text
            method: Aggregation method ("average", "tfidf", "sif")

        Returns:
            Sentence embedding vector
        """
        if not text or not isinstance(text, str):
            return np.zeros(self.vector_dim, dtype=np.float32)

        # Tokenize (simple whitespace + punctuation split)
        words = self._tokenize(text)

        if not words:
            return np.zeros(self.vector_dim, dtype=np.float32)

        # Get word vectors
        word_vectors = []
        word_weights = []

        for word in words:
            vector = self.get_word_vector(word)
            word_vectors.append(vector)

            # Simple weighting (can be enhanced with TF-IDF)
            weight = 1.0
            if len(word) < 3:  # Short words get lower weight
                weight = 0.5
            word_weights.append(weight)

        word_vectors = np.array(word_vectors)
        word_weights = np.array(word_weights)

        # Apply aggregation method
        if method == "average":
            return np.average(word_vectors, axis=0, weights=word_weights).astype(np.float32)

        elif method == "sif":  # Smooth Inverse Frequency
            # Simple SIF implementation
            sif_weights = np.array([0.1 if w < 0.01 else 1.0 for w in word_weights])
            weighted_avg = np.average(word_vectors, axis=0, weights=sif_weights)
            return weighted_avg.astype(np.float32)

        else:  # Default to average
            return np.mean(word_vectors, axis=0).astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for FastText"""
        import re

        # Convert to lowercase
        text = text.lower()

        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)

        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def encode_batch(self, texts: List[str], method: str = "average") -> np.ndarray:
        """
        Encode multiple texts to embeddings

        Args:
            texts: List of input texts
            method: Aggregation method

        Returns:
            Array of embeddings (n_texts, vector_dim)
        """
        self.logger.debug(f"Encoding {len(texts)} texts with FastText")

        embeddings = []
        for text in texts:
            embedding = self.get_sentence_vector(text, method)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def find_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar words based on embeddings

        Args:
            word: Query word
            top_k: Number of similar words to return

        Returns:
            List of (word, similarity) tuples
        """
        if not self.word_vectors:
            return []

        query_vector = self.get_word_vector(word)

        similarities = []
        for vocab_word, vector in self.word_vectors.items():
            if vocab_word != word:
                similarity = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
                similarities.append((vocab_word, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": "FastText",
            "vector_dimension": self.vector_dim,
            "vocabulary_size": len(self.word_vectors),
            "model_path": self.model_path,
            "has_fasttext_model": self.model is not None
        }

    def save_model(self, path: str):
        """Save model state"""
        model_state = {
            "word_vectors": self.word_vectors,
            "vector_dim": self.vector_dim,
            "model_path": self.model_path
        }

        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

        self.logger.info(f"Saved FastText model state to {path}")

    def load_model_state(self, path: str):
        """Load model state"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model_state = pickle.load(f)

            self.word_vectors = model_state.get("word_vectors", {})
            self.vector_dim = model_state.get("vector_dim", 300)
            self.model_path = model_state.get("model_path", "")

            self.logger.info(f"Loaded FastText model state from {path}")


# Global FastText model instance
fasttext_model = FastTextModel()