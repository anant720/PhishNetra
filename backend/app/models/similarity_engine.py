"""
FAISS similarity engine for PhishNetra
Efficient detection of scam variants and unseen patterns
"""

import os
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

from ..core.logging import LoggerMixin
from ..core.config import settings


class SimilarityEngine(LoggerMixin):
    """
    FAISS-based similarity search engine
    Detects scam variants by comparing against known scam embeddings
    """

    def __init__(self, index_path: Optional[str] = None, embedding_model=None):
        self.index_path = index_path or settings.faiss_index_path
        self.embedding_model = embedding_model  # SentenceTransformer instance
        self.index = None
        self.scam_texts = []
        self.scam_metadata = []
        self.embedding_dim = 384  # Default for MiniLM

        self.logger.info(f"Initializing FAISS similarity engine from {self.index_path}")

        # Load or create index
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self.logger.warning(f"FAISS index not found at {self.index_path}. Creating empty index.")
            self._create_empty_index()

    def _create_empty_index(self):
        """Create empty FAISS index"""
        try:
            import faiss

            # Determine embedding dimension
            if self.embedding_model:
                self.embedding_dim = self.embedding_model.get_model_info()["embedding_dimension"]
            else:
                self.embedding_dim = 384  # Default MiniLM dimension

            # Create IVF index for efficiency
            nlist = min(100, max(4, len(self.scam_texts) // 39))  # At least 4, at most 100
            quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            self.logger.info(f"Created empty FAISS index with dimension {self.embedding_dim}")

        except ImportError:
            self.logger.error("FAISS not available. Install with: pip install faiss-cpu")
            self.index = None
        except Exception as e:
            self.logger.error(f"Error creating FAISS index: {e}")
            self.index = None

    def _load_index(self):
        """Load FAISS index from disk"""
        try:
            import faiss

            self.logger.info(f"Loading FAISS index from {self.index_path}")

            # Load FAISS index
            self.index = faiss.read_index(self.index_path)

            # Load metadata
            metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.scam_texts = metadata.get('texts', [])
                    self.scam_metadata = metadata.get('metadata', [])

            self.embedding_dim = self.index.d
            self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors, dimension {self.embedding_dim}")

        except ImportError:
            self.logger.error("FAISS not available")
            self._create_empty_index()
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            self._create_empty_index()

    def add_scam_examples(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add scam examples to the similarity index

        Args:
            texts: List of scam texts
            metadata: Optional metadata for each text
        """
        if not texts:
            return

        if not self.embedding_model:
            self.logger.error("No embedding model available for encoding texts")
            return

        self.logger.info(f"Adding {len(texts)} scam examples to similarity index")

        try:
            # Encode texts
            embeddings = self.embedding_model.encode(texts)

            # Add to index
            if self.index is not None:
                # Train index if needed (for IVF)
                if not self.index.is_trained and len(texts) > 256:
                    self.logger.info("Training FAISS index")
                    self.index.train(embeddings.astype(np.float32))

                # Add vectors
                self.index.add(embeddings.astype(np.float32))

                # Store texts and metadata
                self.scam_texts.extend(texts)
                if metadata:
                    self.scam_metadata.extend(metadata)
                else:
                    # Default metadata
                    default_metadata = [{"source": "unknown", "category": "scam"} for _ in texts]
                    self.scam_metadata.extend(default_metadata)

                self.logger.info(f"Added {len(texts)} vectors to FAISS index. Total: {self.index.ntotal}")

        except Exception as e:
            self.logger.error(f"Error adding scam examples: {e}")

    def search_similar(self, query_text: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar scam texts

        Args:
            query_text: Query text to search for
            top_k: Number of top results to return
            threshold: Similarity threshold (0-1)

        Returns:
            List of (text, similarity_score, metadata) tuples
        """
        if not self.embedding_model or self.index is None or self.index.ntotal == 0:
            self.logger.debug("Similarity search not available (no model or empty index)")
            return []

        top_k = top_k or settings.max_similar_results
        threshold = threshold or settings.similarity_threshold

        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query_text])[0].astype(np.float32)
            query_embedding = np.expand_dims(query_embedding, axis=0)

            # Search
            nprobe = min(settings.faiss_nprobe, self.index.nlist)
            self.index.nprobe = nprobe

            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))

            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.scam_texts):
                    continue

                similarity = float(score)

                # Apply threshold
                if similarity >= threshold:
                    text = self.scam_texts[idx]
                    metadata = self.scam_metadata[idx] if idx < len(self.scam_metadata) else {}
                    results.append((text, similarity, metadata))

            # Sort by similarity (FAISS returns in descending order, but let's be sure)
            results.sort(key=lambda x: x[1], reverse=True)

            self.logger.debug(f"Found {len(results)} similar texts above threshold {threshold}")

            return results[:top_k]

        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

    def find_scam_variants(self, query_text: str) -> Dict[str, Any]:
        """
        Find scam variants and analyze similarity patterns

        Args:
            query_text: Input text to analyze

        Returns:
            Analysis of scam variants
        """
        similar_scams = self.search_similar(query_text)

        if not similar_scams:
            return {
                "has_similar_scams": False,
                "similar_count": 0,
                "max_similarity": 0.0,
                "similar_scams": [],
                "variant_score": 0.0
            }

        # Analyze similarity patterns
        similarities = [sim for _, sim, _ in similar_scams]
        max_similarity = max(similarities)
        avg_similarity = np.mean(similarities)

        # Calculate variant score (how likely this is a scam variant)
        # Higher score = more likely to be a variant
        variant_score = min(max_similarity * 0.8 + avg_similarity * 0.2, 1.0)

        # Categorize similarity
        if max_similarity > 0.9:
            similarity_level = "very_high"
        elif max_similarity > 0.8:
            similarity_level = "high"
        elif max_similarity > 0.7:
            similarity_level = "moderate"
        else:
            similarity_level = "low"

        # Extract common patterns (simplified)
        common_phrases = self._extract_common_phrases(query_text, similar_scams)

        return {
            "has_similar_scams": True,
            "similar_count": len(similar_scams),
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity,
            "variant_score": variant_score,
            "similarity_level": similarity_level,
            "similar_scams": [
                {
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "similarity": sim,
                    "metadata": metadata
                }
                for text, sim, metadata in similar_scams[:5]  # Top 5
            ],
            "common_phrases": common_phrases
        }

    def _extract_common_phrases(self, query_text: str, similar_scams: List[Tuple[str, float, Dict[str, Any]]]) -> List[str]:
        """Extract common phrases between query and similar scams"""
        if not similar_scams:
            return []

        query_lower = query_text.lower()
        common_phrases = []

        # Common scam phrases to check
        scam_phrases = [
            "urgent", "immediate", "transfer money", "send payment",
            "account", "bank", "verify", "confirm", "security",
            "official", "government", "police", "legal", "court",
            "click here", "download", "attachment", "link"
        ]

        for phrase in scam_phrases:
            # Check if phrase appears in query and similar scams
            query_has_phrase = phrase in query_lower
            similar_have_phrase = sum(1 for text, _, _ in similar_scams if phrase in text.lower())

            if query_has_phrase and similar_have_phrase >= len(similar_scams) * 0.3:  # 30% of similar scams
                common_phrases.append(phrase)

        return common_phrases

    def detect_unseen_patterns(self, query_text: str) -> Dict[str, Any]:
        """
        Detect if text contains unseen scam patterns

        Args:
            query_text: Input text

        Returns:
            Pattern detection results
        """
        # This is a simplified implementation
        # In a full system, this would use clustering or anomaly detection

        variant_analysis = self.find_scam_variants(query_text)

        # Analyze text features that might indicate new scam patterns
        text_features = self._analyze_text_features(query_text)

        # Combine variant analysis with text features
        unseen_score = (
            variant_analysis["variant_score"] * 0.6 +
            text_features["suspicious_feature_score"] * 0.4
        )

        return {
            "variant_analysis": variant_analysis,
            "text_features": text_features,
            "unseen_pattern_score": unseen_score,
            "likely_unseen_scam": unseen_score > 0.6
        }

    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze text features for suspicious patterns"""
        text_lower = text.lower()

        suspicious_features = {
            "multiple_urgency_signals": sum(1 for word in ["urgent", "immediate", "now", "quick", "fast", "immediate"] if word in text_lower),
            "authority_impersonation": sum(1 for word in ["official", "government", "police", "irs", "fbi", "court", "legal"] if word in text_lower),
            "money_requests": sum(1 for word in ["money", "transfer", "send", "payment", "pay", "dollar", "rupees"] if word in text_lower),
            "personal_data_requests": sum(1 for word in ["information", "details", "data", "personal", "private"] if word in text_lower),
            "technical_social_engineering": sum(1 for word in ["click", "link", "download", "attachment", "verify", "confirm"] if word in text_lower),
            "emotional_manipulation": sum(1 for word in ["help", "emergency", "family", "friend", "trouble", "problem"] if word in text_lower)
        }

        # Calculate suspicious score
        feature_score = sum(min(count, 3) * 0.1 for count in suspicious_features.values())
        feature_score = min(feature_score, 1.0)

        return {
            "suspicious_features": suspicious_features,
            "suspicious_feature_score": feature_score,
            "feature_count": sum(1 for count in suspicious_features.values() if count > 0)
        }

    def save_index(self, path: Optional[str] = None):
        """Save FAISS index and metadata"""
        save_path = path or self.index_path

        if self.index is None:
            self.logger.warning("No index to save")
            return

        try:
            import faiss

            # Save FAISS index
            faiss.write_index(self.index, save_path)

            # Save metadata
            metadata_path = save_path.replace('.faiss', '_metadata.pkl')
            metadata = {
                'texts': self.scam_texts,
                'metadata': self.scam_metadata
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            self.logger.info(f"Saved FAISS index to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving FAISS index: {e}")

    def get_index_info(self) -> Dict[str, Any]:
        """Get index information"""
        if self.index is not None:
            return {
                "index_type": "FAISS",
                "total_vectors": self.index.ntotal,
                "embedding_dimension": self.embedding_dim,
                "is_trained": self.index.is_trained,
                "nlist": getattr(self.index, 'nlist', 0),
                "nprobe": getattr(self.index, 'nprobe', 0),
                "has_embedding_model": self.embedding_model is not None,
                "scam_texts_count": len(self.scam_texts)
            }
        else:
            return {
                "index_type": "FAISS",
                "total_vectors": 0,
                "embedding_dimension": self.embedding_dim,
                "is_trained": False,
                "has_embedding_model": self.embedding_model is not None,
                "scam_texts_count": 0
            }


# Global similarity engine instance
# Note: embedding_model will be injected during initialization
similarity_engine = SimilarityEngine()