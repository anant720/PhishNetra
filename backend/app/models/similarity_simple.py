"""
Simple similarity engine for PhishNetra
Works without FAISS or advanced ML dependencies
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import difflib

from ..core.logging import LoggerMixin


class SimpleSimilarityEngine(LoggerMixin):
    """
    Simple similarity search without FAISS
    Uses basic text similarity for scam variant detection
    """

    def __init__(self, index_path: Optional[str] = None, embedding_model=None):
        self.index_path = index_path or "./models/simple_similarity.json"
        self.embedding_model = embedding_model  # Optional for advanced similarity
        self.scam_texts = []
        self.scam_metadata = []

        self.logger.info(f"Initializing Simple Similarity Engine")

        # Load existing data if available
        self._load_index()

        # Add some default scam examples if none exist
        if not self.scam_texts:
            self._add_default_examples()

    def _add_default_examples(self):
        """Add default scam examples"""
        default_scams = [
            "Your account has been suspended. Click here to verify: http://fakebank.com/verify",
            "URGENT: Transfer $500 immediately or face account closure",
            "Official IRS notice: You owe $2,450 in taxes. Pay now to avoid arrest",
            "FBI investigation: Your account is linked to fraud. Verify identity here",
            "Congratulations! You've won $1,000,000. Send $100 processing fee to claim",
            "Security alert: Unusual activity detected on your account. Confirm here",
            "Police department: Warrant issued for your arrest. Pay $500 fine immediately",
            "Bank transfer required: Send $1,200 to complete international transaction",
            "Government refund: Claim your $1,800 stimulus check. Provide SSN",
            "Urgent help needed: Send money to family member in emergency"
        ]

        for scam in default_scams:
            self.scam_texts.append(scam)
            self.scam_metadata.append({
                "source": "default",
                "category": "scam",
                "confidence": 0.9
            })

        self.logger.info(f"Added {len(default_scams)} default scam examples")

    def _load_index(self):
        """Load index from file"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self.scam_texts = data.get('texts', [])
                    self.scam_metadata = data.get('metadata', [])
                self.logger.info(f"Loaded {len(self.scam_texts)} scam examples from {self.index_path}")
            except Exception as e:
                self.logger.error(f"Error loading similarity index: {e}")

    def _save_index(self):
        """Save index to file"""
        try:
            data = {
                'texts': self.scam_texts,
                'metadata': self.scam_metadata
            }
            with open(self.index_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.scam_texts)} scam examples to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Error saving similarity index: {e}")

    def add_scam_examples(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add scam examples to the index"""
        if not texts:
            return

        self.scam_texts.extend(texts)

        if metadata:
            self.scam_metadata.extend(metadata)
        else:
            default_metadata = [{"source": "user", "category": "scam"} for _ in texts]
            self.scam_metadata.extend(default_metadata)

        # Save updated index
        self._save_index()

        self.logger.info(f"Added {len(texts)} scam examples. Total: {len(self.scam_texts)}")

    def search_similar(self, query_text: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar scam texts"""
        if not self.scam_texts:
            return []

        top_k = top_k or 5
        threshold = threshold or 0.3

        similarities = []

        for i, scam_text in enumerate(self.scam_texts):
            similarity_score = self._calculate_similarity(query_text, scam_text)

            if similarity_score >= threshold:
                metadata = self.scam_metadata[i] if i < len(self.scam_metadata) else {}
                similarities.append((scam_text, similarity_score, metadata))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Convert to lowercase for comparison
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Exact match gets highest score
        if text1_lower == text2_lower:
            return 1.0

        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, text1_lower, text2_lower)
        sequence_similarity = matcher.ratio()

        # Keyword overlap similarity
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())

        if words1 and words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard_similarity = len(intersection) / len(union)
        else:
            jaccard_similarity = 0.0

        # Combine similarities (weighted average)
        combined_similarity = (sequence_similarity * 0.6) + (jaccard_similarity * 0.4)

        return combined_similarity

    def find_scam_variants(self, query_text: str) -> Dict[str, Any]:
        """Find scam variants for the query text"""
        similar_scams = self.search_similar(query_text, top_k=3)

        if not similar_scams:
            return {
                "has_similar_scams": False,
                "similar_count": 0,
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "variant_score": 0.0,
                "similar_scams": []
            }

        similarities = [sim for _, sim, _ in similar_scams]
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        # Calculate variant score (how likely this is a scam variant)
        variant_score = min(max_similarity * 0.7 + avg_similarity * 0.3, 1.0)

        # Determine similarity level
        if max_similarity > 0.8:
            similarity_level = "very_high"
        elif max_similarity > 0.6:
            similarity_level = "high"
        elif max_similarity > 0.4:
            similarity_level = "moderate"
        else:
            similarity_level = "low"

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
                for text, sim, metadata in similar_scams
            ]
        }

    def detect_unseen_patterns(self, query_text: str) -> Dict[str, Any]:
        """Detect if text contains unseen scam patterns"""
        variant_analysis = self.find_scam_variants(query_text)

        # Simple pattern analysis
        text_features = self._analyze_text_features(query_text)

        # Combine scores
        unseen_score = (
            variant_analysis["variant_score"] * 0.6 +
            text_features["suspicious_score"] * 0.4
        )

        return {
            "variant_analysis": variant_analysis,
            "text_features": text_features,
            "unseen_pattern_score": unseen_score,
            "likely_unseen_scam": unseen_score > 0.5
        }

    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze text for suspicious features"""
        text_lower = text.lower()

        suspicious_features = {
            "has_urls": 'http' in text_lower or 'www' in text_lower,
            "has_email": '@' in text,
            "has_phone": bool(difflib.get_close_matches(text_lower, ['phone', 'call', 'contact'], cutoff=0.8)),
            "has_money": '$' in text or 'rs' in text_lower or 'rupees' in text_lower,
            "excessive_punctuation": text.count('!') > 2 or text.count('?') > 3,
            "all_caps_words": sum(1 for word in text.split() if word.isupper() and len(word) > 2) > 1,
            "urgent_words": any(word in text_lower for word in ['urgent', 'immediate', 'now', 'quick', 'fast']),
            "authority_words": any(word in text_lower for word in ['official', 'government', 'police', 'irs', 'fbi', 'court'])
        }

        suspicious_score = sum(1 for feature, present in suspicious_features.items() if present) / len(suspicious_features)

        return {
            "suspicious_features": suspicious_features,
            "suspicious_score": suspicious_score,
            "feature_count": sum(1 for present in suspicious_features.values() if present)
        }

    def save_index(self, path: Optional[str] = None):
        """Save the index"""
        save_path = path or self.index_path
        self._save_index()

    def get_index_info(self) -> Dict[str, Any]:
        """Get index information"""
        return {
            "index_type": "SimpleSimilarity",
            "total_vectors": len(self.scam_texts),
            "has_embedding_model": self.embedding_model is not None,
            "scam_texts_count": len(self.scam_texts)
        }


# Create global instance
similarity_engine = SimpleSimilarityEngine()