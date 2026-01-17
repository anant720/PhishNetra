"""
DistilBERT classifier for PhishNetra
High-accuracy contextual scam detection
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json

from ..core.logging import LoggerMixin
from ..core.config import settings


class ScamClassifier(nn.Module):
    """DistilBERT-based scam classifier"""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 2):
        super(ScamClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load DistilBERT
        from transformers import DistilBertModel

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)

        # Freeze DistilBERT layers (optional fine-tuning)
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class DistilBERTClassifier(LoggerMixin):
    """
    DistilBERT-based scam detection classifier
    Fine-tuned for contextual understanding and pattern recognition
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or settings.distilbert_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu")
        self.max_seq_length = settings.max_sequence_length

        self.model = None
        self.tokenizer = None
        self.label_encoder = None

        self.logger.info(f"Initializing DistilBERT classifier from {self.model_path}")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load DistilBERT model and tokenizer"""
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

            # Try to load fine-tuned model
            if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, 'pytorch_model.bin')):
                self.logger.info(f"Loading fine-tuned DistilBERT from {self.model_path}")
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
                self._load_label_encoder()
            else:
                self.logger.warning(f"Fine-tuned model not found at {self.model_path}. Using base model.")
                self._create_base_model()

            # Move to device
            self.model.to(self.device)
            self.model.eval()

        except ImportError:
            self.logger.error("transformers not available. Install with: pip install transformers torch")
            self._create_fallback_classifier()
        except Exception as e:
            self.logger.error(f"Error loading DistilBERT: {e}")
            self._create_fallback_classifier()

    def _create_base_model(self):
        """Create base DistilBERT model for inference"""
        from transformers import DistilBertForSequenceClassification

        self.logger.info("Creating base DistilBERT model")

        # Create model with 2 classes (scam/not-scam)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )

        # Create simple label encoder
        self.label_encoder = {
            0: "legitimate",
            1: "scam",
            "legitimate": 0,
            "scam": 1
        }

    def _create_fallback_classifier(self):
        """Create simple fallback classifier"""
        self.logger.warning("Using fallback classifier")

        self.model = None
        self.tokenizer = None

        # Simple rule-based fallback
        self.scam_keywords = {
            'urgent': 0.3, 'immediate': 0.3, 'action': 0.2, 'required': 0.2,
            'transfer': 0.4, 'send': 0.3, 'money': 0.4, 'payment': 0.3,
            'account': 0.3, 'bank': 0.3, 'credit': 0.2, 'card': 0.2,
            'verify': 0.2, 'confirm': 0.2, 'security': 0.2, 'alert': 0.3,
            'official': 0.3, 'government': 0.4, 'police': 0.4, 'irs': 0.5,
            'fbi': 0.5, 'legal': 0.3, 'court': 0.3, 'lawsuit': 0.4,
            'click': 0.3, 'link': 0.3, 'download': 0.2, 'attachment': 0.2,
            'help': 0.2, 'emergency': 0.4, 'family': 0.2, 'friend': 0.1
        }

        self.label_encoder = {
            0: "legitimate",
            1: "scam",
            "legitimate": 0,
            "scam": 1
        }

    def _load_label_encoder(self):
        """Load label encoder from model directory"""
        encoder_path = os.path.join(self.model_path, 'label_encoder.json')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'r') as f:
                self.label_encoder = json.load(f)
        else:
            # Default binary classification
            self.label_encoder = {
                0: "legitimate",
                1: "scam",
                "legitimate": 0,
                "scam": 1
            }

    def predict(self, texts: List[str], return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predict scam probability for texts

        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities

        Returns:
            Prediction results with probabilities and labels
        """
        if not texts:
            return {"predictions": [], "probabilities": [], "labels": []}

        self.logger.debug(f"Classifying {len(texts)} texts with DistilBERT")

        if self.model is not None and self.tokenizer is not None:
            return self._predict_with_model(texts, return_probabilities)
        else:
            return self._predict_with_fallback(texts)

    def _predict_with_model(self, texts: List[str], return_probabilities: bool) -> Dict[str, Any]:
        """Predict using DistilBERT model"""
        try:
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Apply softmax for probabilities
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            # Get predictions
            predictions = np.argmax(probabilities, axis=1)
            labels = [self.label_encoder.get(pred, "unknown") for pred in predictions]

            # Get confidence scores (max probability)
            confidence_scores = np.max(probabilities, axis=1)

            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if return_probabilities else [],
                "labels": labels,
                "confidence_scores": confidence_scores.tolist(),
                "scam_probabilities": probabilities[:, 1].tolist() if len(probabilities[0]) > 1 else probabilities[:, 0].tolist()
            }

        except Exception as e:
            self.logger.error(f"DistilBERT prediction failed: {e}")
            return self._predict_with_fallback(texts)

    def _predict_with_fallback(self, texts: List[str]) -> Dict[str, Any]:
        """Fallback prediction using keyword matching"""
        self.logger.debug("Using fallback classification")

        predictions = []
        probabilities = []
        labels = []
        confidence_scores = []

        for text in texts:
            score = self._calculate_fallback_score(text.lower())

            # Convert score to probability
            probability = min(score, 1.0)  # Cap at 1.0
            scam_probability = probability
            legitimate_probability = 1.0 - probability

            # Determine label
            label = "scam" if probability > 0.5 else "legitimate"
            prediction = 1 if probability > 0.5 else 0

            predictions.append(prediction)
            probabilities.append([legitimate_probability, scam_probability])
            labels.append(label)
            confidence_scores.append(abs(probability - 0.5) * 2)  # Confidence in decision

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "labels": labels,
            "confidence_scores": confidence_scores,
            "scam_probabilities": [p[1] for p in probabilities]
        }

    def _calculate_fallback_score(self, text: str) -> float:
        """Calculate scam score using keyword matching"""
        score = 0.0
        total_weight = 0.0

        for keyword, weight in self.scam_keywords.items():
            if keyword in text:
                score += weight
                total_weight += weight

        # Normalize by total possible weight
        if total_weight > 0:
            score = score / min(total_weight, 2.0)  # Cap normalization

        return min(score, 1.0)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict for a single text"""
        results = self.predict([text])
        return {
            "prediction": results["predictions"][0],
            "probability": results["probabilities"][0] if results["probabilities"] else [],
            "label": results["labels"][0],
            "confidence_score": results["confidence_scores"][0],
            "scam_probability": results["scam_probabilities"][0]
        }

    def get_attention_weights(self, text: str) -> Dict[str, Any]:
        """
        Get attention weights for explainability
        (Simplified version - full implementation would require attention extraction)
        """
        if not self.model or not self.tokenizer:
            return {"error": "Model not available for attention extraction"}

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_seq_length
            )

            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            # For simplicity, return token importance based on position and keywords
            token_scores = {}
            text_lower = text.lower()

            for i, token in enumerate(tokens):
                score = 0.1  # Base score

                # Clean token for matching
                clean_token = token.replace('##', '').lower()

                # Check if token matches scam keywords
                for keyword, weight in self.scam_keywords.items():
                    if keyword in clean_token:
                        score += weight

                # Position bias (CLS and first tokens are important)
                if i == 0:  # CLS token
                    score += 0.2
                elif i < 5:  # Early tokens
                    score += 0.1

                token_scores[token] = min(score, 1.0)

            return {
                "tokens": tokens,
                "attention_weights": token_scores,
                "important_tokens": sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            }

        except Exception as e:
            self.logger.error(f"Attention extraction failed: {e}")
            return {"error": str(e)}

    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for scam patterns using the classifier

        Args:
            text: Input text

        Returns:
            Pattern analysis results
        """
        prediction = self.predict_single(text)
        attention = self.get_attention_weights(text)

        # Extract key phrases (simplified)
        words = text.split()
        suspicious_phrases = []

        scam_indicators = [
            "urgent", "immediate", "transfer money", "send payment",
            "account suspended", "security alert", "official notice",
            "court order", "legal action", "click here", "verify now"
        ]

        for indicator in scam_indicators:
            if indicator in text.lower():
                suspicious_phrases.append(indicator)

        return {
            "prediction": prediction,
            "attention_analysis": attention,
            "suspicious_phrases": suspicious_phrases,
            "text_length": len(text),
            "word_count": len(words)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is not None:
            return {
                "model_type": "DistilBERT",
                "model_path": self.model_path,
                "device": str(self.device),
                "max_seq_length": self.max_seq_length,
                "num_classes": len(self.label_encoder) // 2 if self.label_encoder else 2,
                "has_model": True
            }
        else:
            return {
                "model_type": "DistilBERT",
                "model_path": "fallback",
                "device": "cpu",
                "max_seq_length": self.max_seq_length,
                "num_classes": 2,
                "has_model": False
            }


# Global DistilBERT classifier instance
distilbert_classifier = DistilBERTClassifier()