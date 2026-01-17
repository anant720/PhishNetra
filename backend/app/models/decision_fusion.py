"""
Decision fusion system for PhishNetra
Combines multiple model outputs with confidence weighting
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from ..core.logging import LoggerMixin
from ..core.config import settings


class ThreatCategory(Enum):
    """Complete scam taxonomy - supports multi-label classification"""
    LEGITIMATE = "legitimate"
    FINANCIAL_MANIPULATION = "financial_manipulation"
    PHISHING_REDIRECTION = "phishing_redirection"
    IDENTITY_VERIFICATION_ABUSE = "identity_verification_abuse"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    SOCIAL_ENGINEERING = "social_engineering"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    JOB_WORK_INVESTMENT_SCAM = "job_work_investment_scam"
    TECH_SUPPORT_MALWARE_SCAM = "tech_support_malware_scam"
    DELIVERY_COURIER_SCAM = "delivery_courier_scam"
    LOTTERY_REWARD_SCAM = "lottery_reward_scam"
    HYBRID_EVOLVING_SCAM = "hybrid_evolving_scam"
    UNKNOWN_SCAM = "unknown_scam"


class DecisionFusion(LoggerMixin):
    """
    Advanced decision fusion system
    Combines model outputs using confidence weighting and ensemble methods
    """

    def __init__(self):
        self.weights = {
            'fasttext': settings.fasttext_weight,
            'sentence_transformer': settings.sentence_transformer_weight,
            'distilbert': settings.distilbert_weight,
            'similarity': settings.similarity_weight
        }

        self.logger.info("Initializing Decision Fusion system")
        self.logger.info(f"Model weights: {self.weights}")

    def fuse_predictions(self, model_outputs: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        """
        Fuse predictions from multiple models, incorporating URL analysis

        Args:
            model_outputs: Dictionary containing outputs from all models

        Returns:
            Fused prediction results
        """
        self.logger.debug("Fusing predictions from multiple models")

        # Extract individual model predictions
        predictions = self._extract_predictions(model_outputs)

        if not predictions:
            return self._create_default_response()

        # Calculate weighted risk score (pass model_outputs for URL integration)
        risk_score, confidence = self._calculate_weighted_score(predictions, model_outputs)

        # Determine primary threat category (pass model_outputs for URL analysis)
        threat_category = self._determine_threat_category(predictions, risk_score, model_outputs, text)

        # Multi-label classification: determine all applicable categories
        threat_categories = self._determine_multi_label_categories(predictions, risk_score, threat_category, model_outputs, text)

        # Calculate model confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(predictions)

        # Generate reasoning (include URL analysis in reasoning)
        reasoning = self._generate_reasoning(predictions, risk_score, threat_category, model_outputs, text)

        return {
            "risk_score": risk_score,
            "confidence": confidence,
            "threat_category": threat_category.value,  # Primary category
            "threat_categories": threat_categories,  # All applicable categories with confidence
            "model_confidence_breakdown": confidence_breakdown,
            "reasoning": reasoning,
            "raw_predictions": predictions
        }

    def _extract_predictions(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract predictions from model outputs"""
        predictions = {}

        # FastText predictions (embedding-based similarity)
        if 'fasttext' in model_outputs:
            fasttext_data = model_outputs['fasttext']
            predictions['fasttext'] = {
                'risk_score': fasttext_data.get('similarity_score', 0.0),
                'confidence': fasttext_data.get('confidence', 0.5),
                'patterns': fasttext_data.get('patterns', [])
            }

        # Sentence Transformer predictions
        if 'sentence_transformer' in model_outputs:
            st_data = model_outputs['sentence_transformer']
            # Extract dominant pattern score
            pattern_scores = st_data.get('pattern_scores', {})
            dominant_pattern = st_data.get('dominant_pattern', ('unknown', 0.0))
            
            # Use dominant pattern score as risk score
            if isinstance(dominant_pattern, tuple) and len(dominant_pattern) >= 2:
                dominant_score = float(dominant_pattern[1])
            elif pattern_scores:
                dominant_score = float(max(pattern_scores.values()))
            else:
                dominant_score = 0.0

            # Get dynamic confidence
            confidence = float(st_data.get('embedding_confidence', 0.7))

            predictions['sentence_transformer'] = {
                'risk_score': dominant_score,
                'confidence': confidence,
                'intent_features': st_data.get('intent_features', {}),
                'dominant_pattern': dominant_pattern
            }

        # DistilBERT predictions
        if 'distilbert' in model_outputs:
            bert_data = model_outputs['distilbert']
            
            # Extract scam_probability - could be directly in bert_data or in prediction dict
            scam_prob = bert_data.get('scam_probability', 0.0)
            if scam_prob == 0.0 and 'prediction' in bert_data:
                pred_dict = bert_data['prediction']
                if isinstance(pred_dict, dict):
                    scam_prob = pred_dict.get('scam_probability', 0.0)
            
            # Extract confidence_score
            conf_score = bert_data.get('confidence_score', 0.5)
            if conf_score == 0.5 and 'prediction' in bert_data:
                pred_dict = bert_data['prediction']
                if isinstance(pred_dict, dict):
                    conf_score = pred_dict.get('confidence_score', 0.5)
            
            # Extract label
            label = bert_data.get('label', 'unknown')
            if label == 'unknown' and 'prediction' in bert_data:
                pred_dict = bert_data['prediction']
                if isinstance(pred_dict, dict):
                    label = pred_dict.get('label', 'unknown')
            
            predictions['distilbert'] = {
                'risk_score': float(scam_prob),
                'confidence': float(conf_score),
                'label': label,
                'attention_weights': bert_data.get('attention_analysis', {})
            }

        # Similarity engine predictions
        if 'similarity' in model_outputs:
            sim_data = model_outputs['similarity']
            predictions['similarity'] = {
                'risk_score': sim_data.get('variant_score', 0.0),
                'confidence': min(sim_data.get('max_similarity', 0.0) + 0.2, 1.0),  # Add base confidence
                'similar_count': sim_data.get('similar_count', 0),
                'max_similarity': sim_data.get('max_similarity', 0.0)
            }

        return predictions

    def _calculate_weighted_score(self, predictions: Dict[str, Any], model_outputs: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Calculate weighted risk score and overall confidence, incorporating URL analysis"""
        if not predictions:
            return 0.0, 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0

        for model_name, pred_data in predictions.items():
            weight = self.weights.get(model_name, 0.25)
            risk_score = pred_data.get('risk_score', 0.0)
            confidence = pred_data.get('confidence', 0.5)

            weighted_sum += risk_score * weight * confidence
            total_weight += weight * confidence
            confidence_sum += confidence

        # Calculate base score
        if total_weight > 0:
            risk_score = weighted_sum / total_weight
            avg_confidence = confidence_sum / len(predictions)
        else:
            risk_score = 0.0
            avg_confidence = 0.0

        # Incorporate URL analysis into risk score
        if model_outputs and 'url_analysis' in model_outputs:
            url_data = model_outputs['url_analysis']
            if url_data.get('has_urls', False):
                max_url_risk = url_data.get('max_risk_score', 0.0)
                url_count = url_data.get('url_count', 0)
                
                # URL risk contributes to overall score (weighted)
                # High URL risk significantly increases overall risk
                if max_url_risk > 60:
                    # High-risk URL: add significant weight
                    url_contribution = (max_url_risk / 100.0) * 0.25  # Up to 25% contribution
                    risk_score = risk_score * 0.75 + (risk_score + url_contribution * 100) * 0.25
                elif max_url_risk > 40:
                    # Moderate URL risk
                    url_contribution = (max_url_risk / 100.0) * 0.15  # Up to 15% contribution
                    risk_score = risk_score * 0.85 + (risk_score + url_contribution * 100) * 0.15
                else:
                    # Low URL risk but presence still matters
                    url_contribution = (max_url_risk / 100.0) * 0.05  # Up to 5% contribution
                    risk_score = risk_score * 0.95 + (risk_score + url_contribution * 100) * 0.05
                
                # Multiple URLs increase suspicion
                if url_count > 1:
                    risk_score = min(risk_score + (url_count - 1) * 5, 100)

        # Normalize risk score to 0-100
        risk_score = min(max(risk_score * 100, 0), 100)

        return risk_score, avg_confidence

    def _determine_threat_category(self, predictions: Dict[str, Any], risk_score: float, model_outputs: Optional[Dict[str, Any]] = None, text: str = "") -> ThreatCategory:
        """Determine primary threat category based on intent patterns"""

        text_lower = text.lower()

        # Low risk threshold
        if risk_score < 30:
            return ThreatCategory.LEGITIMATE

        # Analyze patterns from different models using INTENT-BASED reasoning
        category_scores = {
            ThreatCategory.FINANCIAL_MANIPULATION: 0,
            ThreatCategory.PHISHING_REDIRECTION: 0,
            ThreatCategory.IDENTITY_VERIFICATION_ABUSE: 0,
            ThreatCategory.AUTHORITY_IMPERSONATION: 0,
            ThreatCategory.SOCIAL_ENGINEERING: 0,
            ThreatCategory.EMOTIONAL_MANIPULATION: 0,
            ThreatCategory.JOB_WORK_INVESTMENT_SCAM: 0,
            ThreatCategory.TECH_SUPPORT_MALWARE_SCAM: 0,
            ThreatCategory.DELIVERY_COURIER_SCAM: 0,
            ThreatCategory.LOTTERY_REWARD_SCAM: 0,
            ThreatCategory.HYBRID_EVOLVING_SCAM: 0
        }

        # Enhanced Sentence Transformer patterns - INTENT-BASED ANALYSIS
        if 'sentence_transformer' in predictions:
            st_data = predictions['sentence_transformer']
            intent_features = st_data.get('intent_features', {})
            pattern_scores = st_data.get('pattern_scores', {})

            # Financial Manipulation: Intent to influence money transfer
            financial_intent = (
                intent_features.get('financial_transfer', 0) * 2.5 +
                intent_features.get('account_access', 0) * 1.8 +
                pattern_scores.get('financial_manipulation', 0) * 3.0 +
                (1.0 if any(kw in text_lower for kw in ['transfer', 'money', 'payment', 'bank', 'account']) else 0.0) * 1.5
            )
            category_scores[ThreatCategory.FINANCIAL_MANIPULATION] += financial_intent

            # Phishing & Redirection: Intent to redirect/harvest credentials
            phishing_intent = (
                intent_features.get('phishing_indicators', 0) * 2.5 +
                intent_features.get('url_presence', 0) * 2.0 +
                pattern_scores.get('technical_deception', 0) * 2.0 +
                (1.0 if any(kw in text_lower for kw in ['click', 'link', 'visit', 'verify', 'login']) else 0.0) * 1.5
            )
            category_scores[ThreatCategory.PHISHING_REDIRECTION] += phishing_intent

            # Identity Verification Abuse: Intent to extract personal info
            identity_intent = (
                intent_features.get('personal_data_request', 0) * 2.5 +
                intent_features.get('account_access', 0) * 1.5 +
                (1.0 if any(kw in text_lower for kw in ['verify', 'confirm', 'information', 'details']) else 0.0) * 1.5
            )
            category_scores[ThreatCategory.IDENTITY_VERIFICATION_ABUSE] += identity_intent

            # Authority Impersonation: Intent to pretend to be institution
            authority_intent = (
                intent_features.get('authority_claim', 0) * 3.0 +
                intent_features.get('legal_threat', 0) * 2.0 +
                pattern_scores.get('authority_deception', 0) * 2.5 +
                (1.0 if any(kw in text_lower for kw in ['official', 'government', 'police', 'irs', 'fbi', 'court']) else 0.0) * 2.0
            )
            category_scores[ThreatCategory.AUTHORITY_IMPERSONATION] += authority_intent

            # Social Engineering: Intent to manipulate trust
            social_intent = (
                intent_features.get('trust_building', 0) * 2.0 +
                intent_features.get('polite_tone', 0) * 1.0 +
                pattern_scores.get('social_engineering', 0) * 2.0 +
                (1.0 if any(kw in text_lower for kw in ['help', 'support', 'assist', 'please']) else 0.0) * 1.2
            )
            category_scores[ThreatCategory.SOCIAL_ENGINEERING] += social_intent

            # Emotional Manipulation: Intent to trigger fear/urgency/sympathy
            emotional_intent = (
                intent_features.get('emotional_manipulation', 0) * 2.5 +
                intent_features.get('high_urgency', 0) * 2.0 +
                intent_features.get('time_pressure', 0) * 2.0 +
                intent_features.get('fear_appeal', 0) * 2.0 +
                pattern_scores.get('urgency_pressure', 0) * 2.5 +
                (1.0 if any(kw in text_lower for kw in ['urgent', 'immediate', 'emergency', 'family', 'suspended']) else 0.0) * 2.0
            )
            category_scores[ThreatCategory.EMOTIONAL_MANIPULATION] += emotional_intent

            # Job/Work/Investment Scam
            job_investment_intent = (
                intent_features.get('job_opportunity', 0) * 2.5 +
                intent_features.get('investment_promise', 0) * 2.5 +
                intent_features.get('registration_fee', 0) * 3.0 +
                (1.0 if any(kw in text_lower for kw in ['job', 'work', 'investment', 'profit', 'earn', 'bonus']) else 0.0) * 2.0
            )
            category_scores[ThreatCategory.JOB_WORK_INVESTMENT_SCAM] += job_investment_intent

            # Tech Support/Malware
            tech_support_intent = (
                intent_features.get('tech_support', 0) * 2.5 +
                intent_features.get('download_request', 0) * 2.5 +
                (1.0 if any(kw in text_lower for kw in ['virus', 'malware', 'infected', 'support', 'call']) else 0.0) * 2.0
            )
            category_scores[ThreatCategory.TECH_SUPPORT_MALWARE_SCAM] += tech_support_intent

            # Delivery/Courier
            delivery_intent = (
                intent_features.get('delivery_mention', 0) * 2.5 +
                intent_features.get('delivery_action', 0) * 3.0 +
                (1.0 if any(kw in text_lower for kw in ['delivery', 'package', 'courier', 'tracking']) else 0.0) * 2.0
            )
            category_scores[ThreatCategory.DELIVERY_COURIER_SCAM] += delivery_intent

            # Lottery/Reward
            lottery_intent = (
                intent_features.get('lottery_mention', 0) * 3.0 +
                intent_features.get('claim_action', 0) * 2.5 +
                (1.0 if any(kw in text_lower for kw in ['winner', 'prize', 'claim', 'congratulations']) else 0.0) * 2.5
            )
            category_scores[ThreatCategory.LOTTERY_REWARD_SCAM] += lottery_intent

        # DistilBERT patterns (from attention analysis)
        if 'distilbert' in predictions:
            bert_data = predictions['distilbert']
            attention = bert_data.get('attention_analysis', {})
            important_tokens = attention.get('important_tokens', [])
            suspicious_phrases = bert_data.get('suspicious_phrases', [])

            # Intent-based token analysis
            for token, score in important_tokens:
                token_lower = token.lower().replace('##', '')
                
                # Financial manipulation intent
                if any(word in token_lower for word in ['transfer', 'money', 'pay', 'refund', 'debit', 'credit']):
                    category_scores[ThreatCategory.FINANCIAL_MANIPULATION] += score * 1.2
                
                # Phishing intent
                if any(word in token_lower for word in ['click', 'link', 'url', 'verify', 'login', 'password']):
                    category_scores[ThreatCategory.PHISHING_REDIRECTION] += score * 1.3
                
                # Authority impersonation intent
                if any(word in token_lower for word in ['official', 'government', 'police', 'irs', 'fbi', 'court']):
                    category_scores[ThreatCategory.AUTHORITY_IMPERSONATION] += score * 1.5
                
                # Emotional manipulation intent
                if any(word in token_lower for word in ['urgent', 'immediate', 'emergency', 'family', 'help']):
                    category_scores[ThreatCategory.EMOTIONAL_MANIPULATION] += score * 1.2
                
                # Job/Investment scam intent
                if any(word in token_lower for word in ['job', 'work', 'investment', 'profit', 'earn', 'task']):
                    category_scores[ThreatCategory.JOB_WORK_INVESTMENT_SCAM] += score * 1.3
                
                # Tech support intent
                if any(word in token_lower for word in ['support', 'virus', 'malware', 'computer', 'device']):
                    category_scores[ThreatCategory.TECH_SUPPORT_MALWARE_SCAM] += score * 1.2
                
                # Delivery/Courier intent
                if any(word in token_lower for word in ['delivery', 'package', 'courier', 'shipment', 'tracking']):
                    category_scores[ThreatCategory.DELIVERY_COURIER_SCAM] += score * 1.3
                
                # Lottery/Reward intent
                if any(word in token_lower for word in ['lottery', 'winner', 'prize', 'reward', 'claim', 'congratulations']):
                    category_scores[ThreatCategory.LOTTERY_REWARD_SCAM] += score * 1.4

        # URL-based categorization (if URL analysis is present)
        if model_outputs and 'url_analysis' in model_outputs:
            url_data = model_outputs['url_analysis']
            max_url_risk = url_data.get('max_risk_score', 0)
            urls = url_data.get('urls', [])
            
            if max_url_risk > 50:
                # High URL risk suggests phishing or credential harvesting
                category_scores[ThreatCategory.PHISHING_REDIRECTION] += max_url_risk * 0.3
                
                # Check all URLs for specific signals
                for url_result in urls:
                    signals = url_result.get('signals', [])
                    if 'credential_form_detected' in signals:
                        category_scores[ThreatCategory.IDENTITY_VERIFICATION_ABUSE] += 2.0
                    if 'brand_impersonation' in signals:
                        category_scores[ThreatCategory.AUTHORITY_IMPERSONATION] += 1.5
                    if 'shortened_url' in signals:
                        category_scores[ThreatCategory.PHISHING_REDIRECTION] += 1.0
                    if 'redirection_anomaly' in signals:
                        category_scores[ThreatCategory.PHISHING_REDIRECTION] += 1.5

        # Similarity-based categorization
        if 'similarity' in predictions:
            sim_data = predictions['similarity']
            similar_count = sim_data.get('similar_count', 0)
            variant_score = sim_data.get('variant_score', 0)

            # High similarity suggests known scam patterns - boost all relevant categories
            if similar_count > 2 and variant_score > 0.5:
                # Boost categories proportionally
                for category in category_scores:
                    if category_scores[category] > 0:
                        category_scores[category] += variant_score * 0.5

        # Check for hybrid/evolving scams (multiple strong signals)
        active_categories = sum(1 for score in category_scores.values() if score > 1.0)
        if active_categories >= 3:
            # Multiple strong intents detected - likely hybrid scam
            category_scores[ThreatCategory.HYBRID_EVOLVING_SCAM] += active_categories * 0.5

        # Select primary category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])

        # If no clear category and risk score is high, mark as unknown scam
        if best_category[1] == 0 and risk_score > 50:
            return ThreatCategory.UNKNOWN_SCAM

        return best_category[0]

    def _determine_multi_label_categories(self, predictions: Dict[str, Any], risk_score: float, primary_category: ThreatCategory, model_outputs: Optional[Dict[str, Any]] = None, text: str = "") -> List[Dict[str, Any]]:
        """
        Determine all applicable threat categories (multi-label classification)
        Returns list of categories with confidence scores
        """
        text_lower = text.lower()

        if risk_score < 30:
            return [{"category": ThreatCategory.LEGITIMATE.value, "confidence": 1.0 - (risk_score / 30)}]

        # Recalculate all category scores (same logic as _determine_threat_category)
        category_scores = {
            ThreatCategory.FINANCIAL_MANIPULATION: 0,
            ThreatCategory.PHISHING_REDIRECTION: 0,
            ThreatCategory.IDENTITY_VERIFICATION_ABUSE: 0,
            ThreatCategory.AUTHORITY_IMPERSONATION: 0,
            ThreatCategory.SOCIAL_ENGINEERING: 0,
            ThreatCategory.EMOTIONAL_MANIPULATION: 0,
            ThreatCategory.JOB_WORK_INVESTMENT_SCAM: 0,
            ThreatCategory.TECH_SUPPORT_MALWARE_SCAM: 0,
            ThreatCategory.DELIVERY_COURIER_SCAM: 0,
            ThreatCategory.LOTTERY_REWARD_SCAM: 0,
            ThreatCategory.HYBRID_EVOLVING_SCAM: 0
        }

        # Apply same intent-based analysis as primary category
        if 'sentence_transformer' in predictions:
            st_data = predictions['sentence_transformer']
            intent_features = st_data.get('intent_features', {})
            pattern_scores = st_data.get('pattern_scores', {})

            category_scores[ThreatCategory.FINANCIAL_MANIPULATION] += (
                intent_features.get('financial_transfer', 0) * 1.5 +
                intent_features.get('account_access', 0) * 1.2 +
                pattern_scores.get('financial_manipulation', 0) * 2.0
            )
            category_scores[ThreatCategory.PHISHING_REDIRECTION] += (
                intent_features.get('phishing_indicators', 0) * 2.0 +
                pattern_scores.get('technical_deception', 0) * 1.5
            )
            category_scores[ThreatCategory.IDENTITY_VERIFICATION_ABUSE] += (
                intent_features.get('personal_data_request', 0) * 2.0 +
                intent_features.get('account_access', 0) * 1.0
            )
            category_scores[ThreatCategory.AUTHORITY_IMPERSONATION] += (
                intent_features.get('authority_claim', 0) * 2.0 +
                intent_features.get('legal_threat', 0) * 1.5 +
                pattern_scores.get('authority_deception', 0) * 2.0
            )
            category_scores[ThreatCategory.SOCIAL_ENGINEERING] += (
                intent_features.get('trust_building', 0) * 1.5 +
                pattern_scores.get('social_engineering', 0) * 2.0
            )
            category_scores[ThreatCategory.EMOTIONAL_MANIPULATION] += (
                intent_features.get('emotional_manipulation', 0) * 2.0 +
                intent_features.get('high_urgency', 0) * 1.5 +
                intent_features.get('time_pressure', 0) * 1.5 +
                pattern_scores.get('urgency_pressure', 0) * 2.0
            )

        if 'distilbert' in predictions:
            bert_data = predictions['distilbert']
            attention = bert_data.get('attention_analysis', {})
            important_tokens = attention.get('important_tokens', [])

            for token, score in important_tokens:
                token_lower = token.lower().replace('##', '')
                if any(word in token_lower for word in ['transfer', 'money', 'pay', 'refund']):
                    category_scores[ThreatCategory.FINANCIAL_MANIPULATION] += score * 1.2
                if any(word in token_lower for word in ['click', 'link', 'verify', 'login']):
                    category_scores[ThreatCategory.PHISHING_REDIRECTION] += score * 1.3
                if any(word in token_lower for word in ['official', 'government', 'police']):
                    category_scores[ThreatCategory.AUTHORITY_IMPERSONATION] += score * 1.5
                if any(word in token_lower for word in ['urgent', 'emergency', 'family']):
                    category_scores[ThreatCategory.EMOTIONAL_MANIPULATION] += score * 1.2
                if any(word in token_lower for word in ['job', 'investment', 'profit']):
                    category_scores[ThreatCategory.JOB_WORK_INVESTMENT_SCAM] += score * 1.3
                if any(word in token_lower for word in ['support', 'virus', 'malware']):
                    category_scores[ThreatCategory.TECH_SUPPORT_MALWARE_SCAM] += score * 1.2
                if any(word in token_lower for word in ['delivery', 'package', 'courier']):
                    category_scores[ThreatCategory.DELIVERY_COURIER_SCAM] += score * 1.3
                if any(word in token_lower for word in ['lottery', 'winner', 'prize']):
                    category_scores[ThreatCategory.LOTTERY_REWARD_SCAM] += score * 1.4

        # URL-based categorization (if URL analysis is present in model_outputs)
        if model_outputs and 'url_analysis' in model_outputs:
            url_data = model_outputs['url_analysis']
            max_url_risk = url_data.get('max_risk_score', 0)
            urls = url_data.get('urls', [])
            
            if max_url_risk > 50:
                category_scores[ThreatCategory.PHISHING_REDIRECTION] += max_url_risk * 0.3
                for url_result in urls:
                    signals = url_result.get('signals', [])
                    if 'credential_form_detected' in signals:
                        category_scores[ThreatCategory.IDENTITY_VERIFICATION_ABUSE] += 2.0
                    if 'brand_impersonation' in signals:
                        category_scores[ThreatCategory.AUTHORITY_IMPERSONATION] += 1.5
                    if 'shortened_url' in signals:
                        category_scores[ThreatCategory.PHISHING_REDIRECTION] += 1.0

        # Check for hybrid scams
        active_categories = sum(1 for score in category_scores.values() if score > 1.0)
        if active_categories >= 3:
            category_scores[ThreatCategory.HYBRID_EVOLVING_SCAM] += active_categories * 0.5

        # Normalize scores to 0-1 confidence range
        max_score = max(category_scores.values()) if category_scores.values() else 1.0
        if max_score == 0:
            max_score = 1.0

        # Return all categories with score > threshold (0.3)
        threshold = 0.3
        multi_label = []
        for category, score in category_scores.items():
            normalized_confidence = min(score / max_score, 1.0)
            if normalized_confidence >= threshold:
                multi_label.append({
                    "category": category.value,
                    "confidence": round(normalized_confidence, 3)
                })

        # Sort by confidence (highest first)
        multi_label.sort(key=lambda x: x["confidence"], reverse=True)

        # Always include primary category if not already present
        primary_dict = {"category": primary_category.value, "confidence": 1.0}
        if not any(c["category"] == primary_category.value for c in multi_label):
            multi_label.insert(0, primary_dict)
        else:
            # Update primary category confidence to 1.0
            for item in multi_label:
                if item["category"] == primary_category.value:
                    item["confidence"] = 1.0
                    break
            # Re-sort
            multi_label.sort(key=lambda x: x["confidence"], reverse=True)

        return multi_label

    def _calculate_confidence_breakdown(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence breakdown for each model"""
        breakdown = {}

        for model_name, pred_data in predictions.items():
            confidence = pred_data.get('confidence', 0.0)
            risk_score = pred_data.get('risk_score', 0.0)

            # Adjust confidence based on risk score consistency
            # High confidence in high-risk predictions, low confidence in borderline cases
            if risk_score > 0.7:
                adjusted_confidence = min(confidence + 0.2, 1.0)
            elif risk_score < 0.3:
                adjusted_confidence = min(confidence + 0.1, 1.0)
            else:
                # Borderline cases have lower confidence
                adjusted_confidence = confidence * 0.8

            breakdown[model_name] = round(adjusted_confidence, 3)

        return breakdown

    def _generate_reasoning(self, predictions: Dict[str, Any], risk_score: float, threat_category: ThreatCategory, model_outputs: Optional[Dict[str, Any]] = None, text: str = "") -> str:
        """Generate human-readable reasoning for the prediction, including URL analysis"""
        reasons = []

        # Overall risk assessment
        if risk_score < 30:
            reasons.append("Low overall risk score indicates legitimate content.")
        elif risk_score < 60:
            reasons.append("Moderate risk score suggests potential concerns but not definitive.")
        else:
            reasons.append("High risk score indicates strong scam indicators.")

        # URL analysis reasoning
        if model_outputs and 'url_analysis' in model_outputs:
            url_data = model_outputs['url_analysis']
            if url_data.get('has_urls', False):
                url_count = url_data.get('url_count', 0)
                max_url_risk = url_data.get('max_risk_score', 0.0)
                urls = url_data.get('urls', [])
                
                if url_count > 0:
                    reasons.append(f"Message contains {url_count} URL(s).")
                    
                    if max_url_risk > 60:
                        reasons.append("URL analysis indicates high risk.")
                        # Add specific URL signals
                        for url_result in urls:
                            if url_result.get('risk_score', 0) > 60:
                                signals = url_result.get('signals', [])
                                if signals:
                                    reasons.append(f"URL shows: {', '.join(signals[:3])}.")
                    elif max_url_risk > 40:
                        reasons.append("URL analysis indicates suspicious characteristics.")
                    else:
                        reasons.append("URL analysis shows low risk indicators.")

        # Model-specific reasoning
        if 'sentence_transformer' in predictions:
            st_data = predictions['sentence_transformer']
            dominant_pattern = st_data.get('dominant_pattern', ('unknown', 0.0))
            if isinstance(dominant_pattern, tuple) and len(dominant_pattern) >= 2 and dominant_pattern[1] > 0.5:
                reasons.append(f"Semantic analysis detected {dominant_pattern[0].replace('_', ' ')} patterns.")

        if 'distilbert' in predictions:
            bert_data = predictions['distilbert']
            label = bert_data.get('label', 'unknown')
            if label == 'scam':
                reasons.append("Contextual analysis classified content as suspicious.")

        if 'similarity' in predictions:
            sim_data = predictions['similarity']
            similar_count = sim_data.get('similar_count', 0)
            if similar_count > 0:
                reasons.append(f"Found {similar_count} similar known scam patterns.")

        # Category-specific reasoning
        category_reasoning = {
            ThreatCategory.FINANCIAL_MANIPULATION: "Content appears to involve financial transactions or account access.",
            ThreatCategory.PHISHING_REDIRECTION: "Content contains links or requests for verification that may be malicious.",
            ThreatCategory.IDENTITY_VERIFICATION_ABUSE: "Content attempts to extract personal information or credentials.",
            ThreatCategory.AUTHORITY_IMPERSONATION: "Content appears to impersonate trusted entities or individuals.",
            ThreatCategory.EMOTIONAL_MANIPULATION: "Content creates false urgency or emotional pressure to act quickly.",
            ThreatCategory.SOCIAL_ENGINEERING: "Content attempts to manipulate emotions or build false trust.",
            ThreatCategory.TECH_SUPPORT_MALWARE_SCAM: "Content appears to be technical support scam.",
            ThreatCategory.JOB_WORK_INVESTMENT_SCAM: "Content appears to be job, work, or investment scam.",
            ThreatCategory.DELIVERY_COURIER_SCAM: "Content appears to be delivery or courier scam.",
            ThreatCategory.LOTTERY_REWARD_SCAM: "Content appears to be lottery or reward scam.",
            ThreatCategory.HYBRID_EVOLVING_SCAM: "Content shows multiple scam patterns (hybrid/evolving scam).",
            ThreatCategory.UNKNOWN_SCAM: "Content shows scam patterns but doesn't match known categories.",
            ThreatCategory.LEGITIMATE: "Content appears to be legitimate communication."
        }

        reasons.append(category_reasoning.get(threat_category, "Unable to categorize threat type."))

        return " ".join(reasons)

    def _create_default_response(self) -> Dict[str, Any]:
        """Create default response when no predictions available"""
        return {
            "risk_score": 0.0,
            "confidence": 0.0,
            "threat_category": ThreatCategory.LEGITIMATE.value,
            "model_confidence_breakdown": {},
            "reasoning": "No model predictions available. Content appears legitimate.",
            "raw_predictions": {}
        }

    def get_fusion_info(self) -> Dict[str, Any]:
        """Get fusion system information"""
        return {
            "fusion_method": "weighted_ensemble",
            "model_weights": self.weights,
            "supported_categories": [cat.value for cat in ThreatCategory],
            "confidence_calculation": "weighted_average_with_adjustments"
        }


# Global decision fusion instance
decision_fusion = DecisionFusion()