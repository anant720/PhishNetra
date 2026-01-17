"""
Explainability reasoning engine for PhishNetra
Provides human-readable explanations and highlights
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..core.logging import LoggerMixin


class ReasoningEngine(LoggerMixin):
    """
    Advanced reasoning engine for scam detection explanations
    Provides detailed, human-readable analysis
    """

    def __init__(self):
        self.logger.info("Initializing Reasoning Engine")

        # Scam pattern templates for explanations
        self.pattern_templates = {
            "financial_manipulation": {
                "keywords": ["transfer", "send money", "payment", "bank account", "credit card", "pay now"],
                "explanation": "Requests for financial transactions, especially urgent money transfers",
                "severity": "high"
            },
            "authority_impersonation": {
                "keywords": ["official", "government", "police", "irs", "fbi", "court", "legal", "department"],
                "explanation": "Claims of authority from government or official institutions",
                "severity": "high"
            },
            "urgency_pressure": {
                "keywords": ["urgent", "immediate", "now", "quick", "fast", "deadline", "expires", "limited time"],
                "explanation": "Creates false urgency to pressure quick action without thinking",
                "severity": "medium"
            },
            "phishing_attempts": {
                "keywords": ["click link", "verify account", "confirm identity", "download", "attachment", "login"],
                "explanation": "Attempts to get you to click malicious links or provide credentials",
                "severity": "high"
            },
            "social_engineering": {
                "keywords": ["help needed", "emergency", "family", "friend", "trouble", "support", "assist"],
                "explanation": "Emotional manipulation targeting trust and relationships",
                "severity": "medium"
            },
            "technical_deception": {
                "keywords": ["security alert", "account suspended", "unusual activity", "verify email"],
                "explanation": "Fake security warnings to create panic",
                "severity": "high"
            }
        }

    def generate_explanation(self, text: str, model_outputs: Dict[str, Any], fused_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for scam detection result

        Args:
            text: Original input text
            model_outputs: Raw outputs from all models
            fused_result: Fused prediction result

        Returns:
            Detailed explanation with highlights and reasoning
        """

        self.logger.debug("Generating comprehensive explanation")

        # Extract key insights from model outputs
        highlights = self.extract_highlights(text, model_outputs)
        risk_factors = self.identify_risk_factors(text, model_outputs)
        confidence_analysis = self.analyze_confidence(fused_result)

        # Build narrative explanation
        narrative = self.build_narrative_explanation(
            text, fused_result, highlights, risk_factors, confidence_analysis
        )

        # Generate recommendations
        recommendations = self.generate_recommendations(fused_result["risk_score"])

        return {
            "narrative_explanation": narrative,
            "highlighted_phrases": highlights,
            "risk_factors": risk_factors,
            "confidence_analysis": confidence_analysis,
            "recommendations": recommendations,
            "model_contributions": self.summarize_model_contributions(model_outputs)
        }

    def extract_highlights(self, text: str, model_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and highlight suspicious phrases in the text

        Args:
            text: Input text
            model_outputs: Model outputs containing attention and pattern info

        Returns:
            List of highlighted phrases with explanations
        """

        highlights = []
        text_lower = text.lower()

        # Get attention weights from DistilBERT if available
        attention_highlights = []
        if 'distilbert' in model_outputs:
            bert_data = model_outputs['distilbert']
            attention = bert_data.get('attention_analysis', {})
            important_tokens = attention.get('important_tokens', [])

            for token, score in important_tokens[:5]:  # Top 5
                if score > 0.3:  # Only highlight significant tokens
                    clean_token = token.replace('##', '').lower()
                    if clean_token in text_lower:
                        attention_highlights.append({
                            "phrase": clean_token,
                            "score": score,
                            "reason": "High attention from contextual analysis"
                        })

        # Pattern-based highlights
        for pattern_name, pattern_info in self.pattern_templates.items():
            for keyword in pattern_info["keywords"]:
                if keyword.lower() in text_lower:
                    highlights.append({
                        "phrase": keyword,
                        "category": pattern_name,
                        "severity": pattern_info["severity"],
                        "explanation": pattern_info["explanation"]
                    })

        # Add attention-based highlights
        highlights.extend(attention_highlights)

        # Remove duplicates and sort by severity
        seen_phrases = set()
        unique_highlights = []

        for highlight in highlights:
            phrase = highlight["phrase"].lower()
            if phrase not in seen_phrases:
                seen_phrases.add(phrase)
                unique_highlights.append(highlight)

        # Sort by severity (high -> medium -> low)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        unique_highlights.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

        return unique_highlights[:10]  # Limit to top 10

    def identify_risk_factors(self, text: str, model_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify specific risk factors present in the text

        Args:
            text: Input text
            model_outputs: Model outputs

        Returns:
            List of identified risk factors
        """

        risk_factors = []
        text_lower = text.lower()

        # Analyze from sentence transformer patterns
        if 'sentence_transformer' in model_outputs:
            st_data = model_outputs['sentence_transformer']
            intent_features = st_data.get('intent_features', {})

            risk_mappings = {
                "financial_transfer": "Requests money transfer",
                "account_access": "Requests account access or credentials",
                "high_urgency": "Creates false urgency",
                "authority_claim": "Claims false authority",
                "legal_threat": "Makes legal threats",
                "emotional_manipulation": "Uses emotional manipulation",
                "phishing_indicators": "Contains phishing indicators"
            }

            for feature, score in intent_features.items():
                if score > 0.5:  # Threshold for significant features
                    description = risk_mappings.get(feature, f"Suspicious {feature.replace('_', ' ')}")
                    risk_factors.append({
                        "factor": feature,
                        "description": description,
                        "score": score,
                        "source": "semantic_analysis"
                    })

        # Analyze from similarity engine
        if 'similarity' in model_outputs:
            sim_data = model_outputs['similarity']
            similar_count = sim_data.get('similar_count', 0)
            max_similarity = sim_data.get('max_similarity', 0.0)

            if similar_count > 0:
                risk_factors.append({
                    "factor": "similar_known_scams",
                    "description": f"Similar to {similar_count} known scam patterns",
                    "score": min(max_similarity, 1.0),
                    "source": "similarity_search"
                })

        # Text-based risk factors
        text_factors = self._analyze_text_risk_factors(text)
        risk_factors.extend(text_factors)

        # Sort by score
        risk_factors.sort(key=lambda x: x.get('score', 0), reverse=True)

        return risk_factors

    def _analyze_text_risk_factors(self, text: str) -> List[Dict[str, Any]]:
        """Analyze text for common risk factors"""
        factors = []
        text_lower = text.lower()

        # Check for suspicious URL patterns
        url_pattern = re.compile(r'http[s]?://[^\s]+')
        urls = url_pattern.findall(text)

        if urls:
            factors.append({
                "factor": "contains_urls",
                "description": f"Contains {len(urls)} URL(s) - verify before clicking",
                "score": 0.7,
                "source": "text_analysis"
            })

        # Check for phone numbers
        phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        phones = phone_pattern.findall(text)

        if phones:
            factors.append({
                "factor": "contains_phone_numbers",
                "description": f"Contains {len(phones)} phone number(s) - scammers often provide contact numbers",
                "score": 0.6,
                "source": "text_analysis"
            })

        # Check for money mentions
        money_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|rupees?|INR|euros?|EUR)\b')
        money_mentions = money_pattern.findall(text)

        if money_mentions:
            factors.append({
                "factor": "money_discussion",
                "description": f"Discusses money amounts - common in financial scams",
                "score": 0.5,
                "source": "text_analysis"
            })

        # Check for excessive punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')

        if exclamation_count > 3:
            factors.append({
                "factor": "excessive_exclamation",
                "description": f"Uses {exclamation_count} exclamation marks - creates artificial urgency",
                "score": 0.4,
                "source": "text_analysis"
            })

        if question_count > 5:
            factors.append({
                "factor": "excessive_questions",
                "description": f"Asks {question_count} questions - may be fishing for information",
                "score": 0.3,
                "source": "text_analysis"
            })

        # Check for ALL CAPS words
        caps_words = [word for word in text.split() if word.isupper() and len(word) > 2]
        if len(caps_words) > 2:
            factors.append({
                "factor": "excessive_caps",
                "description": f"Uses {len(caps_words)} ALL CAPS words - creates artificial emphasis",
                "score": 0.3,
                "source": "text_analysis"
            })

        return factors

    def analyze_confidence(self, fused_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze confidence levels and provide interpretation

        Args:
            fused_result: Fused prediction result

        Returns:
            Confidence analysis
        """

        overall_confidence = fused_result.get("confidence", 0.0)
        confidence_breakdown = fused_result.get("model_confidence_breakdown", {})

        # Interpret confidence levels
        if overall_confidence > 0.8:
            confidence_level = "very_high"
            interpretation = "Strong consensus among all models"
        elif overall_confidence > 0.6:
            confidence_level = "high"
            interpretation = "Good agreement between models"
        elif overall_confidence > 0.4:
            confidence_level = "moderate"
            interpretation = "Mixed signals from different models"
        else:
            confidence_level = "low"
            interpretation = "Uncertain prediction, more analysis needed"

        # Identify strongest and weakest models
        if confidence_breakdown:
            strongest_model = max(confidence_breakdown.items(), key=lambda x: x[1])
            weakest_model = min(confidence_breakdown.items(), key=lambda x: x[1])
        else:
            strongest_model = weakest_model = ("unknown", 0.0)

        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "interpretation": interpretation,
            "strongest_model": strongest_model,
            "weakest_model": weakest_model,
            "model_breakdown": confidence_breakdown
        }

    def build_narrative_explanation(self, text: str, fused_result: Dict[str, Any],
                                  highlights: List[Dict[str, Any]], risk_factors: List[Dict[str, Any]],
                                  confidence_analysis: Dict[str, Any]) -> str:
        """
        Build a coherent narrative explanation

        Args:
            text: Original text
            fused_result: Fused results
            highlights: Highlighted phrases
            risk_factors: Identified risk factors
            confidence_analysis: Confidence analysis

        Returns:
            Human-readable narrative
        """

        risk_score = fused_result.get("risk_score", 0)
        threat_category = fused_result.get("threat_category", "unknown")

        # Start with overall assessment
        if risk_score < 30:
            narrative = "This message appears to be legitimate"
        elif risk_score < 60:
            narrative = "This message shows some concerning patterns but is not definitively suspicious"
        else:
            narrative = "This message shows strong indicators of being a scam"

        narrative += f" with a risk score of {risk_score:.1f}/100"

        if threat_category != "legitimate":
            narrative += f" and is categorized as {threat_category.replace('_', ' ')}"

        narrative += ". "

        # Add confidence interpretation
        confidence_level = confidence_analysis.get("confidence_level", "unknown")
        interpretation = confidence_analysis.get("interpretation", "")

        if confidence_level in ["very_high", "high"]:
            narrative += f"The analysis is {confidence_level.replace('_', ' ')} confidence. {interpretation}. "
        else:
            narrative += f"The analysis shows {confidence_level} confidence. {interpretation}. "

        # Highlight key risk factors
        if risk_factors:
            top_factors = risk_factors[:3]  # Top 3 factors
            factor_descriptions = [f["description"] for f in top_factors]
            narrative += f"Key concerns include: {'; '.join(factor_descriptions)}. "

        # Mention suspicious phrases
        if highlights:
            high_severity = [h for h in highlights if h.get("severity") == "high"]
            if high_severity:
                phrases = [f'"{h["phrase"]}"' for h in high_severity[:3]]
                narrative += f"Highly suspicious phrases detected: {', '.join(phrases)}. "

        # Add recommendation based on risk level
        if risk_score > 70:
            narrative += "This appears to be a high-risk message that should not be trusted."
        elif risk_score > 40:
            narrative += "Exercise caution and verify the sender before taking any action."
        else:
            narrative += "This message appears safe, but always be vigilant with unsolicited communications."

        return narrative

    def generate_recommendations(self, risk_score: float) -> List[str]:
        """
        Generate actionable recommendations based on risk score

        Args:
            risk_score: Risk score (0-100)

        Returns:
            List of recommendations
        """

        recommendations = []

        if risk_score > 80:
            recommendations.extend([
                "Do not respond to this message",
                "Do not click any links or provide personal information",
                "Report the message to relevant authorities",
                "Warn contacts who may receive similar messages"
            ])
        elif risk_score > 60:
            recommendations.extend([
                "Verify the sender through official channels",
                "Do not provide sensitive information",
                "Check for similar messages from trusted sources",
                "Consider reporting if it seems suspicious"
            ])
        elif risk_score > 40:
            recommendations.extend([
                "Be cautious with any requests for information",
                "Verify claims independently",
                "Consider the context and sender reputation"
            ])
        else:
            recommendations.extend([
                "Message appears safe",
                "Continue normal caution with unsolicited messages"
            ])

        # Always include general advice
        recommendations.extend([
            "Never share passwords, PINs, or financial information via unsolicited messages",
            "Use official websites and contact numbers for verification",
            "Trust your instincts - if something feels wrong, it probably is"
        ])

        return list(set(recommendations))  # Remove duplicates

    def summarize_model_contributions(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize what each model contributed to the final decision

        Args:
            model_outputs: Raw model outputs

        Returns:
            Model contribution summary
        """

        contributions = {}

        for model_name, outputs in model_outputs.items():
            if model_name == 'fasttext':
                contributions['fasttext'] = {
                    "role": "Text embedding and spelling tolerance",
                    "contribution": "Handles noisy text and multilingual content"
                }
            elif model_name == 'sentence_transformer':
                pattern_scores = outputs.get('pattern_scores', {})
                top_pattern = max(pattern_scores.items(), key=lambda x: x[1]) if pattern_scores else ('none', 0)
                contributions['sentence_transformer'] = {
                    "role": "Semantic understanding and intent detection",
                    "contribution": f"Detected {top_pattern[0].replace('_', ' ')} patterns"
                }
            elif model_name == 'distilbert':
                label = outputs.get('label', 'unknown')
                contributions['distilbert'] = {
                    "role": "Contextual classification",
                    "contribution": f"Classified as {label} with contextual analysis"
                }
            elif model_name == 'similarity':
                similar_count = outputs.get('similar_count', 0)
                contributions['similarity'] = {
                    "role": "Variant detection",
                    "contribution": f"Found {similar_count} similar known scam patterns"
                }

        return contributions


# Global reasoning engine instance
reasoning_engine = ReasoningEngine()