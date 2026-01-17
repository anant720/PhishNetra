"""
Text preprocessing utilities for PhishNetra
Handles multilingual text, emojis, URLs, and normalization
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import logging

from .logging import LoggerMixin


class TextPreprocessor(LoggerMixin):
    """
    Advanced text preprocessor for scam detection
    Handles various text formats and noise
    """

    def __init__(self):
        self.logger.info("Initializing TextPreprocessor")

        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        # Email patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Phone number patterns (basic)
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')

        # Money patterns
        self.money_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|rupees?|INR|euros?|EUR|pounds?|GBP)\b')

        # Emojis and special characters
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002700-\U000027BF"  # dingbats
            "\U0001f926-\U0001f937"  # gestures
            "\U00010000-\U0010ffff"  # other unicode
            "\u2640-\u2642"  # gender symbols
            "\u2600-\u2B55"  # misc symbols
            "\u200d"  # zero width joiner
            "\u23cf"  # eject symbol
            "\u23e9"  # fast forward
            "\u231a"  # watch
            "\ufe0f"  # variation selector
            "\u3030"  # wavy dash
            "]+",
            flags=re.UNICODE
        )

        # Hinglish patterns (common Indian English mixes)
        self.hinglish_patterns = {
            r'\brs\b': 'rupees',
            r'\bpls\b': 'please',
            r'\bok\b': 'okay',
            r'\bthx\b': 'thanks',
            r'\bmsg\b': 'message',
            r'\bplz\b': 'please',
            r'\bfrnd\b': 'friend',
            r'\bacc\b': 'account',
            r'\bamt\b': 'amount',
            r'\btxn\b': 'transaction',
            r'\bpls\b': 'please',
            r'\bhw\b': 'how',
            r'\bwht\b': 'what',
            r'\bwhr\b': 'where',
            r'\bwen\b': 'when',
            r'\bfr\b': 'for',
            r'\bnd\b': 'and',
            r'\btht\b': 'that',
            r'\bths\b': 'this',
            r'\bwth\b': 'with',
            r'\byr\b': 'your'
        }

    def preprocess_text(self, text: str, preserve_metadata: bool = True) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing

        Args:
            text: Input text to preprocess
            preserve_metadata: Whether to preserve extracted metadata

        Returns:
            Dictionary with processed text and metadata
        """

        if not text or not isinstance(text, str):
            return {
                "processed_text": "",
                "original_text": text or "",
                "metadata": {}
            }

        original_text = text
        metadata = {}

        # Extract metadata if requested
        if preserve_metadata:
            metadata = self._extract_metadata(text)

        # Apply preprocessing steps
        text = self._normalize_unicode(text)
        text = self._handle_urls(text)
        text = self._handle_emails(text)
        text = self._handle_phone_numbers(text)
        text = self._handle_money(text)
        text = self._normalize_hinglish(text)
        text = self._clean_text(text)
        text = self._normalize_whitespace(text)

        self.logger.debug(f"Preprocessed text: '{original_text[:50]}...' -> '{text[:50]}...'")

        return {
            "processed_text": text,
            "original_text": original_text,
            "metadata": metadata
        }

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize unicode (NFKC - compatibility decomposition followed by canonical composition)
        text = unicodedata.normalize('NFKC', text)
        return text

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text"""
        metadata = {
            "urls": self.url_pattern.findall(text),
            "emails": self.email_pattern.findall(text),
            "phones": self.phone_pattern.findall(text),
            "money_mentions": self.money_pattern.findall(text),
            "has_emojis": bool(self.emoji_pattern.search(text)),
            "text_length": len(text),
            "word_count": len(text.split())
        }

        # URL analysis
        if metadata["urls"]:
            domains = []
            for url in metadata["urls"]:
                try:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        domains.append(parsed.netloc)
                except:
                    continue
            metadata["url_domains"] = domains

        return metadata

    def _handle_urls(self, text: str) -> str:
        """Replace URLs with placeholders"""
        return self.url_pattern.sub("[URL]", text)

    def _handle_emails(self, text: str) -> str:
        """Replace emails with placeholders"""
        return self.email_pattern.sub("[EMAIL]", text)

    def _handle_phone_numbers(self, text: str) -> str:
        """Replace phone numbers with placeholders"""
        return self.phone_pattern.sub("[PHONE]", text)

    def _handle_money(self, text: str) -> str:
        """Replace money mentions with placeholders"""
        return self.money_pattern.sub("[MONEY]", text)

    def _normalize_hinglish(self, text: str) -> str:
        """Normalize common Hinglish abbreviations"""
        text = text.lower()

        for pattern, replacement in self.hinglish_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive punctuation and normalizing"""
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        return ' '.join(text.split())

    def batch_preprocess(self, texts: List[str], preserve_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Preprocess multiple texts

        Args:
            texts: List of texts to preprocess
            preserve_metadata: Whether to preserve metadata

        Returns:
            List of preprocessing results
        """
        self.logger.info(f"Batch preprocessing {len(texts)} texts")
        return [self.preprocess_text(text, preserve_metadata) for text in texts]

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive text statistics"""
        stats = {
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            "punctuation_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
            "has_caps_lock": any(word.isupper() and len(word) > 2 for word in text.split()),
            "has_exclamation": '!' in text,
            "has_question": '?' in text,
            "has_ellipsis": '...' in text
        }

        return stats


# Global preprocessor instance
preprocessor = TextPreprocessor()