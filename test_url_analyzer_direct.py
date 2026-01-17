#!/usr/bin/env python3
"""
Test URL analyzer directly
"""

import sys
import os

# Minimal setup
class LoggerMixin:
    def __init__(self):
        pass

    def info(self, msg):
        print(f"INFO: {msg}")

    def warning(self, msg):
        print(f"WARNING: {msg}")

    def error(self, msg):
        print(f"ERROR: {msg}")

    def debug(self, msg):
        print(f"DEBUG: {msg}")

# Simple URL analyzer class
class URLAnalyzer(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.logger.info("Initializing URL Analyzer")

        # Simple known domains
        self.known_legitimate_domains = {
            "google.com", "microsoft.com", "amazon.com", "apple.com", "facebook.com"
        }

    def extract_urls(self, text: str):
        """Simple URL extraction"""
        import re

        urls = []

        # Basic URL pattern
        url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)
        urls.extend(url_pattern.findall(text))

        # Domain mentions
        domain_pattern = re.compile(r'\b(?:visit|go to|check)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE)
        mentions = domain_pattern.findall(text)
        for domain in mentions:
            if domain not in [url.split('://')[1] for url in urls if '://' in url]:
                urls.append(f"https://{domain}")

        return urls

# Test the analyzer
def test_analyzer():
    analyzer = URLAnalyzer()

    test_cases = [
        "Check this link: https://google.com",
        "Visit bank.com to verify",
        "Your link: bit.ly/xyz123",
        "No URLs here"
    ]

    for text in test_cases:
        urls = analyzer.extract_urls(text)
        print(f"Text: {text}")
        print(f"URLs: {urls}")
        print()

if __name__ == "__main__":
    test_analyzer()