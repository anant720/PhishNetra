"""
Simple URL analyzer for PhishNetra - Minimal working version
"""

import re
from typing import List, Dict, Any
from urllib.parse import urlparse

from ..core.logging import LoggerMixin

# Basic URL patterns
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]+)+(?:[:\d]+)?(?:/[^\s]*)?',
    re.IGNORECASE
)

SHORTLINK_DOMAINS = {
    "bit.ly", "tinyurl.com", "ow.ly", "goo.gl", "t.co", "short.link",
    "rebrand.ly", "cutt.ly", "is.gd", "v.gd", "shorturl.at", "tiny.cc"
}

class SimpleURLAnalyzer(LoggerMixin):
    """Simple URL analyzer for basic URL detection"""

    def __init__(self):
        self.logger.info("Initializing Simple URL Analyzer")

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        try:
            urls = URL_PATTERN.findall(text)
            self.logger.debug(f"Found {len(urls)} URLs: {urls}")
            return urls
        except Exception as e:
            self.logger.error(f"URL extraction failed: {e}")
            return []

    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """Basic URL analysis"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            risk_score = 0.0
            signals = []
            reasoning = []

            # Check for suspicious domain patterns
            if '-' in domain:
                risk_score += 20
                signals.append("hyphenated_domain")
                reasoning.append("Domain contains hyphens (suspicious)")

            if any(char.isdigit() for char in domain):
                risk_score += 15
                signals.append("numbers_in_domain")
                reasoning.append("Domain contains numbers")

            if len(domain.split('.')) > 3:
                risk_score += 25
                signals.append("excessive_subdomains")
                reasoning.append("Domain has excessive subdomains")

            # Check for shortened URLs
            if domain in SHORTLINK_DOMAINS:
                risk_score += 30
                signals.append("shortened_url")
                reasoning.append("URL uses known shortener service")

            # Check for suspicious TLDs
            tld = domain.split('.')[-1] if '.' in domain else ""
            suspicious_tlds = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'click', 'download', 'stream'}
            if tld in suspicious_tlds:
                risk_score += 35
                signals.append("suspicious_tld")
                reasoning.append(f"Suspicious TLD: .{tld}")

            # Check for brand impersonation
            known_brands = ['google', 'microsoft', 'apple', 'amazon', 'paypal', 'facebook', 'bank', 'sbi', 'hdfc']
            domain_lower = domain.lower()
            for brand in known_brands:
                if brand in domain_lower and domain_lower != f"{brand}.com":
                    risk_score += 40
                    signals.append("brand_impersonation")
                    reasoning.append(f"Possible impersonation of {brand}")
                    break

            # Determine verdict
            if risk_score >= 70:
                verdict = "High Risk"
            elif risk_score >= 40:
                verdict = "Suspicious"
            else:
                verdict = "Likely Legitimate"

            return {
                "url": url,
                "risk_score": min(risk_score, 100.0),
                "verdict": verdict,
                "signals": signals,
                "reasoning": reasoning,
                "details": {
                    "domain": domain,
                    "tld": tld
                }
            }

        except Exception as e:
            self.logger.error(f"URL analysis failed for {url}: {e}")
            return {
                "url": url,
                "risk_score": 50.0,
                "verdict": "Suspicious",
                "signals": ["analysis_error"],
                "reasoning": [f"Analysis failed: {str(e)}"],
                "details": {}
            }

# Create global instance
simple_url_analyzer = SimpleURLAnalyzer()