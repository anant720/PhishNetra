#!/usr/bin/env python3
"""
Direct test of URL analyzer without full app dependencies
"""

import re
import sys
import os

# Add only the core modules we need
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Simple URL extraction function
def extract_urls_simple(text: str):
    """Simple URL extraction"""

    # URL patterns
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]+)+(?:[:\d]+)?(?:/[^\s]*)?',
        re.IGNORECASE
    )

    DOMAIN_MENTION_PATTERN = re.compile(
        r'\b(?:visit|go to|check|access|open|click|link|website|site|login to|connect to)\s+([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:\.[a-zA-Z]{2,}))',
        re.IGNORECASE
    )

    DIRECT_DOMAIN_PATTERN = re.compile(
        r'\b([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:\.[a-zA-Z]{2,}))\b',
        re.IGNORECASE
    )

    urls = []

    # 1. Extract full URLs
    full_urls = URL_PATTERN.findall(text)
    urls.extend(full_urls)

    # 2. Extract domain mentions with action verbs
    domain_mentions = DOMAIN_MENTION_PATTERN.findall(text)
    for domain in domain_mentions:
        if domain and '.' in domain:
            if not any(domain in url for url in full_urls):
                urls.append(f"https://{domain}")

    # 3. Extract suspicious direct domains
    direct_domains = DIRECT_DOMAIN_PATTERN.findall(text)
    for domain in direct_domains:
        if (domain and '.' in domain and
            len(domain) > 4 and
            not any(domain in url for url in urls) and
            ('-' in domain or len(domain.split('.')) > 2 or any(c.isdigit() for c in domain))):
            urls.append(f"https://{domain}")

    # Remove duplicates
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls

def test_direct_extraction():
    """Test direct URL extraction"""

    test_cases = [
        "Check this link: https://google.com",
        "Visit http://example.com for more info",
        "Go to bank.com to verify your account",
        "Click here: https://paypal-secure.com/login?token=abc123",
        "Your link: bit.ly/xyz123",
        "No URLs here just text",
        "Multiple urls: https://site1.com and visit site2.net",
        "Shortened: goo.gl/abc123 and tinyurl.com/def456"
    ]

    print("Testing Direct URL Extraction")
    print("=" * 40)

    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text}")

        urls = extract_urls_simple(test_text)
        print(f"  Extracted URLs: {urls}")

        if urls:
            print("  SUCCESS: URLs found")
        else:
            print("  FAILED: No URLs found")

if __name__ == "__main__":
    test_direct_extraction()