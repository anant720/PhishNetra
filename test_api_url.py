#!/usr/bin/env python3
"""
Test URL detection via API
"""

import requests
import json

def test_api_url_detection():
    """Test URL detection through the API"""

    test_messages = [
        "Check this link: https://google.com",
        "Visit http://example.com for more info",
        "Go to bank.com to verify your account",
        "Click here: https://paypal-secure.com/login?token=abc123",
        "Your link: bit.ly/xyz123",
        "No URLs here just text"
    ]

    api_url = "http://localhost:8000/api/v1/analyze"

    print("Testing URL Detection via API")
    print("=" * 40)

    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message[:50]}{'...' if len(message) > 50 else ''}")

        try:
            payload = {
                "text": message,
                "include_explainability": True
            }

            response = requests.post(api_url, json=payload, timeout=10)
            result = response.json()

            url_analysis = result.get('url_analysis', {})
            has_urls = url_analysis.get('has_urls', False)
            url_count = url_analysis.get('url_count', 0)
            urls = url_analysis.get('urls', [])

            print(f"  Has URLs: {has_urls}")
            print(f"  URL Count: {url_count}")
            if urls:
                for url_info in urls:
                    print(f"    - {url_info.get('url', 'unknown')}: Risk {url_info.get('risk_score', 0):.1f}")

            if has_urls:
                print("  SUCCESS: URLs detected")
            else:
                print("  FAILED: No URLs detected")

        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_api_url_detection()