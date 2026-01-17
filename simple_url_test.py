#!/usr/bin/env python3
"""
Simple test for URL detection without dependencies
"""

import re

# URL patterns to test
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]+)+(?:[:\d]+)?(?:/[^\s]*)?',
    re.IGNORECASE
)

def test_url_detection():
    """Test URL detection with simple regex"""

    test_cases = [
        "Check this link: https://google.com",
        "Visit http://example.com for more info",
        "Go to bank.com to verify your account",
        "Click here: https://paypal-secure.com/login?token=abc123",
        "Your link: bit.ly/xyz123",
        "No URLs here just text"
    ]

    print("Testing Simple URL Detection")
    print("=" * 40)

    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text}")

        matches = URL_PATTERN.findall(test_text)
        print(f"  Regex matches: {matches}")

        if matches:
            print("  SUCCESS: URLs found")
        else:
            print("  FAILED: No URLs found")

if __name__ == "__main__":
    test_url_detection()