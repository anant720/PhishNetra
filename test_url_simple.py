#!/usr/bin/env python3
"""
Simple test for URL detection in PhishNetra
"""

import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_url_detection():
    """Test URL detection in main analysis pipeline"""
    try:
        from app.api.dependencies import analysis_service

        test_cases = [
            "Check this link: https://google.com",
            "Visit bank.com to verify your account",
            "Your link: bit.ly/xyz123",
            "No URLs here",
        ]

        print("Testing URL Detection in PhishNetra Analysis Pipeline")
        print("=" * 60)

        for i, test_text in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_text[:50]}...")
            print("-" * 50)

            result = await analysis_service.analyze_text(test_text, include_explainability=True)

            url_analysis = result.get('url_analysis', {})
            has_urls = url_analysis.get('has_urls', False)
            url_count = url_analysis.get('url_count', 0)

            print(f"Has URLs detected: {has_urls}")
            print(f"URL count: {url_count}")
            print(f"Risk score: {result.get('risk_score', 0):.1f}")
            print(f"Category: {result.get('threat_category', 'unknown')}")

            if has_urls and url_analysis.get('urls'):
                print("Detected URLs:")
                for url_result in url_analysis['urls'][:3]:  # Show first 3
                    print(f"  - {url_result.get('url', '')} ({url_result.get('verdict', '')})")
            elif has_urls:
                print("URL analysis present but no URL details")
            else:
                print("❌ No URLs detected")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_url_detection())