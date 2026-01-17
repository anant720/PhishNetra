#!/usr/bin/env python3
"""
Test URL analysis in PhishNetra API
"""

import requests
import json

def test_url_analysis():
    """Test URL analysis via API"""

    test_text = "Check this scam link: https://fake-bank.com/verify and visit bank-secure.net"

    print("Testing URL Analysis in PhishNetra API")
    print("=" * 50)
    print(f"Test text: {test_text}")

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/analyze",
            json={
                "text": test_text,
                "include_explainability": True
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("[SUCCESS] API call successful")
            print(f"Risk Score: {result.get('risk_score', 0):.1f}")
            print(f"Category: {result.get('threat_category', 'unknown')}")

            # Check URL analysis
            url_analysis = result.get('url_analysis', {})
            if url_analysis:
                print("\n[URL ANALYSIS] Results:")
                print(f"  - URLs detected: {url_analysis.get('has_urls', False)}")
                print(f"  - URL count: {url_analysis.get('url_count', 0)}")
                print(f"  - Max risk score: {url_analysis.get('max_risk_score', 0):.1f}")

                urls = url_analysis.get('urls', [])
                if urls:
                    print(f"  - Detailed URL analysis for {len(urls)} URL(s):")
                    for i, url_result in enumerate(urls, 1):
                        print(f"    {i}. {url_result.get('url', '')}")
                        print(f"       Risk: {url_result.get('risk_score', 0):.1f}")
                        print(f"       Verdict: {url_result.get('verdict', 'unknown')}")
                        signals = url_result.get('signals', [])
                        if signals:
                            print(f"       Signals: {', '.join(signals[:3])}")
            else:
                print("[ERROR] No URL analysis found in response")

            return True
        else:
            print(f"[ERROR] API error: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_url_analysis()
    if success:
        print("\n✅ URL analysis is working correctly!")
    else:
        print("\n❌ URL analysis has issues.")