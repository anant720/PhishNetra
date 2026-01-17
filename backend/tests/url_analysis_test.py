"""
Self-testing for URL analysis pipeline
Tests URL-based scam detection with at least 10 examples including:
- Fake bank login pages
- Shortened phishing links
- Legitimate URLs (hard negatives)
- New domains mimicking known brands
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.url_analyzer import url_analyzer
from app.api.dependencies import analysis_service


# Test cases: URL-based scam examples
URL_TEST_CASES = [
    # 1. Fake bank login page
    {
        "text": "URGENT: Your SBI account has been suspended. Click here to verify: http://sb1-verify.com/login",
        "expected_risk_min": 60,
        "expected_category": "phishing_redirection",
        "description": "Fake bank login page with typosquatting"
    },
    
    # 2. Shortened phishing link
    {
        "text": "You've won a prize! Claim now: https://bit.ly/xyz123",
        "expected_risk_min": 50,
        "expected_category": "phishing_redirection",
        "description": "Shortened URL in scam message"
    },
    
    # 3. Brand impersonation
    {
        "text": "Your PayPal account needs verification. Visit paypa1-verify.net to update your information.",
        "expected_risk_min": 70,
        "expected_category": "identity_verification_abuse",
        "description": "PayPal brand impersonation with typosquatting"
    },
    
    # 4. Legitimate URL (hard negative)
    {
        "text": "Check out our services at https://www.google.com",
        "expected_risk_max": 30,
        "expected_category": "legitimate",
        "description": "Legitimate Google URL"
    },
    
    # 5. New domain mimicking known brand
    {
        "text": "Your Microsoft account requires immediate attention. Verify at microsoft-account-update.xyz",
        "expected_risk_min": 60,
        "expected_category": "phishing_redirection",
        "description": "New domain with suspicious TLD mimicking Microsoft"
    },
    
    # 6. Credential harvesting attempt
    {
        "text": "Security alert: Your account has been compromised. Login at secure-verify-login.com to reset password.",
        "expected_risk_min": 70,
        "expected_category": "identity_verification_abuse",
        "description": "Credential harvesting with suspicious domain pattern"
    },
    
    # 7. Financial manipulation with URL
    {
        "text": "Your payment failed. Update your card at payment-gateway-update.net/verify",
        "expected_risk_min": 65,
        "expected_category": "financial_manipulation",
        "description": "Financial manipulation with payment gateway URL"
    },
    
    # 8. Legitimate URL in normal context (hard negative)
    {
        "text": "Visit our website at https://github.com for more information.",
        "expected_risk_max": 30,
        "expected_category": "legitimate",
        "description": "Legitimate GitHub URL in normal context"
    },
    
    # 9. Multiple redirects (suspicious)
    {
        "text": "Claim your reward now: http://reward-claim.tk/redirect",
        "expected_risk_min": 60,
        "expected_category": "phishing_redirection",
        "description": "Suspicious TLD with reward scam"
    },
    
    # 10. Authority impersonation with URL
    {
        "text": "IRS notice: Your tax refund is pending. Verify at irs-verify-official.info",
        "expected_risk_min": 70,
        "expected_category": "authority_impersonation",
        "description": "IRS impersonation with suspicious domain"
    },
    
    # 11. Domain mention (without http://)
    {
        "text": "Your account needs verification. Visit bank-verify-site.com immediately.",
        "expected_risk_min": 55,
        "expected_category": "phishing_redirection",
        "description": "Domain mention requesting action"
    },
    
    # 12. Legitimate domain in scam context (should still detect scam)
    {
        "text": "URGENT: Your account will be closed. Click http://www.google.com to verify immediately!",
        "expected_risk_min": 50,  # Scam language should still trigger
        "expected_category": "phishing_redirection",
        "description": "Legitimate URL but scam language (should detect scam intent)"
    }
]


async def test_url_analysis():
    """Test URL analysis on all test cases"""
    print("=" * 80)
    print("URL ANALYSIS SELF-TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    results = []
    
    for i, test_case in enumerate(URL_TEST_CASES, 1):
        print(f"Test {i}/{len(URL_TEST_CASES)}: {test_case['description']}")
        print(f"  Text: {test_case['text'][:80]}...")
        
        try:
            # Perform analysis
            result = await analysis_service.analyze_text(
                text=test_case['text'],
                include_explainability=True
            )
            
            risk_score = result.get('risk_score', 0)
            threat_category = result.get('threat_category', 'unknown')
            url_analysis = result.get('url_analysis', {})
            
            # Check expectations
            test_passed = True
            issues = []
            
            # Check risk score
            if 'expected_risk_min' in test_case:
                if risk_score < test_case['expected_risk_min']:
                    test_passed = False
                    issues.append(f"Risk score {risk_score:.1f} below minimum {test_case['expected_risk_min']}")
            
            if 'expected_risk_max' in test_case:
                if risk_score > test_case['expected_risk_max']:
                    test_passed = False
                    issues.append(f"Risk score {risk_score:.1f} above maximum {test_case['expected_risk_max']}")
            
            # Check category (fuzzy match)
            expected_cat = test_case.get('expected_category', '')
            if expected_cat and expected_cat != 'legitimate':
                if expected_cat not in threat_category:
                    # Check if it's in threat_categories
                    threat_categories = result.get('threat_categories', [])
                    category_found = any(
                        expected_cat in cat.get('category', '') 
                        for cat in threat_categories
                    )
                    if not category_found:
                        issues.append(f"Expected category '{expected_cat}' not found (got '{threat_category}')")
                        # Don't fail for category mismatch, just warn
            
            # Check URL analysis was performed
            if url_analysis.get('has_urls'):
                max_url_risk = url_analysis.get('max_risk_score', 0)
                print(f"  ✓ URL analysis performed: {url_analysis.get('url_count', 0)} URL(s), max risk: {max_url_risk:.1f}")
            else:
                # Check if URLs should have been detected
                if 'http' in test_case['text'] or '.com' in test_case['text']:
                    issues.append("URLs detected in text but URL analysis not performed")
            
            if test_passed and not issues:
                print(f"  ✓ PASSED - Risk: {risk_score:.1f}, Category: {threat_category}")
                passed += 1
            else:
                print(f"  ✗ FAILED - Risk: {risk_score:.1f}, Category: {threat_category}")
                for issue in issues:
                    print(f"    - {issue}")
                failed += 1
            
            results.append({
                'test': i,
                'description': test_case['description'],
                'passed': test_passed,
                'risk_score': risk_score,
                'category': threat_category,
                'issues': issues
            })
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed += 1
            results.append({
                'test': i,
                'description': test_case['description'],
                'passed': False,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(URL_TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(URL_TEST_CASES)*100):.1f}%")
    print()
    
    # Detailed failures
    if failed > 0:
        print("FAILED TESTS:")
        for result in results:
            if not result.get('passed', False):
                print(f"  Test {result['test']}: {result['description']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                else:
                    print(f"    Risk Score: {result.get('risk_score', 0):.1f}")
                    print(f"    Category: {result.get('category', 'unknown')}")
                    for issue in result.get('issues', []):
                        print(f"    - {issue}")
        print()
    
    # Critical check: No phishing URLs labeled as legitimate
    print("CRITICAL CHECK: Phishing URLs labeled as legitimate")
    print("-" * 80)
    phishing_tests = [r for r in results if 'phishing' in r.get('description', '').lower() or 
                      'fake' in r.get('description', '').lower() or
                      'impersonation' in r.get('description', '').lower()]
    
    false_negatives = 0
    for result in phishing_tests:
        if result.get('risk_score', 0) < 40 or result.get('category', '') == 'legitimate':
            false_negatives += 1
            print(f"  ✗ CRITICAL: {result['description']} - Risk: {result.get('risk_score', 0):.1f}, Category: {result.get('category', 'unknown')}")
    
    if false_negatives == 0:
        print("  ✓ No false negatives detected - all phishing URLs correctly identified")
    else:
        print(f"  ✗ {false_negatives} false negative(s) detected - RETRAIN REQUIRED")
    
    print()
    print("=" * 80)
    
    # Return success if all tests passed and no false negatives
    return passed == len(URL_TEST_CASES) and false_negatives == 0


if __name__ == "__main__":
    success = asyncio.run(test_url_analysis())
    sys.exit(0 if success else 1)
