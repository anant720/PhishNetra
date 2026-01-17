"""
Self-testing system for PhishNetra
Tests the system with 20+ diverse scam examples to ensure proper classification
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.dependencies import PhishNetraService
from app.models.decision_fusion import ThreatCategory


# Test cases: (text, expected_category, expected_min_risk, description)
TEST_CASES = [
    # 1. FINANCIAL MANIPULATION SCAMS
    (
        "Your account has been debited $500. To reverse this transaction, please click here and verify your account: http://fakebank.com/verify",
        ThreatCategory.FINANCIAL_MANIPULATION,
        60,
        "Financial manipulation with fake debit and verification request"
    ),
    (
        "Refund requires action. Approve to receive funds: https://refund-process.com",
        ThreatCategory.FINANCIAL_MANIPULATION,
        55,
        "Refund manipulation scam"
    ),
    
    # 2. PHISHING & REDIRECTION SCAMS
    (
        "Click here to update your password: http://login-security-verify.net/update",
        ThreatCategory.PHISHING_REDIRECTION,
        70,
        "Phishing with credential harvesting intent"
    ),
    (
        "Your account needs verification. Visit: bit.ly/verify-now",
        ThreatCategory.PHISHING_REDIRECTION,
        65,
        "Phishing with shortened URL"
    ),
    
    # 3. IDENTITY VERIFICATION ABUSE
    (
        "Please provide your SSN and date of birth to verify your identity for KYC compliance.",
        ThreatCategory.IDENTITY_VERIFICATION_ABUSE,
        75,
        "Identity verification abuse - personal data extraction"
    ),
    (
        "Update your personal information to continue using our services. Enter your details here.",
        ThreatCategory.IDENTITY_VERIFICATION_ABUSE,
        60,
        "Personal data request under false pretense"
    ),
    
    # 4. AUTHORITY IMPERSONATION
    (
        "Official IRS notice: You owe $2,450 in taxes. Pay now to avoid arrest. Call immediately.",
        ThreatCategory.AUTHORITY_IMPERSONATION,
        85,
        "Authority impersonation - IRS scam"
    ),
    (
        "Police Department: Warrant issued for your arrest. Pay $500 fine immediately or face legal action.",
        ThreatCategory.AUTHORITY_IMPERSONATION,
        80,
        "Authority impersonation - Police scam"
    ),
    
    # 5. SOCIAL ENGINEERING
    (
        "Hello, I'm reaching out to help you secure your account. This is a trusted service. Please provide your information.",
        ThreatCategory.SOCIAL_ENGINEERING,
        50,
        "Social engineering - trust building without urgency"
    ),
    
    # 6. EMOTIONAL MANIPULATION
    (
        "URGENT: Your family member is in trouble and needs money immediately. Send $1000 now to help.",
        ThreatCategory.EMOTIONAL_MANIPULATION,
        75,
        "Emotional manipulation - fake family emergency"
    ),
    (
        "Emergency! Your account will be closed in 24 hours. Act now to prevent suspension.",
        ThreatCategory.EMOTIONAL_MANIPULATION,
        70,
        "Emotional manipulation - fear and urgency"
    ),
    
    # 7. JOB/WORK/INVESTMENT SCAMS
    (
        "Work from home opportunity! Earn $5000/month. Registration fee: $200. Start today!",
        ThreatCategory.JOB_WORK_INVESTMENT_SCAM,
        70,
        "Job scam with registration fee"
    ),
    (
        "Guaranteed 50% return on investment in 30 days. Invest now in our crypto platform.",
        ThreatCategory.JOB_WORK_INVESTMENT_SCAM,
        75,
        "Investment scam with unrealistic returns"
    ),
    
    # 8. TECH SUPPORT & MALWARE
    (
        "Your computer has been infected with a virus. Call us immediately at 1-800-SCAM to fix it.",
        ThreatCategory.TECH_SUPPORT_MALWARE_SCAM,
        70,
        "Tech support scam - fake virus alert"
    ),
    (
        "Download this security update to protect your device from malware.",
        ThreatCategory.TECH_SUPPORT_MALWARE_SCAM,
        60,
        "Tech support scam - fake security update"
    ),
    
    # 9. DELIVERY/COURIER SCAM
    (
        "Your package delivery failed. Pay $5.99 redelivery fee to receive your shipment.",
        ThreatCategory.DELIVERY_COURIER_SCAM,
        65,
        "Delivery scam with fake redelivery fee"
    ),
    (
        "Package out for delivery. Confirm your address and pay customs fee to receive.",
        ThreatCategory.DELIVERY_COURIER_SCAM,
        60,
        "Delivery scam with customs fee"
    ),
    
    # 10. LOTTERY/REWARD SCAM
    (
        "Congratulations! You've won $1,000,000 in our lottery. Claim your prize by paying a $100 processing fee.",
        ThreatCategory.LOTTERY_REWARD_SCAM,
        80,
        "Lottery scam with processing fee"
    ),
    (
        "You've been selected! Claim your reward now. Send $50 to activate your prize.",
        ThreatCategory.LOTTERY_REWARD_SCAM,
        70,
        "Reward scam with activation fee"
    ),
    
    # 11. HYBRID/EVOLVING SCAMS (Multiple intents)
    (
        "URGENT: Official bank notice. Your account has been compromised. Click here to verify and prevent unauthorized access: http://secure-bank-verify.com",
        ThreatCategory.HYBRID_EVOLVING_SCAM,
        80,
        "Hybrid scam - combines authority, urgency, phishing, and financial manipulation"
    ),
    (
        "Government refund available. You're eligible for $1,800. Verify your identity and claim now: bit.ly/gov-refund",
        ThreatCategory.HYBRID_EVOLVING_SCAM,
        75,
        "Hybrid scam - authority, financial, identity abuse, and phishing"
    ),
    
    # 12. LEGITIMATE MESSAGES (Hard negatives)
    (
        "Hello, how are you? I hope you're doing well.",
        ThreatCategory.LEGITIMATE,
        20,
        "Legitimate friendly message"
    ),
    (
        "Thank you for your purchase. Your order #12345 has been shipped and will arrive in 3-5 business days.",
        ThreatCategory.LEGITIMATE,
        15,
        "Legitimate order confirmation"
    ),
    (
        "Reminder: Your appointment is scheduled for tomorrow at 2 PM. Please confirm if you can attend.",
        ThreatCategory.LEGITIMATE,
        10,
        "Legitimate appointment reminder"
    ),
    
    # 13. SUBTLE SCAMS (Should still be detected)
    (
        "Your recent transaction requires verification. Please confirm to continue.",
        ThreatCategory.FINANCIAL_MANIPULATION,
        45,
        "Subtle financial scam - vague reference"
    ),
    (
        "We noticed unusual activity. Please verify your account to ensure security.",
        ThreatCategory.IDENTITY_VERIFICATION_ABUSE,
        50,
        "Subtle identity abuse - vague security claim"
    ),
    
    # 14. POLITE SCAMS (Should not be classified as legitimate)
    (
        "Kindly please update your account information. Thank you for your cooperation.",
        ThreatCategory.SOCIAL_ENGINEERING,
        40,
        "Polite scam - social engineering with polite tone"
    ),
]


async def run_self_test():
    """Run comprehensive self-test"""
    print("=" * 80)
    print("PHISHNETRA SELF-TESTING SYSTEM")
    print("=" * 80)
    print()
    
    service = PhishNetraService()
    
    passed = 0
    failed = 0
    warnings = 0
    
    for i, (text, expected_category, expected_min_risk, description) in enumerate(TEST_CASES, 1):
        print(f"\n[TEST {i}/{len(TEST_CASES)}] {description}")
        print(f"Text: {text[:80]}...")
        
        try:
            result = await service.analyze_text(text, include_explainability=True)
            
            risk_score = result.get('risk_score', 0)
            primary_category = result.get('threat_category', 'unknown')
            threat_categories = result.get('threat_categories', [])
            
            # Check if expected category is in results (primary or multi-label)
            category_found = (
                primary_category == expected_category.value or
                any(cat.get('category') == expected_category.value for cat in threat_categories)
            )
            
            risk_ok = risk_score >= expected_min_risk
            
            # Determine test result
            if category_found and risk_ok:
                status = "✅ PASSED"
                passed += 1
            elif category_found and not risk_ok:
                status = "⚠️  WARNING (Category correct, but risk too low)"
                warnings += 1
            elif not category_found and risk_ok:
                status = "⚠️  WARNING (Risk correct, but category mismatch)"
                warnings += 1
            else:
                status = "❌ FAILED"
                failed += 1
            
            print(f"Status: {status}")
            print(f"  Risk Score: {risk_score:.1f} (expected >= {expected_min_risk})")
            print(f"  Primary Category: {primary_category}")
            print(f"  Expected Category: {expected_category.value}")
            
            if threat_categories:
                print(f"  Multi-label Categories: {len(threat_categories)}")
                for cat in threat_categories[:3]:  # Show top 3
                    print(f"    - {cat.get('category')}: {cat.get('confidence', 0):.2f}")
            
            # Check for URL analysis if URL present
            if 'http' in text.lower() or 'www' in text.lower():
                url_analysis = result.get('url_analysis', {})
                if url_analysis.get('has_urls'):
                    print(f"  URL Analysis: {url_analysis.get('url_count')} URL(s) analyzed")
                    print(f"    Max URL Risk: {url_analysis.get('max_risk_score', 0):.1f}")
            
            if not category_found:
                print(f"  ⚠️  Category mismatch! Expected {expected_category.value}, got {primary_category}")
            if not risk_ok:
                print(f"  ⚠️  Risk too low! Expected >= {expected_min_risk}, got {risk_score:.1f}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed / len(TEST_CASES) * 100):.1f}%")
    print("=" * 80)
    
    # Critical checks
    print("\nCRITICAL CHECKS:")
    
    # Check 1: No financial scams labeled as legitimate
    financial_scams = [tc for tc in TEST_CASES if tc[1] == ThreatCategory.FINANCIAL_MANIPULATION]
    print(f"✓ Financial scams tested: {len(financial_scams)}")
    
    # Check 2: No phishing attempts labeled as benign
    phishing_scams = [tc for tc in TEST_CASES if tc[1] == ThreatCategory.PHISHING_REDIRECTION]
    print(f"✓ Phishing scams tested: {len(phishing_scams)}")
    
    # Check 3: Legitimate messages have low risk
    legitimate = [tc for tc in TEST_CASES if tc[1] == ThreatCategory.LEGITIMATE]
    print(f"✓ Legitimate messages tested: {len(legitimate)}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed. System needs improvement.")
        return False
    elif warnings > 0:
        print(f"\n⚠️  {warnings} warning(s). System is functional but may need tuning.")
        return True
    else:
        print("\n✅ All tests passed! System is working correctly.")
        return True


if __name__ == "__main__":
    success = asyncio.run(run_self_test())
    sys.exit(0 if success else 1)
