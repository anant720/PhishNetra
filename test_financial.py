import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.api.dependencies import analysis_service
import asyncio

async def test_financial_detection():
    """Test if financial keywords are properly detected"""

    test_messages = [
        "URGENT: Transfer money to this account now or your account will be suspended. Call 555-0123",
        "Send $500 to this PayPal account immediately or lose access",
        "Your bank account has been frozen. Verify with your debit card number",
        "Investment opportunity: Send Bitcoin to this wallet for guaranteed returns",
        "This is legitimate business communication about a project"
    ]

    print("Testing Financial Keyword Detection\n")
    print("=" * 50)

    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message[:60]}{'...' if len(message) > 60 else ''}")
        print("-" * 50)

        try:
            result = await analysis_service.analyze_text(message)

            risk_score = result.get('risk_score', 0)
            threat_category = result.get('threat_category', 'unknown')
            reasoning = result.get('reasoning', '')

            print(f"Risk Score: {risk_score:.1f}")
            print(f"Category: {threat_category.replace('_', ' ')}")
            print(f"Reasoning: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")

            # Check if financial category was detected for financial messages
            if i <= 4 and threat_category == 'financial_manipulation':
                print("✅ CORRECT: Financial category detected")
            elif i == 5 and threat_category == 'legitimate':
                print("✅ CORRECT: Legitimate category for clean message")
            elif i <= 4 and threat_category != 'financial_manipulation':
                print(f"❌ ISSUE: Expected financial_manipulation, got {threat_category}")
            else:
                print("⚠️  Mixed results - needs review")

        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_financial_detection())