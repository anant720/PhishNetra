interface RecommendationsPanelProps {
  riskScore: number
  recommendations: string[]
  threatCategory: string
}

export function RecommendationsPanel({ riskScore, recommendations, threatCategory }: RecommendationsPanelProps) {
  // Default recommendations based on risk score
  const getDefaultRecommendations = (score: number) => {
    if (score >= 80) {
      return [
        "Do not respond to this message or provide any personal information",
        "Do not click any links or download attachments",
        "Report this message to the appropriate authorities or platform",
        "Warn your contacts about similar suspicious messages",
        "Consider changing passwords if you've interacted with similar content"
      ]
    } else if (score >= 60) {
      return [
        "Verify the sender through official channels before responding",
        "Do not share sensitive information via this communication method",
        "Check for similar messages from trusted sources",
        "Be cautious with any requests for immediate action",
        "Consider reporting if the message seems suspicious"
      ]
    } else if (score >= 40) {
      return [
        "Exercise caution when responding to this message",
        "Verify any claims independently using official sources",
        "Avoid sharing personal or financial information",
        "Take time to consider the request rather than acting immediately",
        "Trust your instincts - if something feels wrong, it probably is"
      ]
    } else {
      return [
        "This message appears legitimate but always stay vigilant",
        "Continue practicing good cybersecurity habits",
        "Be cautious with unsolicited messages even if they seem safe",
        "Report any future suspicious communications"
      ]
    }
  }

  const defaultRecs = getDefaultRecommendations(riskScore)
  const allRecommendations = recommendations.length > 0 ? recommendations : defaultRecs

  // Category-specific additional recommendations
  const getCategorySpecificRecommendations = (category: string) => {
    const categoryRecs: Record<string, string[]> = {
      phishing: [
        "Never enter login credentials on suspicious websites",
        "Check URL carefully - scammers use similar-looking domains",
        "Use official apps instead of clicking links in messages",
        "Enable two-factor authentication on all accounts"
      ],
      financial_scam: [
        "Never send money to unverified recipients",
        "Contact your bank/financial institution directly using official numbers",
        "Be extremely cautious with investment opportunities",
        "Verify all financial transactions through official channels"
      ],
      social_engineering: [
        "Verify the identity of people asking for help through multiple channels",
        "Never send money for emergencies without verification",
        "Be suspicious of urgent requests from 'friends' or 'family'",
        "Set up verification protocols for sensitive requests"
      ],
      authority_scam: [
        "Government agencies never ask for payment via unusual methods",
        "Verify official communications through official websites",
        "Be suspicious of threats of legal action or arrest",
        "Contact agencies using verified contact information"
      ]
    }

    return categoryRecs[category] || []
  }

  const categoryRecs = getCategorySpecificRecommendations(threatCategory)

  return (
    <div className="space-y-6">
      {/* Risk-based Recommendations */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <div className={`p-2 rounded-lg ${
            riskScore >= 70 ? 'bg-red-100 text-red-600' :
            riskScore >= 40 ? 'bg-amber-100 text-amber-600' : 'bg-green-100 text-green-600'
          }`}>
            {riskScore >= 70 ? '🚨' : riskScore >= 40 ? '⚠️' : '✅'}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-neutral-900">
              {riskScore >= 70 ? 'High Risk - Take Immediate Action' :
               riskScore >= 40 ? 'Medium Risk - Exercise Caution' :
               'Low Risk - Stay Vigilant'}
            </h3>
            <p className="text-neutral-600 text-sm">
              Recommendations based on risk score of {riskScore.toFixed(1)}/100
            </p>
          </div>
        </div>

        <div className="space-y-3">
          {allRecommendations.map((rec, index) => (
            <div key={index} className="flex items-start space-x-3 p-3 bg-neutral-50 rounded-lg">
              <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-neutral-800">{rec}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Category-specific Recommendations */}
      {categoryRecs.length > 0 && (
        <div className="card">
          <h4 className="text-lg font-semibold text-neutral-900 mb-4">
            Specific to {threatCategory.replace('_', ' ').toUpperCase()} Threats
          </h4>

          <div className="space-y-3">
            {categoryRecs.map((rec, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                <p className="text-blue-900">{rec}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* General Security Best Practices */}
      <div className="card bg-neutral-50">
        <h4 className="text-lg font-semibold text-neutral-900 mb-4">
          General Security Best Practices
        </h4>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex items-start space-x-2">
              <span className="text-green-600 font-bold">✓</span>
              <span className="text-sm text-neutral-700">Use official websites and verified contact information</span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-green-600 font-bold">✓</span>
              <span className="text-sm text-neutral-700">Enable two-factor authentication everywhere possible</span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-green-600 font-bold">✓</span>
              <span className="text-sm text-neutral-700">Regularly update passwords and security settings</span>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-start space-x-2">
              <span className="text-red-600 font-bold">✗</span>
              <span className="text-sm text-neutral-700">Never share passwords or financial information via email/SMS</span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-red-600 font-bold">✗</span>
              <span className="text-sm text-neutral-700">Don't click links in unsolicited messages</span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-red-600 font-bold">✗</span>
              <span className="text-sm text-neutral-700">Avoid urgent requests that pressure quick action</span>
            </div>
          </div>
        </div>
      </div>

      {/* Report Button */}
      <div className="card border-amber-200 bg-amber-50">
        <div className="text-center">
          <h4 className="text-lg font-semibold text-amber-900 mb-2">
            Help Improve Detection
          </h4>
          <p className="text-amber-800 mb-4">
            Report suspicious messages to help train better AI models and protect others.
          </p>
          <button className="btn bg-amber-600 hover:bg-amber-700 text-white">
            Report This Message
          </button>
        </div>
      </div>
    </div>
  )
}