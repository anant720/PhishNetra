interface RiskVisualizationProps {
  riskScore: number
  confidence: number
  threatCategory: string
  size?: 'small' | 'large'
}

export function RiskVisualization({
  riskScore = 0,
  confidence = 0,
  threatCategory = 'unknown',
  size = 'small'
}: RiskVisualizationProps) {
  const getRiskColor = (score: number) => {
    if (score >= 70) return 'risk-high'
    if (score >= 40) return 'risk-medium'
    return 'risk-low'
  }

  const getRiskLabel = (score: number) => {
    if (score >= 70) return 'High Risk'
    if (score >= 40) return 'Medium Risk'
    return 'Low Risk'
  }

  const riskColor = getRiskColor(riskScore)
  const riskLabel = getRiskLabel(riskScore)

  if (size === 'large') {
    return (
      <div className="bg-white p-6 rounded-lg border border-neutral-200 min-w-[250px]">
        <div className="text-center">
          <div className={`text-4xl font-bold mb-2 ${
            riskScore >= 70 ? 'text-risk-high' :
            riskScore >= 40 ? 'text-risk-medium' : 'text-risk-low'
          }`}>
            {riskScore.toFixed(1)}
          </div>

          <div className="text-sm text-neutral-600 mb-4">Risk Score</div>

          {/* Risk Meter */}
          <div className="risk-meter mb-4">
            <div
              className={`risk-meter-fill ${riskColor}`}
              style={{ width: `${riskScore}%` }}
            ></div>
          </div>

          <div className="text-sm font-medium text-neutral-900 mb-2">
            {riskLabel}
          </div>

          <div className="text-xs text-neutral-500 mb-3">
            {threatCategory.replace('_', ' ').toUpperCase()}
          </div>

          {/* Confidence Indicator */}
          <div className="flex items-center justify-center space-x-2 text-sm">
            <span className="text-neutral-600">Confidence:</span>
            <span className="font-medium text-neutral-900">
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    )
  }

  // Small size for compact displays
  return (
    <div className="flex items-center space-x-4">
      <div className="flex-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium text-neutral-900">
            Risk Score: {riskScore.toFixed(1)}
          </span>
          <span className={`text-sm font-medium ${
            riskScore >= 70 ? 'text-risk-high' :
            riskScore >= 40 ? 'text-risk-medium' : 'text-risk-low'
          }`}>
            {riskLabel}
          </span>
        </div>

        <div className="risk-meter">
          <div
            className={`risk-meter-fill ${riskColor}`}
            style={{ width: `${riskScore}%` }}
          ></div>
        </div>
      </div>

      <div className="text-right">
        <div className="text-xs text-neutral-500">Confidence</div>
        <div className="text-sm font-medium text-neutral-900">
          {(confidence * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  )
}