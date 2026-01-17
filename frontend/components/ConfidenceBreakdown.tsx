interface ConfidenceBreakdownProps {
  confidenceBreakdown: Record<string, number>
  overallConfidence: number
}

export function ConfidenceBreakdown({ confidenceBreakdown, overallConfidence }: ConfidenceBreakdownProps) {
  const models = [
    {
      name: 'FastText',
      key: 'fasttext',
      description: 'Handles spelling errors, slang, and multilingual text',
      icon: '🔤'
    },
    {
      name: 'Sentence Transformer',
      key: 'sentence_transformer',
      description: 'Captures semantic meaning and intent detection',
      icon: '🧠'
    },
    {
      name: 'DistilBERT',
      key: 'distilbert',
      description: 'Contextual classification and pattern recognition',
      icon: '🎯'
    },
    {
      name: 'Similarity Search',
      key: 'similarity',
      description: 'Detects variants of known scam patterns',
      icon: '🔍'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Overall Confidence */}
      <div className="card">
        <h3 className="text-lg font-semibold text-neutral-900 mb-4">Overall Analysis Confidence</h3>

        <div className="flex items-center space-x-4 mb-4">
          <div className="text-3xl font-bold text-primary-600">
            {(overallConfidence * 100).toFixed(1)}%
          </div>
          <div className="flex-1">
            <div className="bg-neutral-200 rounded-full h-3">
              <div
                className="bg-primary-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${overallConfidence * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        <p className="text-neutral-600">
          {overallConfidence > 0.8
            ? "High confidence across all AI models with strong agreement."
            : overallConfidence > 0.6
            ? "Good agreement between models with reliable assessment."
            : overallConfidence > 0.4
            ? "Moderate confidence with some variation between models."
            : "Lower confidence - results should be interpreted cautiously."
          }
        </p>
      </div>

      {/* Model Breakdown */}
      <div className="grid md:grid-cols-2 gap-6">
        {models.map((model) => {
          const confidence = confidenceBreakdown[model.key] || 0
          const percentage = (confidence * 100).toFixed(1)

          return (
            <div key={model.key} className="card">
              <div className="flex items-start space-x-3">
                <div className="text-2xl">{model.icon}</div>
                <div className="flex-1">
                  <h4 className="font-semibold text-neutral-900 mb-1">
                    {model.name}
                  </h4>
                  <p className="text-sm text-neutral-600 mb-3">
                    {model.description}
                  </p>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-700">Confidence</span>
                      <span className="text-sm font-medium text-neutral-900">
                        {percentage}%
                      </span>
                    </div>

                    <div className="bg-neutral-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          confidence > 0.8 ? 'bg-green-500' :
                          confidence > 0.6 ? 'bg-yellow-500' :
                          confidence > 0.4 ? 'bg-orange-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${confidence * 100}%` }}
                      ></div>
                    </div>

                    <div className="text-xs text-neutral-500">
                      {confidence > 0.8 ? 'Very High' :
                       confidence > 0.6 ? 'High' :
                       confidence > 0.4 ? 'Moderate' : 'Low'} Confidence
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Model Contributions Explanation */}
      <div className="card">
        <h4 className="text-lg font-semibold text-neutral-900 mb-4">How Models Work Together</h4>

        <div className="space-y-4 text-sm text-neutral-700">
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <strong>Ensemble Approach:</strong> RiskAnalyzer AI combines multiple AI models using a weighted fusion system. Each model specializes in different aspects of scam detection.
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <strong>Confidence Weighting:</strong> Models with higher confidence in their predictions have more influence on the final risk score.
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <strong>Robust Detection:</strong> Even if one model is uncertain, others can provide reliable assessment, making the system more robust against edge cases.
            </div>
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="card bg-neutral-50">
        <h4 className="text-lg font-semibold text-neutral-900 mb-4">Technical Details</h4>

        <div className="grid md:grid-cols-2 gap-6 text-sm">
          <div>
            <h5 className="font-medium text-neutral-900 mb-2">Decision Fusion Algorithm</h5>
            <ul className="space-y-1 text-neutral-600">
              <li>• Weighted ensemble voting</li>
              <li>• Confidence-based model weighting</li>
              <li>• Risk score normalization (0-100)</li>
              <li>• Dynamic threshold adaptation</li>
            </ul>
          </div>

          <div>
            <h5 className="font-medium text-neutral-900 mb-2">Quality Metrics</h5>
            <ul className="space-y-1 text-neutral-600">
              <li>• F1-Score: Target >0.90</li>
              <li>• ROC-AUC: Target >0.95</li>
              <li>• Inference: &lt;500ms per analysis</li>
              <li>• Memory: &lt;2GB total usage</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}