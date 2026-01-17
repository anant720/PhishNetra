import { useState } from 'react'
import { RiskVisualization } from './RiskVisualization'
import { HighlightsDisplay } from './HighlightsDisplay'
import { ConfidenceBreakdown } from './ConfidenceBreakdown'
import { RecommendationsPanel } from './RecommendationsPanel'
import { URLAnalysisDisplay } from './URLAnalysisDisplay'

interface AnalysisResultsProps {
  results: any
  onNewAnalysis: () => void
}

export function AnalysisResults({ results, onNewAnalysis }: AnalysisResultsProps) {
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'urls', label: 'URL Analysis', icon: '🔗' },
    { id: 'highlights', label: 'Highlights', icon: '🔍' },
    { id: 'confidence', label: 'Model Confidence', icon: '🎯' },
    { id: 'recommendations', label: 'Recommendations', icon: '💡' },
  ]

  return (
    <div className="space-y-6">
      {/* Header with Risk Score */}
      <div className="card">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-neutral-900 mb-2">
              Analysis Results
            </h2>
            <p className="text-neutral-600">
              Risk assessment completed using multiple AI models
            </p>
          </div>

          <div className="flex items-center space-x-4">
            <RiskVisualization
              riskScore={results?.risk_score || 0}
              confidence={results?.confidence || 0}
              threatCategory={results?.threat_category || 'unknown'}
              size="large"
            />

            <button
              onClick={onNewAnalysis}
              className="btn btn-secondary whitespace-nowrap"
            >
              New Analysis
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="card">
        <div className="border-b border-neutral-200 mb-6">
          <nav className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors relative ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
                {/* URL Analysis Indicator Badge */}
                {tab.id === 'urls' && results.url_analysis && results.url_analysis.has_urls && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {results.url_analysis.url_count}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="min-h-[400px]">
          {activeTab === 'overview' && (
            <OverviewTab results={results} />
          )}

          {activeTab === 'urls' && (
            <URLAnalysisTab results={results} />
          )}

          {activeTab === 'highlights' && (
            <HighlightsTab results={results} />
          )}

          {activeTab === 'confidence' && (
            <ConfidenceTab results={results} />
          )}

          {activeTab === 'recommendations' && (
            <RecommendationsTab results={results} />
          )}
        </div>
      </div>
    </div>
  )
}

function OverviewTab({ results }: { results: any }) {
  const riskScore = results?.risk_score || 0
  const confidence = results?.confidence || 0
  const threatCategory = results?.threat_category || 'unknown'

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'text-red-600 bg-red-50'
    if (score >= 40) return 'text-amber-600 bg-amber-50'
    return 'text-green-600 bg-green-50'
  }

  const getRiskLabel = (score: number) => {
    if (score >= 70) return 'High Risk'
    if (score >= 40) return 'Moderate Risk'
    return 'Low Risk'
  }

  const getCategoryDescription = (category: string) => {
    const descriptions: { [key: string]: string } = {
      'legitimate': 'Message appears to be legitimate communication',
      'financial_manipulation': 'Attempt to manipulate money transfer or financial information',
      'phishing_redirection': 'Attempt to redirect to malicious website or phishing page',
      'identity_verification_abuse': 'Requesting personal information or credentials inappropriately',
      'authority_impersonation': 'Impersonating government, police, or official authority',
      'social_engineering': 'Using psychological manipulation or emotional appeal',
      'emotional_manipulation': 'Creating fear, urgency, or emotional pressure',
      'job_work_investment_scam': 'False job offer, investment opportunity, or work promise',
      'tech_support_malware_scam': 'Fake tech support or malware infection claim',
      'delivery_courier_scam': 'Fake delivery or package issue requiring payment',
      'lottery_reward_scam': 'False lottery win or reward requiring payment',
      'hybrid_evolving_scam': 'Multiple scam techniques combined',
      'unknown_scam': 'Scam pattern detected but category unclear'
    }
    return descriptions[category] || 'Analysis completed'
  }

  return (
    <div className="space-y-6">
      {/* Risk Assessment Summary */}
      <div className={`p-6 rounded-lg border-2 ${getRiskColor(riskScore)}`}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xl font-bold mb-1">Risk Assessment: {getRiskLabel(riskScore)}</h3>
            <p className="text-sm opacity-80">{getCategoryDescription(threatCategory)}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{riskScore.toFixed(1)}</div>
            <div className="text-sm">Risk Score</div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mt-4">
          <div className="bg-white/50 p-3 rounded-lg">
            <div className="text-lg font-semibold">{(confidence * 100).toFixed(0)}%</div>
            <div className="text-sm">AI Confidence</div>
          </div>
          <div className="bg-white/50 p-3 rounded-lg">
            <div className="text-lg font-semibold capitalize">{threatCategory.replace('_', ' ')}</div>
            <div className="text-sm">Primary Category</div>
          </div>
          <div className="bg-white/50 p-3 rounded-lg">
            <div className="text-lg font-semibold">
              {results.model_confidence_breakdown ? Object.keys(results.model_confidence_breakdown).length : 4}
            </div>
            <div className="text-sm">AI Models</div>
          </div>
        </div>

        {/* URL Analysis Indicator */}
        {results.url_analysis && results.url_analysis.has_urls && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-blue-600">🔗</span>
                <span className="text-blue-800 font-medium">
                  {results.url_analysis.url_count} URL{results.url_analysis.url_count !== 1 ? 's' : ''} Analyzed
                </span>
              </div>
              <button
                onClick={() => setActiveTab('urls')}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium underline"
              >
                View Details →
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detailed Reasoning */}
      <div className="bg-white p-6 rounded-lg border border-neutral-200">
        <h3 className="text-lg font-semibold text-neutral-900 mb-3">Analysis Explanation</h3>
        <p className="text-neutral-700 leading-relaxed">
          {results?.reasoning || 'Analysis completed with comprehensive risk assessment.'}
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border border-neutral-200">
          <div className="text-2xl font-bold text-neutral-900">{riskScore.toFixed(1)}</div>
          <div className="text-sm text-neutral-600">Risk Score (0-100)</div>
          <div className="text-xs text-neutral-500 mt-1">Higher = More Risk</div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-neutral-200">
          <div className="text-2xl font-bold text-neutral-900">{(confidence * 100).toFixed(0)}%</div>
          <div className="text-sm text-neutral-600">AI Confidence</div>
          <div className="text-xs text-neutral-500 mt-1">Model Certainty</div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-neutral-200">
          <div className="text-2xl font-bold text-neutral-900 capitalize">{threatCategory.replace('_', ' ')}</div>
          <div className="text-sm text-neutral-600">Scam Category</div>
          <div className="text-xs text-neutral-500 mt-1">Type of Threat</div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-neutral-200">
          <div className="text-2xl font-bold text-neutral-900">
            {results.model_confidence_breakdown ? Object.keys(results.model_confidence_breakdown).length : 4}
          </div>
          <div className="text-sm text-neutral-600">AI Models Used</div>
          <div className="text-xs text-neutral-500 mt-1">Multi-Model Analysis</div>
        </div>
      </div>

      {/* URL Analysis Summary */}
      {results.url_analysis && results.url_analysis.has_urls && (
        <div className={`p-6 rounded-lg border ${
          results.url_analysis.max_risk_score >= 70
            ? 'bg-red-50 border-red-200'
            : results.url_analysis.max_risk_score >= 40
            ? 'bg-amber-50 border-amber-200'
            : 'bg-blue-50 border-blue-200'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-lg font-semibold">
              🔗 URL Analysis
            </h4>
            <span className="text-sm font-medium">
              {results.url_analysis.url_count} URL{results.url_analysis.url_count !== 1 ? 's' : ''} • 
              Risk: {results.url_analysis.max_risk_score.toFixed(1)}
            </span>
          </div>
          <p className="text-sm text-neutral-700">
            {results.url_analysis.max_risk_score >= 70
              ? 'High-risk URLs detected. Review URL analysis tab for details.'
              : results.url_analysis.max_risk_score >= 40
              ? 'Suspicious URLs detected. Review URL analysis tab for details.'
              : 'URLs detected. Review URL analysis tab for details.'}
          </p>
        </div>
      )}

      {/* URL Analysis Alert */}
      {results.url_analysis && results.url_analysis.has_urls && (
        <div className="bg-amber-50 p-6 rounded-lg border border-amber-200">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🔗</span>
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-amber-900">URLs Detected & Analyzed</h4>
              <p className="text-amber-800">
                {results.url_analysis.url_count} URL{results.url_analysis.url_count !== 1 ? 's were' : ' was'} found and analyzed for security risks.
                Click the "URL Analysis" tab above to see detailed results.
              </p>
            </div>
            <button
              onClick={() => setActiveTab('urls')}
              className="btn btn-primary"
            >
              View URL Analysis
            </button>
          </div>
        </div>
      )}

      {/* Understanding Your Results */}
      <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
        <h4 className="text-lg font-semibold text-blue-900 mb-3">📖 Understanding Your Results</h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <h5 className="font-medium text-blue-800 mb-2">Risk Score Scale:</h5>
            <ul className="space-y-1 text-blue-700">
              <li><strong>0-30:</strong> Low Risk - Likely legitimate</li>
              <li><strong>31-60:</strong> Moderate Risk - Exercise caution</li>
              <li><strong>61-100:</strong> High Risk - Strong scam indicators</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-blue-800 mb-2">What to Do:</h5>
            <ul className="space-y-1 text-blue-700">
              <li><strong>Low Risk:</strong> Proceed with normal caution</li>
              <li><strong>Moderate:</strong> Verify independently</li>
              <li><strong>High Risk:</strong> Do not engage - report if needed</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Quick Highlights */}
      {results.explanation?.highlighted_phrases && results.explanation.highlighted_phrases.length > 0 && (
        <div className="bg-amber-50 p-6 rounded-lg border border-amber-200">
          <h4 className="text-lg font-semibold text-amber-900 mb-3">⚠️ Suspicious Elements Detected</h4>
          <div className="flex flex-wrap gap-2">
            {results.explanation.highlighted_phrases.slice(0, 6).map((highlight: any, index: number) => (
              <span
                key={index}
                className={`px-3 py-1 rounded-full text-sm font-medium ${
                  highlight.severity === 'high'
                    ? 'bg-red-100 text-red-800'
                    : highlight.severity === 'medium'
                    ? 'bg-amber-100 text-amber-800'
                    : 'bg-blue-100 text-blue-800'
                }`}
              >
                {highlight.phrase}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function HighlightsTab({ results }: { results: any }) {
  return (
    <div className="space-y-6">
      <HighlightsDisplay
        highlights={results?.explanation?.highlighted_phrases || []}
        originalText={results?.original_text || ''}
        highlightedHtml={results?.highlighted_text_html}
      />
    </div>
  )
}

function ConfidenceTab({ results }: { results: any }) {
  return (
    <div className="space-y-6">
      <ConfidenceBreakdown
        confidenceBreakdown={results?.model_confidence_breakdown || {}}
        overallConfidence={results?.confidence || 0}
      />
    </div>
  )
}

function URLAnalysisTab({ results }: { results: any }) {
  return (
    <div className="space-y-6">
      <URLAnalysisDisplay urlAnalysis={results?.url_analysis} />
    </div>
  )
}

function RecommendationsTab({ results }: { results: any }) {
  return (
    <div className="space-y-6">
      <RecommendationsPanel
        riskScore={results?.risk_score || 0}
        recommendations={results?.explanation?.recommendations || []}
        threatCategory={results?.threat_category || 'unknown'}
      />
    </div>
  )
}