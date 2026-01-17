import { useState } from 'react'

interface URLAnalysisDisplayProps {
  urlAnalysis: any
}

export function URLAnalysisDisplay({ urlAnalysis }: URLAnalysisDisplayProps) {
  if (!urlAnalysis || !urlAnalysis.has_urls) {
    return (
      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <p className="text-neutral-600">No URLs detected in the message.</p>
      </div>
    )
  }

  const urls = urlAnalysis.urls || []
  const maxRisk = urlAnalysis.max_risk_score || 0

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-neutral-900">
            URL Analysis Summary
          </h3>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            maxRisk >= 70
              ? 'bg-red-100 text-red-800'
              : maxRisk >= 40
              ? 'bg-amber-100 text-amber-800'
              : 'bg-green-100 text-green-800'
          }`}>
            {urlAnalysis.url_count} URL{urlAnalysis.url_count !== 1 ? 's' : ''} Found
          </span>
        </div>
        <p className="text-neutral-700">
          Maximum URL risk score: <strong>{maxRisk.toFixed(1)}</strong>
        </p>
      </div>

      {/* Individual URL Analysis */}
      <div className="space-y-4">
        {urls.map((urlResult: any, index: number) => (
          <URLCard key={index} urlResult={urlResult} />
        ))}
      </div>
    </div>
  )
}

function URLCard({ urlResult }: { urlResult: any }) {
  const [expanded, setExpanded] = useState(false)
  
  const riskScore = urlResult.risk_score || 0
  const verdict = urlResult.verdict || 'Unknown'
  const signals = urlResult.signals || []
  const reasoning = urlResult.reasoning || []
  const details = urlResult.details || {}

  const getVerdictColor = (verdict: string) => {
    if (verdict === 'High Risk') return 'text-red-600 bg-red-50 border-red-200'
    if (verdict === 'Suspicious') return 'text-amber-600 bg-amber-50 border-amber-200'
    return 'text-green-600 bg-green-50 border-green-200'
  }

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'text-red-600'
    if (score >= 40) return 'text-amber-600'
    return 'text-green-600'
  }

  return (
    <div className={`border rounded-lg p-6 transition-all ${getVerdictColor(verdict)}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-2xl">🔗</span>
            <div className="flex-1">
              <a
                href={urlResult.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 break-all font-mono text-sm"
              >
                {urlResult.url}
              </a>
            </div>
          </div>

          <div className="flex items-center gap-4 mb-3">
            <div>
              <span className="text-sm text-neutral-600">Risk Score: </span>
              <span className={`text-lg font-bold ${getRiskColor(riskScore)}`}>
                {riskScore.toFixed(1)}
              </span>
            </div>
            <div>
              <span className="text-sm text-neutral-600">Verdict: </span>
              <span className="text-lg font-semibold">{verdict}</span>
            </div>
          </div>

          {/* Signals */}
          {signals.length > 0 && (
            <div className="mb-3">
              <span className="text-sm font-medium text-neutral-700">Risk Signals: </span>
              <div className="flex flex-wrap gap-2 mt-2">
                {signals.map((signal: string, idx: number) => (
                  <span
                    key={idx}
                    className="px-2 py-1 bg-white/70 rounded text-xs font-medium"
                  >
                    {signal.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Reasoning */}
          {reasoning.length > 0 && (
            <div className="mb-3">
              <ul className="list-disc list-inside text-sm text-neutral-700 space-y-1">
                {reasoning.slice(0, expanded ? reasoning.length : 2).map((reason: string, idx: number) => (
                  <li key={idx}>{reason}</li>
                ))}
              </ul>
              {reasoning.length > 2 && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="text-sm text-blue-600 hover:text-blue-800 mt-2"
                >
                  {expanded ? 'Show less' : `Show ${reasoning.length - 2} more reasons`}
                </button>
              )}
            </div>
          )}

          {/* Details (expandable) */}
          {details && Object.keys(details).length > 0 && (
            <div className="mt-4">
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-sm font-medium text-neutral-700 hover:text-neutral-900"
              >
                {expanded ? '▼' : '▶'} Technical Details
              </button>
              
              {expanded && (
                <div className="mt-3 space-y-3 text-sm">
                  {details.domain_structure && (
                    <div className="bg-white/50 p-3 rounded">
                      <strong>Domain Structure:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-neutral-700">
                        {details.domain_structure.typosquatting_risk && (
                          <li>Typosquatting detected</li>
                        )}
                        {details.domain_structure.lookalike_risk && (
                          <li>Lookalike domain pattern</li>
                        )}
                        {details.domain_structure.excessive_subdomains && (
                          <li>Excessive subdomains ({details.domain_structure.parts_count} parts)</li>
                        )}
                        {details.domain_structure.unusual_tld && (
                          <li>Unusual TLD: {details.domain_structure.tld}</li>
                        )}
                      </ul>
                    </div>
                  )}

                  {details.https_certificate && (
                    <div className="bg-white/50 p-3 rounded">
                      <strong>HTTPS & Certificate:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-neutral-700">
                        <li>HTTPS: {details.https_certificate.uses_https ? 'Yes' : 'No'}</li>
                        {details.https_certificate.suspicious_certificate && (
                          <li>⚠️ Suspicious certificate detected</li>
                        )}
                        {details.https_certificate.new_certificate && (
                          <li>New certificate (recently issued)</li>
                        )}
                      </ul>
                    </div>
                  )}

                  {details.domain_reputation && (
                    <div className="bg-white/50 p-3 rounded">
                      <strong>Domain Reputation:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-neutral-700">
                        {details.domain_reputation.new_domain_risk && (
                          <li>New domain (estimated age: {details.domain_reputation.estimated_age_days || 'unknown'} days)</li>
                        )}
                        {details.domain_reputation.suspicious_reputation && (
                          <li>⚠️ Suspicious reputation indicators</li>
                        )}
                      </ul>
                    </div>
                  )}

                  {details.redirection_behavior && (
                    <div className="bg-white/50 p-3 rounded">
                      <strong>Redirection Behavior:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-neutral-700">
                        <li>Redirects: {details.redirection_behavior.redirect_count}</li>
                        {details.redirection_behavior.excessive_redirects && (
                          <li>⚠️ Excessive redirects detected</li>
                        )}
                        {details.redirection_behavior.mismatched_final_domain && (
                          <li>⚠️ Final domain differs from initial</li>
                        )}
                        {details.redirection_behavior.final_domain && (
                          <li>Final domain: {details.redirection_behavior.final_domain}</li>
                        )}
                      </ul>
                    </div>
                  )}

                  {details.content_fingerprinting && (
                    <div className="bg-white/50 p-3 rounded">
                      <strong>Content Analysis:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-neutral-700">
                        {details.content_fingerprinting.login_form_detected && (
                          <li>⚠️ Login/credential form detected</li>
                        )}
                        {details.content_fingerprinting.payment_gateway_detected && (
                          <li>⚠️ Payment gateway detected</li>
                        )}
                        {details.content_fingerprinting.brand_impersonation_detected && (
                          <li>⚠️ Brand impersonation: {details.content_fingerprinting.impersonated_brand || 'unknown'}</li>
                        )}
                        {details.content_fingerprinting.page_title && (
                          <li>Page title: {details.content_fingerprinting.page_title}</li>
                        )}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
