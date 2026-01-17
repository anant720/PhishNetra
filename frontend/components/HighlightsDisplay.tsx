interface HighlightsDisplayProps {
  highlights: any[]
  originalText: string
  highlightedHtml?: string
}

export function HighlightsDisplay({ highlights, originalText, highlightedHtml }: HighlightsDisplayProps) {
  if (!highlights || highlights.length === 0) {
    return (
      <div className="text-center py-12">
        <svg className="w-16 h-16 text-neutral-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <h3 className="text-lg font-medium text-neutral-900 mb-2">No Suspicious Elements Found</h3>
        <p className="text-neutral-600">
          The AI models did not detect any concerning patterns in this text.
        </p>
      </div>
    )
  }

  // Group highlights by severity
  const highSeverity = highlights.filter(h => h.severity === 'high')
  const mediumSeverity = highlights.filter(h => h.severity === 'medium')
  const lowSeverity = highlights.filter(h => h.severity === 'low')

  return (
    <div className="space-y-6">
      {/* Highlighted Text Display */}
      {highlightedHtml && (
        <div className="card">
          <h3 className="text-lg font-semibold text-neutral-900 mb-4">Highlighted Analysis</h3>
          <div
            className="text-neutral-800 leading-relaxed p-4 bg-neutral-50 rounded-lg border"
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        </div>
      )}

      {/* Highlights Breakdown */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* High Severity */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-3 h-3 bg-risk-high rounded-full"></div>
            <h4 className="text-lg font-semibold text-neutral-900">High Risk</h4>
            <span className="bg-risk-high text-white text-xs px-2 py-1 rounded-full">
              {highSeverity.length}
            </span>
          </div>

          {highSeverity.length > 0 ? (
            <div className="space-y-3">
              {highSeverity.map((highlight, index) => (
                <div key={index} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="font-medium text-red-900 mb-1">
                    "{highlight.phrase}"
                  </div>
                  <div className="text-sm text-red-700">
                    {highlight.explanation}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-neutral-500 text-sm">No high-risk elements detected</p>
          )}
        </div>

        {/* Medium Severity */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-3 h-3 bg-risk-medium rounded-full"></div>
            <h4 className="text-lg font-semibold text-neutral-900">Medium Risk</h4>
            <span className="bg-risk-medium text-white text-xs px-2 py-1 rounded-full">
              {mediumSeverity.length}
            </span>
          </div>

          {mediumSeverity.length > 0 ? (
            <div className="space-y-3">
              {mediumSeverity.map((highlight, index) => (
                <div key={index} className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <div className="font-medium text-amber-900 mb-1">
                    "{highlight.phrase}"
                  </div>
                  <div className="text-sm text-amber-700">
                    {highlight.explanation}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-neutral-500 text-sm">No medium-risk elements detected</p>
          )}
        </div>

        {/* Low Severity */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-3 h-3 bg-risk-low rounded-full"></div>
            <h4 className="text-lg font-semibold text-neutral-900">Low Risk</h4>
            <span className="bg-risk-low text-white text-xs px-2 py-1 rounded-full">
              {lowSeverity.length}
            </span>
          </div>

          {lowSeverity.length > 0 ? (
            <div className="space-y-3">
              {lowSeverity.map((highlight, index) => (
                <div key={index} className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="font-medium text-blue-900 mb-1">
                    "{highlight.phrase}"
                  </div>
                  <div className="text-sm text-blue-700">
                    {highlight.explanation}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-neutral-500 text-sm">No low-risk elements detected</p>
          )}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="card">
        <h4 className="text-lg font-semibold text-neutral-900 mb-4">Analysis Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-risk-high">{highSeverity.length}</div>
            <div className="text-sm text-neutral-600">High Risk</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-risk-medium">{mediumSeverity.length}</div>
            <div className="text-sm text-neutral-600">Medium Risk</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-risk-low">{lowSeverity.length}</div>
            <div className="text-sm text-neutral-600">Low Risk</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-neutral-900">{highlights.length}</div>
            <div className="text-sm text-neutral-600">Total</div>
          </div>
        </div>
      </div>
    </div>
  )
}