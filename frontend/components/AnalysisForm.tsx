import { useState } from 'react'

interface AnalysisFormProps {
  onAnalyze: (text: string) => void
  isAnalyzing: boolean
  onClear: () => void
}

export function AnalysisForm({ onAnalyze, isAnalyzing, onClear }: AnalysisFormProps) {
  const [text, setText] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('') // Clear previous errors

    const trimmedText = text.trim()

    // Client-side validation
    if (!trimmedText) {
      setError('Please enter some text to analyze')
      return
    }

    // Additional validation for very short text
    if (trimmedText.length < 3) {
      setError('Please enter at least 3 characters to analyze')
      return
    }

    // If validation passes, call the analyze function
    onAnalyze(trimmedText)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      const file = files[0]
      if (file.type === 'text/plain') {
        const reader = new FileReader()
        reader.onload = (e) => {
          const content = e.target?.result as string
          setText(content)
        }
        reader.readAsText(file)
      }
    }
  }

  const handlePaste = (e: React.ClipboardEvent) => {
    const pastedText = e.clipboardData.getData('text')
    if (pastedText) {
      setText(prev => prev + pastedText)
    }
  }

  const characterCount = text.length
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div
        className={`relative ${
          isDragging
            ? 'border-primary-400 bg-primary-50'
            : 'border-neutral-300 hover:border-neutral-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <textarea
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setError('') // Clear error when user types
          }}
          onPaste={handlePaste}
          placeholder="Paste suspicious text, email, or message here for AI-powered scam analysis...

Examples of what to analyze:
• Email claiming your account is suspended
• SMS with urgent payment requests
• Messages containing suspicious links
• Texts asking for personal information
• Offers that seem too good to be true"

          className={`textarea w-full min-h-[200px] border-2 focus:ring-0 resize-none ${
            error ? 'border-red-300 focus:border-red-500' : 'border-neutral-300 focus:border-primary-500'
          }`}
          disabled={isAnalyzing}
        />

        {isDragging && (
          <div className="absolute inset-0 bg-primary-100 bg-opacity-80 flex items-center justify-center rounded-lg">
            <div className="text-center">
              <svg className="w-12 h-12 text-primary-600 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-primary-700 font-medium">Drop text file here</p>
            </div>
          </div>
        )}
      </div>

      {/* Validation Error Message */}
      {error && (
        <div className="text-red-600 text-sm flex items-center space-x-1 bg-red-50 p-3 rounded-lg border border-red-200">
          <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* Character/Word Count */}
      <div className="flex justify-between items-center text-sm text-neutral-500">
        <div className="flex space-x-4">
          <span>{characterCount} characters</span>
          <span>{wordCount} words</span>
        </div>

        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setText('')}
            className="text-neutral-500 hover:text-neutral-700 px-3 py-1 rounded transition-colors"
            disabled={isAnalyzing}
          >
            Clear
          </button>
          <button
            type="button"
            onClick={onClear}
            className="text-neutral-500 hover:text-neutral-700 px-3 py-1 rounded transition-colors"
            disabled={isAnalyzing}
          >
            Reset All
          </button>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        <button
          type="submit"
          disabled={isAnalyzing || !text.trim()}
          className="btn btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              Analyze Text
            </>
          )}
        </button>

        <div className="flex flex-col sm:flex-row gap-2">
          <button
            type="button"
            onClick={() => {
              const sampleText = "Dear customer,\n\nYour account has been temporarily suspended due to suspicious activity. To restore access, please verify your identity by clicking the link below and providing your login credentials.\n\nVerify Account: https://secure-bank-login.com/verify\n\nThis is urgent and must be completed within 24 hours.\n\nThank you,\nBank Security Team"
              setText(sampleText)
            }}
            className="btn btn-outline text-sm"
            disabled={isAnalyzing}
          >
            Bank Scam
          </button>

          <button
            type="button"
            onClick={() => {
              const sampleText = "🚨 URGENT: Your package delivery failed! Pay $49.99 now to reschedule.\n\nClick here: http://delivery-fix.com/pay\n\nYour items will be returned if not paid within 2 hours."
              setText(sampleText)
            }}
            className="btn btn-outline text-sm"
            disabled={isAnalyzing}
          >
            Delivery Scam
          </button>

          <button
            type="button"
            onClick={() => {
              const sampleText = "Congratulations! You've won $1,000,000!\n\nTo claim your prize, send $500 processing fee to: bitcoin-wallet-12345\n\nContact: winner@lottery-prize.org"
              setText(sampleText)
            }}
            className="btn btn-outline text-sm"
            disabled={isAnalyzing}
          >
            Lottery Scam
          </button>

          <button
            type="button"
            onClick={() => {
              const sampleText = "IRS NOTICE: You owe $2,847 in back taxes.\n\nPay immediately or face arrest warrant.\n\nPay Now: https://irs-payment.gov/settle\n\nFailure to comply will result in legal action."
              setText(sampleText)
            }}
            className="btn btn-outline text-sm"
            disabled={isAnalyzing}
          >
            Authority Scam
          </button>
        </div>
      </div>

      {/* Help Text */}
      <div className="text-sm text-neutral-500 bg-neutral-50 p-4 rounded-lg">
        <div className="flex items-start space-x-2">
          <svg className="w-5 h-5 text-neutral-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <p className="font-medium text-neutral-700 mb-1">Tips for better analysis:</p>
            <ul className="space-y-1 text-neutral-600">
              <li>• Include the full message with links and contact information</li>
              <li>• Add context like sender information if available</li>
              <li>• Our AI can detect spelling variations and coded language</li>
            </ul>
          </div>
        </div>
      </div>
    </form>
  )
}