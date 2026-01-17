import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'
import toast, { Toaster } from 'react-hot-toast'
import { AnalysisForm } from './AnalysisForm'
import { AnalysisResults } from './AnalysisResults'
import { LoadingSpinner } from './LoadingSpinner'

interface AnalysisInterfaceProps {
  onAnalysisComplete?: (results: any) => void
}

export function AnalysisInterface({ onAnalysisComplete }: AnalysisInterfaceProps) {
  const [results, setResults] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  // API mutation for analysis
  const analyzeMutation = useMutation({
    mutationFn: async (text: string) => {
      const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL 
        ? `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/v1/analyze`
        : '/api/v1/analyze'
      
      const response = await axios.post(apiUrl, {
        text: text,
        include_explainability: true,
        format_type: 'full'
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 30000 // 30 second timeout
      })
      return response.data
    },
    onSuccess: (data) => {
      setResults(data)
      onAnalysisComplete?.(data)
      toast.success('Analysis completed successfully!')

      // Log high-risk detections
      if (data.risk_score > 70) {
        toast.error('High-risk content detected!', {
          duration: 6000,
        })
      }
    },
    onError: (error: any) => {
      console.error('Analysis error:', error)

      // Handle validation errors from Pydantic/FastAPI
      let errorMessage = 'Analysis failed. Please try again.'

      if (error.response?.data?.detail) {
        const detail = error.response.data.detail

        // If detail is an array of validation errors
        if (Array.isArray(detail)) {
          errorMessage = detail.map((err: any) =>
            err.msg || err.message || JSON.stringify(err)
          ).join('; ')
        }
        // If detail is a string
        else if (typeof detail === 'string') {
          errorMessage = detail
        }
        // If detail is an object (single validation error)
        else if (typeof detail === 'object' && detail.msg) {
          errorMessage = detail.msg
        }
      }

      toast.error(errorMessage)
    },
    onSettled: () => {
      setIsAnalyzing(false)
    }
  })

  const handleAnalyze = async (text: string) => {
    // Final validation before API call
    const trimmedText = text.trim()
    
    if (!trimmedText) {
      toast.error('Please enter some text to analyze')
      return
    }

    if (trimmedText.length < 3) {
      toast.error('Please enter at least 3 characters to analyze')
      return
    }

    setIsAnalyzing(true)
    setResults(null)
    
    try {
      analyzeMutation.mutate(trimmedText)
    } catch (err) {
      setIsAnalyzing(false)
      toast.error('Failed to start analysis. Please try again.')
    }
  }

  const handleClear = () => {
    setResults(null)
    setIsAnalyzing(false)
  }

  return (
    <div className="space-y-8">
      {/* Analysis Form */}
      <div className="card">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-neutral-900 mb-2">
            Text Analysis
          </h2>
          <p className="text-neutral-600">
            Enter any suspicious message, email, or text to analyze for scam patterns using our AI models.
          </p>
        </div>

        <AnalysisForm
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
          onClear={handleClear}
        />
      </div>

      {/* Loading State */}
      {isAnalyzing && (
        <div className="card">
          <div className="text-center py-12">
            <LoadingSpinner />
            <h3 className="text-lg font-semibold text-neutral-900 mt-4 mb-2">
              Analyzing with AI Models
            </h3>
            <p className="text-neutral-600">
              Running multiple deep learning models for comprehensive risk assessment...
            </p>
            <div className="mt-6 flex justify-center space-x-2">
              <div className="flex items-center space-x-2 text-sm text-neutral-500">
                <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
                <span>FastText Processing</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-neutral-500">
                <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                <span>Sentence Analysis</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-neutral-500">
                <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                <span>Context Classification</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {results && !isAnalyzing && (
        <AnalysisResults
          results={results}
          onNewAnalysis={handleClear}
        />
      )}

      {/* Sample Texts */}
      {!results && !isAnalyzing && (
        <div className="card">
          <h3 className="text-lg font-semibold text-neutral-900 mb-4">
            Try These Examples
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <button
              onClick={() => handleAnalyze("URGENT: Your account has been suspended. Click here to verify: http://banksecure-login.com")}
              className="p-4 border border-neutral-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors text-left"
            >
              <div className="font-medium text-neutral-900 mb-1">Phishing Example</div>
              <div className="text-sm text-neutral-600">Fake account suspension notice with suspicious link</div>
            </button>

            <button
              onClick={() => handleAnalyze("Congratulations! You've won $1,000,000. Send your bank details to claim your prize.")}
              className="p-4 border border-neutral-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors text-left"
            >
              <div className="font-medium text-neutral-900 mb-1">Prize Scam</div>
              <div className="text-sm text-neutral-600">Fake lottery win requesting personal information</div>
            </button>

            <button
              onClick={() => handleAnalyze("Hi mom, my phone broke and I need money for a new one. Can you send $500 to this account? Love you")}
              className="p-4 border border-neutral-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors text-left"
            >
              <div className="font-medium text-neutral-900 mb-1">Social Engineering</div>
              <div className="text-sm text-neutral-600">Impersonation scam pretending to be family</div>
            </button>

            <button
              onClick={() => handleAnalyze("Your package is delayed. Pay $25 processing fee now to expedite delivery.")}
              className="p-4 border border-neutral-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors text-left"
            >
              <div className="font-medium text-neutral-900 mb-1">Delivery Scam</div>
              <div className="text-sm text-neutral-600">Fake delivery notification with payment request</div>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}