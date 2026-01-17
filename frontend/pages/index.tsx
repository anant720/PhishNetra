import { useState } from 'react'
import Head from 'next/head'
import { AnalysisInterface } from '@/components/AnalysisInterface'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'

export default function Home() {
  const [currentAnalysis, setCurrentAnalysis] = useState(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-primary-50">
      <Head>
        <title>PhishNetra - Advanced Scam Detection</title>
        <meta name="description" content="AI-powered scam detection system using multiple deep learning models" />
        <meta name="keywords" content="scam detection, AI, fraud analysis, cybersecurity, risk analysis" />
      </Head>

      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12 animate-fade-in">
            <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-4">
              Phish
              <span className="text-primary-600">Netra</span>
            </h1>
            <p className="text-xl text-neutral-600 max-w-2xl mx-auto leading-relaxed">
              Advanced AI-powered scam detection using multiple deep learning architectures.
              Understand intent, detect manipulation, and protect against new scam patterns.
            </p>
          </div>

          {/* Analysis Interface */}
          <div className="animate-slide-up">
            <AnalysisInterface
              onAnalysisComplete={setCurrentAnalysis}
            />
          </div>

          {/* Features Section */}
          <div className="mt-16 grid md:grid-cols-3 gap-8">
            <div className="card text-center">
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-neutral-900 mb-2">Multi-Model AI</h3>
              <p className="text-neutral-600">
                Combines FastText, Sentence Transformers, DistilBERT, and similarity search for comprehensive analysis.
              </p>
            </div>

            <div className="card text-center">
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-neutral-900 mb-2">Real-time Analysis</h3>
              <p className="text-neutral-600">
                Instant risk assessment with detailed explanations and highlighted suspicious content.
              </p>
            </div>

            <div className="card text-center">
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-neutral-900 mb-2">High Accuracy</h3>
              <p className="text-neutral-600">
                Advanced ensemble methods with explainable AI for trustworthy risk assessments.
              </p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}