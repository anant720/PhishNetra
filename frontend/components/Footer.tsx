export function Footer() {
  return (
    <footer className="bg-neutral-50 border-t border-neutral-200 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center space-x-3 mb-4 md:mb-0">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="text-xl font-bold text-neutral-900">PhishNetra</span>
          </div>

          <div className="flex items-center space-x-6 text-sm text-neutral-600">
            <span>Advanced AI-powered scam detection</span>
            <span>•</span>
            <span>Multiple deep learning architectures</span>
            <span>•</span>
            <span>Real-time analysis</span>
          </div>
        </div>
      </div>
    </footer>
  )
}