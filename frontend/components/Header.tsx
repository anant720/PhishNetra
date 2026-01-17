import Link from 'next/link'

export function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-neutral-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="text-xl font-bold text-neutral-900">PhishNetra</span>
          </Link>

          <nav className="hidden md:flex items-center space-x-6">
            <Link href="/" className="text-neutral-600 hover:text-primary-600 transition-colors">
              Analyze
            </Link>
            <Link href="#features" className="text-neutral-600 hover:text-primary-600 transition-colors">
              Features
            </Link>
            <Link href="#about" className="text-neutral-600 hover:text-primary-600 transition-colors">
              About
            </Link>
          </nav>

          <div className="flex items-center space-x-4">
            <div className="hidden sm:flex items-center space-x-2 text-sm text-neutral-500">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>AI Models Active</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}