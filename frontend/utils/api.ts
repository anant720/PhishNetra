/**
 * API Configuration and Utilities
 * Handles API base URL for both development and production
 */

const getApiBaseUrl = (): string => {
  // Check for environment variable (set by Vercel or build process)
  if (typeof window !== 'undefined') {
    // Client-side: use environment variable or fallback
    return process.env.NEXT_PUBLIC_API_BASE_URL || '/api'
  }
  
  // Server-side: use environment variable or default
  return process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
}

export const API_BASE_URL = getApiBaseUrl()

export const API_ENDPOINTS = {
  analyze: `${API_BASE_URL}/api/v1/analyze`,
  health: `${API_BASE_URL}/api/v1/health`,
} as const
