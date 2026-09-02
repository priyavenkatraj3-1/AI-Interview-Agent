const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // response body wasn't JSON — keep the generic status message
    }
    const error = new Error(detail)
    error.status = res.status
    throw error
  }
  return res.json()
}

// Idempotent: generates the final evaluation the first time, returns the
// already-generated one on any later call — safe to call on every mount,
// same as every round's start_* endpoint.
export function generateFinalEvaluation(sessionId) {
  return request(`/api/sessions/${sessionId}/final-evaluation/generate`, { method: 'POST' })
}

export function getFinalEvaluation(sessionId) {
  return request(`/api/sessions/${sessionId}/final-evaluation`)
}
