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

export function startCoding(sessionId) {
  return request(`/api/sessions/${sessionId}/coding/start`, { method: 'POST' })
}

export function getCodingCurrent(sessionId) {
  return request(`/api/sessions/${sessionId}/coding/current`)
}

export function runCode(sessionId, code) {
  return request(`/api/sessions/${sessionId}/coding/run`, {
    method: 'POST',
    body: JSON.stringify({ code }),
  })
}

export function submitCode(sessionId, code) {
  return request(`/api/sessions/${sessionId}/coding/submit`, {
    method: 'POST',
    body: JSON.stringify({ code }),
  })
}

export function getCodingResult(sessionId) {
  return request(`/api/sessions/${sessionId}/coding/result`)
}
