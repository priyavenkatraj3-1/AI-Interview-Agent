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

export function startTechnical(sessionId) {
  return request(`/api/sessions/${sessionId}/technical/start`, { method: 'POST' })
}

export function getTechnicalCurrent(sessionId) {
  return request(`/api/sessions/${sessionId}/technical/current`)
}

export function submitTechnicalAnswer(sessionId, answer) {
  return request(`/api/sessions/${sessionId}/technical/answer`, {
    method: 'POST',
    body: JSON.stringify({ answer }),
  })
}

export function getTechnicalResult(sessionId) {
  return request(`/api/sessions/${sessionId}/technical/result`)
}
