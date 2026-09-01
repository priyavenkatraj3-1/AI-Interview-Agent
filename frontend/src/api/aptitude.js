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

export function createSession({ candidateName, targetCompany }) {
  return request('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ candidate_name: candidateName || null, target_company: targetCompany }),
  })
}

export function getSession(sessionId) {
  return request(`/api/sessions/${sessionId}`)
}

export function startAptitude(sessionId) {
  return request(`/api/sessions/${sessionId}/aptitude/start`, { method: 'POST' })
}

export function getCurrentQuestion(sessionId) {
  return request(`/api/sessions/${sessionId}/aptitude/current`)
}

export function submitAnswer(sessionId, selectedOption) {
  return request(`/api/sessions/${sessionId}/aptitude/answer`, {
    method: 'POST',
    body: JSON.stringify({ selected_option: selectedOption }),
  })
}

export function getAptitudeResult(sessionId) {
  return request(`/api/sessions/${sessionId}/aptitude/result`)
}
