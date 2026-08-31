import { useEffect, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function App() {
  const [health, setHealth] = useState({ state: 'loading' })

  useEffect(() => {
    let cancelled = false

    fetch(`${API_BASE_URL}/health`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        if (!cancelled) setHealth({ state: 'ok', data })
      })
      .catch((err) => {
        if (!cancelled) setHealth({ state: 'error', message: err.message })
      })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main id="center">
      <h1>AI Interview Agent</h1>
      <p>Day 1 foundation — frontend is up and talking to the backend below.</p>

      <div className={`status status-${health.state}`}>
        {health.state === 'loading' && <span>Checking backend…</span>}
        {health.state === 'ok' && (
          <span>
            Backend reachable — <code>{health.data.service}</code> ({health.data.status})
          </span>
        )}
        {health.state === 'error' && (
          <span>
            Backend unreachable at <code>{API_BASE_URL}</code> ({health.message})
          </span>
        )}
      </div>
    </main>
  )
}

export default App
