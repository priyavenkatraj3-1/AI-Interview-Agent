import { useEffect, useState } from 'react'
import './App.css'
import AptitudeResult from './AptitudeResult'
import AptitudeRound from './AptitudeRound'
import CodingResult from './CodingResult'
import CodingRound from './CodingRound'
import CompanySelect from './CompanySelect'
import FinalEvaluation from './FinalEvaluation'
import HRResult from './HRResult'
import HRRound from './HRRound'
import TechnicalResult from './TechnicalResult'
import TechnicalRound from './TechnicalRound'
import { createSession, getSession } from './api/aptitude'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const SESSION_STORAGE_KEY = 'aptitude_session_id'

function App() {
  const [health, setHealth] = useState({ state: 'loading' })
  const [view, setView] = useState(() => {
    const storedId = localStorage.getItem(SESSION_STORAGE_KEY)
    return storedId ? { status: 'resuming', sessionId: storedId } : { status: 'landing' }
  })
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState(null)

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

  // Resume an in-progress or finished aptitude session after a browser
  // refresh — the source of truth is the backend (InterviewSession /
  // StageProgress), localStorage only remembers *which* session to ask for.
  useEffect(() => {
    if (view.status !== 'resuming') return undefined
    let cancelled = false

    getSession(view.sessionId)
      .then((session) => {
        if (cancelled) return
        if (session.current_stage === 'aptitude') {
          setView({ status: 'aptitude', sessionId: session.id })
        } else if (session.current_stage === 'coding') {
          setView({ status: 'coding', sessionId: session.id })
        } else if (session.current_stage === 'technical') {
          setView({ status: 'technical', sessionId: session.id })
        } else if (session.current_stage === 'hr') {
          setView({ status: 'hr', sessionId: session.id })
        } else {
          // completed: all four rounds are done, so the final evaluation
          // is the resume landing spot.
          setView({ status: 'final-evaluation', sessionId: session.id })
        }
      })
      .catch(() => {
        if (cancelled) return
        localStorage.removeItem(SESSION_STORAGE_KEY)
        setView({ status: 'landing' })
      })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view.status])

  const handleStart = ({ targetCompany, candidateName }) => {
    setStarting(true)
    setStartError(null)
    createSession({ targetCompany, candidateName })
      .then((session) => {
        localStorage.setItem(SESSION_STORAGE_KEY, session.id)
        setView({ status: 'aptitude', sessionId: session.id })
      })
      .catch((err) => setStartError(err.message))
      .finally(() => setStarting(false))
  }

  const handleRestart = () => {
    localStorage.removeItem(SESSION_STORAGE_KEY)
    setView({ status: 'landing' })
  }

  return (
    <main id="center">
      <h1>AI Interview Agent</h1>

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

      {view.status === 'resuming' && <div className="card">Loading…</div>}
      {view.status === 'landing' && <CompanySelect onStart={handleStart} starting={starting} error={startError} />}
      {view.status === 'aptitude' && (
        <AptitudeRound
          sessionId={view.sessionId}
          onCompleted={() => setView({ status: 'coding', sessionId: view.sessionId })}
        />
      )}
      {view.status === 'coding' && (
        <CodingRound
          sessionId={view.sessionId}
          onCompleted={() => setView({ status: 'technical', sessionId: view.sessionId })}
        />
      )}
      {view.status === 'technical' && (
        <TechnicalRound
          sessionId={view.sessionId}
          onCompleted={() => setView({ status: 'hr', sessionId: view.sessionId })}
        />
      )}
      {view.status === 'hr' && (
        <HRRound
          sessionId={view.sessionId}
          onCompleted={() => setView({ status: 'final-evaluation', sessionId: view.sessionId })}
        />
      )}
      {view.status === 'final-evaluation' && (
        <FinalEvaluation sessionId={view.sessionId} onRestart={handleRestart} />
      )}
      {view.status === 'hr-result' && <HRResult sessionId={view.sessionId} onRestart={handleRestart} />}
      {view.status === 'technical-result' && (
        <TechnicalResult sessionId={view.sessionId} onRestart={handleRestart} />
      )}
      {view.status === 'coding-result' && <CodingResult sessionId={view.sessionId} onRestart={handleRestart} />}
      {view.status === 'result' && <AptitudeResult sessionId={view.sessionId} onRestart={handleRestart} />}
    </main>
  )
}

export default App
