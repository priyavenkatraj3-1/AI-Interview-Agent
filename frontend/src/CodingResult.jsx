import { useEffect, useState } from 'react'
import { getCodingResult } from './api/coding'

function CodingResult({ sessionId, onRestart }) {
  const [state, setState] = useState({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    getCodingResult(sessionId)
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((err) => {
        if (!cancelled) setState({ status: 'error', message: err.message })
      })
    return () => {
      cancelled = true
    }
  }, [sessionId])

  if (state.status === 'loading') return <div className="card">Loading result…</div>
  if (state.status === 'error') return <div className="card error-text">Failed to load result: {state.message}</div>

  const { data } = state

  return (
    <div className="card">
      <h2>Coding Round Complete</h2>
      <p className="result-score">
        {data.score} / {data.total_problems} ({data.percentage}%)
      </p>
      <p>Final difficulty: {data.difficulty_final}/5</p>
      <p>Total time: {Math.round(data.total_time_seconds)}s</p>

      <h3>Problems</h3>
      <ul className="topic-breakdown">
        {data.history.map((entry) => (
          <li key={entry.index}>
            {entry.title} ({entry.topic.replace(/_/g, ' ')}): {entry.passed_count}/{entry.total_count} tests passed
            {entry.is_correct ? ' ✅' : ' ❌'}
          </li>
        ))}
      </ul>

      <p className="coding-next-note">Technical round is coming soon.</p>

      <button type="button" className="primary-button" onClick={onRestart}>
        Start a new session
      </button>
    </div>
  )
}

export default CodingResult
