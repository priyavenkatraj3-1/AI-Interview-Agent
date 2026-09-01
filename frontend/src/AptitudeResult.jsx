import { useEffect, useState } from 'react'
import { getAptitudeResult } from './api/aptitude'

function AptitudeResult({ sessionId, onRestart }) {
  const [state, setState] = useState({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    getAptitudeResult(sessionId)
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
      <h2>Aptitude Round Complete</h2>
      <p className="result-score">
        {data.score} / {data.total_questions} ({data.percentage}%)
      </p>
      <p>Final difficulty: {data.difficulty_final}/5</p>
      <p>Total time: {Math.round(data.total_time_seconds)}s</p>

      <h3>Topic breakdown</h3>
      <ul className="topic-breakdown">
        {Object.entries(data.topic_breakdown).map(([topic, stats]) => (
          <li key={topic}>
            {topic.replace(/_/g, ' ')}: {stats.correct}/{stats.total}
          </li>
        ))}
      </ul>

      <button type="button" className="primary-button" onClick={onRestart}>
        Start a new session
      </button>
    </div>
  )
}

export default AptitudeResult
