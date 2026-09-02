import { useEffect, useState } from 'react'
import { generateFinalEvaluation } from './api/finalEvaluation'

function FinalEvaluation({ sessionId, onRestart }) {
  const [state, setState] = useState({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    generateFinalEvaluation(sessionId)
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((err) => {
        if (cancelled) return
        // A 409 here means one of the four rounds isn't complete yet —
        // worth telling apart from a generic API/network failure.
        if (err.status === 409) {
          setState({ status: 'incomplete', message: err.message })
        } else {
          setState({ status: 'error', message: err.message })
        }
      })
    return () => {
      cancelled = true
    }
  }, [sessionId])

  if (state.status === 'loading') return <div className="card">Generating final evaluation…</div>
  if (state.status === 'incomplete') {
    return (
      <div className="card error-text">
        Final evaluation isn't available yet: {state.message}
      </div>
    )
  }
  if (state.status === 'error') {
    return <div className="card error-text">Failed to load final evaluation: {state.message}</div>
  }

  const { data } = state

  return (
    <div className="card">
      <h2>Final Evaluation</h2>
      <p className="result-score">{data.overall_score}%</p>
      <p>
        Recommendation: <strong>{data.recommendation}</strong>
      </p>

      <h3>Round Scores</h3>
      <ul className="plain-list">
        <li>
          Aptitude: {data.aptitude.score}/{data.aptitude.total} ({data.aptitude.percentage}%)
        </li>
        <li>
          Coding: {data.coding.score}/{data.coding.total} ({data.coding.percentage}%)
        </li>
        <li>
          Technical: {data.technical.score}/{data.technical.total} ({data.technical.percentage}%)
        </li>
        <li>
          HR: {data.hr.score}/{data.hr.total} ({data.hr.percentage}%)
        </li>
      </ul>

      <h3>Strengths</h3>
      <ul className="plain-list">
        {data.strengths.map((strength, index) => (
          <li key={index}>{strength}</li>
        ))}
      </ul>

      <h3>Weaknesses</h3>
      <ul className="plain-list">
        {data.weaknesses.map((weakness, index) => (
          <li key={index}>{weakness}</li>
        ))}
      </ul>

      <p className="explanation">{data.summary}</p>

      <h3>Hiring Verdict</h3>
      <p className="explanation">{data.hiring_verdict}</p>

      <h3>14-Day Remediation Plan</h3>
      <ul className="plain-list">
        {data.remediation_plan.map((entry) => (
          <li key={entry.day}>
            Day {entry.day} — {entry.focus}: {entry.action}
          </li>
        ))}
      </ul>

      <button type="button" className="primary-button" onClick={onRestart}>
        Start a new session
      </button>
    </div>
  )
}

export default FinalEvaluation
