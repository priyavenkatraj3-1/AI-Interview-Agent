import { useEffect, useRef, useState } from 'react'
import { runCode, startCoding, submitCode } from './api/coding'

function TestCaseResults({ summary, label }) {
  return (
    <div className="test-case-results">
      <p>
        {label}: {summary.passed_count}/{summary.total_count} passed
      </p>
      <ul className="test-case-list">
        {summary.results.map((result, index) => (
          <li key={index} className={`test-case ${result.passed ? 'passed' : 'failed'}`}>
            <p>
              Test {index + 1}: {result.passed ? 'Passed' : 'Failed'}
              {result.timed_out ? ' (timed out)' : ''}
            </p>
            <p className="test-case-io">
              Input: <code>{result.input}</code> · Expected: <code>{result.expected_output}</code>
              {result.actual_output !== null && (
                <>
                  {' '}
                  · Got: <code>{result.actual_output}</code>
                </>
              )}
            </p>
            {result.error && <p className="error-text test-case-error">Error: {result.error}</p>}
            {result.stdout && <pre className="code-block">stdout:{'\n'}{result.stdout}</pre>}
            {result.stderr && <pre className="code-block">stderr:{'\n'}{result.stderr}</pre>}
          </li>
        ))}
      </ul>
    </div>
  )
}

function CodingRound({ sessionId, onCompleted }) {
  const [state, setState] = useState({ status: 'loading' })
  const [code, setCode] = useState('')
  const [running, setRunning] = useState(false)
  const [runResult, setRunResult] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitResult, setSubmitResult] = useState(null)
  // Refs (not state) so a fast double-click on Run/Submit fired before
  // React re-renders the disabled button is still blocked synchronously —
  // same reasoning as AptitudeRound's submittingRef.
  const runningRef = useRef(false)
  const submittingRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    startCoding(sessionId)
      .then((data) => {
        if (cancelled) return
        if (data.completed) {
          onCompleted()
          return
        }
        setState({ status: 'problem', data })
        setCode(data.problem.starter_code)
      })
      .catch((err) => {
        if (!cancelled) setState({ status: 'error', message: err.message })
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  if (state.status === 'loading') return <div className="card">Loading coding problem…</div>
  if (state.status === 'error') return <div className="card error-text">Failed to load: {state.message}</div>

  const { data } = state
  const { problem } = data

  const handleRun = () => {
    if (runningRef.current) return
    runningRef.current = true
    setRunning(true)
    setRunResult(null)
    runCode(sessionId, code)
      .then((result) => setRunResult({ status: 'ok', result }))
      .catch((err) => setRunResult({ status: 'error', message: err.message }))
      .finally(() => {
        runningRef.current = false
        setRunning(false)
      })
  }

  const handleSubmit = () => {
    if (submittingRef.current) return
    submittingRef.current = true
    setSubmitting(true)
    submitCode(sessionId, code)
      .then((result) => setSubmitResult({ status: 'ok', result }))
      .catch((err) => setSubmitResult({ status: 'error', message: err.message }))
      .finally(() => {
        submittingRef.current = false
        setSubmitting(false)
      })
  }

  const handleNext = () => {
    const { result } = submitResult
    if (result.completed) {
      onCompleted()
      return
    }
    setState({
      status: 'problem',
      data: {
        ...data,
        problem: result.next_problem,
        current_index: result.current_index,
        difficulty: result.difficulty,
        score: result.score,
      },
    })
    setCode(result.next_problem.starter_code)
    setRunResult(null)
    setSubmitResult(null)
  }

  const feedback = submitResult && submitResult.status === 'ok' ? submitResult.result : null
  const locked = !!feedback // once graded, lock this problem's editor/actions

  return (
    <div className="card coding-card">
      <div className="aptitude-header">
        <span>
          Problem {data.current_index}/{data.total_problems}
        </span>
        <span>Score: {data.score}</span>
        <span>Difficulty: {data.difficulty}/5</span>
      </div>

      <h3 className="question-text">{problem.title}</h3>
      <p className="question-topic">
        {problem.topic.replace(/_/g, ' ')} · {problem.pattern.replace(/_/g, ' ')}
      </p>
      <p>{problem.description}</p>
      <p className="coding-constraints">
        <strong>Constraints:</strong> {problem.constraints}
      </p>

      {problem.examples.length > 0 && (
        <div className="coding-examples">
          <strong>Examples</strong>
          {problem.examples.map((example, index) => (
            <pre key={index} className="code-block">
              {`Input: ${example.input}\nOutput: ${example.output}`}
              {example.explanation ? `\nExplanation: ${example.explanation}` : ''}
            </pre>
          ))}
        </div>
      )}

      <label className="coding-editor-label" htmlFor="coding-editor">
        Implement <code>{problem.function_name}</code> (Python)
      </label>
      <textarea
        id="coding-editor"
        className="code-editor"
        spellCheck={false}
        rows={14}
        value={code}
        disabled={locked}
        onChange={(e) => setCode(e.target.value)}
      />

      <div className="coding-actions">
        <button type="button" className="secondary-button" disabled={running || locked} onClick={handleRun}>
          {running ? 'Running…' : 'Run Code'}
        </button>
        <button type="button" className="primary-button" disabled={submitting || locked} onClick={handleSubmit}>
          {submitting ? 'Submitting…' : 'Submit'}
        </button>
      </div>

      {runResult && (
        <div className="coding-run-result">
          {runResult.status === 'error' ? (
            <p className="error-text">Run failed: {runResult.message}</p>
          ) : (
            <TestCaseResults summary={runResult.result} label="Sample test results" />
          )}
        </div>
      )}

      {submitResult && (
        <div className="coding-submit-result">
          {submitResult.status === 'error' ? (
            <p className="error-text">Submit failed: {submitResult.message}</p>
          ) : (
            <>
              <p>{feedback.is_correct ? '✅ All hidden tests passed' : '❌ Some hidden tests failed'}</p>
              <TestCaseResults summary={feedback} label="Hidden test results" />
            </>
          )}
        </div>
      )}

      {feedback && (
        <button type="button" className="primary-button" onClick={handleNext}>
          {feedback.completed ? 'See Result' : 'Next Problem'}
        </button>
      )}
    </div>
  )
}

export default CodingRound
