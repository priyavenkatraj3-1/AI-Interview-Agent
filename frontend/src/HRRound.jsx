import { useEffect, useRef, useState } from 'react'
import { startHR, submitHRAnswer } from './api/hr'

function HRRound({ sessionId, onCompleted }) {
  const [state, setState] = useState({ status: 'loading' })
  const [answer, setAnswer] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [feedback, setFeedback] = useState(null)
  // A ref (not state) so a second click fired before React re-renders the
  // disabled button is still blocked synchronously — same reasoning as
  // AptitudeRound/CodingRound/TechnicalRound's submittingRef.
  const submittingRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    startHR(sessionId)
      .then((data) => {
        if (cancelled) return
        if (data.completed) {
          onCompleted()
          return
        }
        setState({ status: 'question', data })
      })
      .catch((err) => {
        if (!cancelled) setState({ status: 'error', message: err.message })
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  if (state.status === 'loading') return <div className="card">Loading question…</div>
  if (state.status === 'error') return <div className="card error-text">Failed to load: {state.message}</div>

  const { data } = state
  const { question } = data

  const handleSubmit = () => {
    if (!answer.trim() || submittingRef.current) return
    submittingRef.current = true
    setSubmitting(true)
    submitHRAnswer(sessionId, answer)
      .then((result) => setFeedback(result))
      .catch((err) => setState({ status: 'error', message: err.message }))
      .finally(() => {
        submittingRef.current = false
        setSubmitting(false)
      })
  }

  const handleNext = () => {
    if (feedback.completed) {
      onCompleted()
      return
    }
    setState({
      status: 'question',
      data: {
        ...data,
        question: feedback.next_question,
        current_index: feedback.current_index,
        difficulty: feedback.difficulty,
        score: feedback.score,
      },
    })
    setAnswer('')
    setFeedback(null)
  }

  return (
    <div className="card">
      <div className="aptitude-header">
        <span>
          Question {data.current_index}/{data.total_questions}
        </span>
        <span>Score: {data.score}</span>
        <span>Difficulty: {data.difficulty}/5</span>
      </div>

      <p className="question-topic">
        {question.topic.replace(/_/g, ' ')} · {question.pattern.replace(/_/g, ' ')}
      </p>
      <p className="question-text">{question.question}</p>

      <label className="coding-editor-label" htmlFor="hr-answer">
        Your answer
      </label>
      <textarea
        id="hr-answer"
        className="text-input technical-answer"
        rows={8}
        value={answer}
        disabled={!!feedback}
        onChange={(e) => setAnswer(e.target.value)}
        placeholder="Type your answer here…"
      />

      {feedback && (
        <div className="feedback">
          <p>
            {feedback.is_correct ? '✅ Strong answer' : '❌ Needs improvement'} · Score: {feedback.answer_score}/100
          </p>
          <p className="explanation">{feedback.feedback}</p>
        </div>
      )}

      {!feedback ? (
        <button
          type="button"
          className="primary-button"
          disabled={!answer.trim() || submitting}
          onClick={handleSubmit}
        >
          {submitting ? 'Submitting…' : 'Submit Answer'}
        </button>
      ) : (
        <button type="button" className="primary-button" onClick={handleNext}>
          {feedback.completed ? 'See Result' : 'Next Question'}
        </button>
      )}
    </div>
  )
}

export default HRRound
