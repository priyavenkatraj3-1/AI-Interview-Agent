import { useEffect, useRef, useState } from 'react'
import { startAptitude, submitAnswer } from './api/aptitude'

function elapsedSeconds(sinceIso) {
  if (!sinceIso) return 0
  return Math.max(0, Math.floor((Date.now() - new Date(sinceIso).getTime()) / 1000))
}

function formatClock(totalSeconds) {
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function AptitudeRound({ sessionId, onCompleted }) {
  const [state, setState] = useState({ status: 'loading' })
  const [selected, setSelected] = useState(null)
  const [feedback, setFeedback] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [tick, setTick] = useState(0)
  const roundStartedAt = useRef(null)
  // A ref (not state) so a second click fired before React re-renders the
  // disabled button is still blocked synchronously — otherwise two rapid
  // clicks can both read the pre-update `submitting` value and both fire
  // submitAnswer, with the second one silently grading the *next*
  // question's answer instead of being rejected.
  const submittingRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    startAptitude(sessionId)
      .then((data) => {
        if (cancelled) return
        if (!roundStartedAt.current) roundStartedAt.current = data.started_at
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

  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(interval)
  }, [])

  if (state.status === 'loading') return <div className="card">Loading question…</div>
  if (state.status === 'error') return <div className="card error-text">Failed to load: {state.message}</div>

  const { data } = state
  const { question } = data

  const handleSubmit = () => {
    if (selected === null || submittingRef.current) return
    submittingRef.current = true
    setSubmitting(true)
    submitAnswer(sessionId, selected)
      .then((result) => {
        setFeedback(result)
      })
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
    setSelected(null)
    setFeedback(null)
  }

  void tick // re-render trigger for the ticking timer below

  return (
    <div className="card">
      <div className="aptitude-header">
        <span>
          Question {data.current_index}/{data.total_questions}
        </span>
        <span>Score: {data.score}</span>
        <span>Difficulty: {data.difficulty}/5</span>
        <span>Question time: {formatClock(elapsedSeconds(question.presented_at))}</span>
        <span>Total time: {formatClock(elapsedSeconds(roundStartedAt.current))}</span>
      </div>

      <p className="question-topic">
        {question.topic.replace(/_/g, ' ')} · {question.pattern.replace(/_/g, ' ')}
      </p>
      <p className="question-text">{question.question}</p>

      <div className="options">
        {question.options.map((option, index) => {
          let className = 'option'
          if (feedback) {
            if (index === feedback.correct_option) className += ' correct'
            else if (index === selected) className += ' incorrect'
          } else if (index === selected) {
            className += ' selected'
          }
          return (
            <button
              key={index}
              type="button"
              className={className}
              disabled={!!feedback}
              onClick={() => setSelected(index)}
            >
              {option}
            </button>
          )
        })}
      </div>

      {feedback && (
        <div className="feedback">
          <p>{feedback.is_correct ? '✅ Correct' : '❌ Incorrect'}</p>
          <p className="explanation">{feedback.explanation}</p>
        </div>
      )}

      {!feedback ? (
        <button
          type="button"
          className="primary-button"
          disabled={selected === null || submitting}
          onClick={handleSubmit}
        >
          {submitting ? 'Submitting…' : 'Submit'}
        </button>
      ) : (
        <button type="button" className="primary-button" onClick={handleNext}>
          {feedback.completed ? 'See Result' : 'Next Question'}
        </button>
      )}
    </div>
  )
}

export default AptitudeRound
