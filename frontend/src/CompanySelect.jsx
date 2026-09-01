import { useState } from 'react'

const COMPANIES = [
  { value: 'TCS_NQT', label: 'TCS NQT' },
  { value: 'INFOSYS', label: 'Infosys' },
  { value: 'WIPRO', label: 'Wipro' },
]

function CompanySelect({ onStart, starting, error }) {
  const [targetCompany, setTargetCompany] = useState('TCS_NQT')
  const [candidateName, setCandidateName] = useState('')

  return (
    <div className="card">
      <h2>Aptitude Round</h2>
      <p>15 fresh, AI-generated MCQs. Pick the company style to practice for.</p>

      <div className="company-options" role="radiogroup" aria-label="Target company">
        {COMPANIES.map((company) => (
          <button
            key={company.value}
            type="button"
            className={`company-option ${targetCompany === company.value ? 'selected' : ''}`}
            aria-pressed={targetCompany === company.value}
            onClick={() => setTargetCompany(company.value)}
          >
            {company.label}
          </button>
        ))}
      </div>

      <input
        type="text"
        placeholder="Your name (optional)"
        value={candidateName}
        onChange={(e) => setCandidateName(e.target.value)}
        className="text-input"
      />

      {error && <p className="error-text">{error}</p>}

      <button
        type="button"
        className="primary-button"
        disabled={starting}
        onClick={() => onStart({ targetCompany, candidateName })}
      >
        {starting ? 'Starting…' : 'Start Aptitude Round'}
      </button>
    </div>
  )
}

export default CompanySelect
