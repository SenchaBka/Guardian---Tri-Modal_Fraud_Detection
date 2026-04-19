import { useState, useEffect } from 'react'

const EXAMPLES = [
  { label: 'Typosquatting', merchant: 'Amaz0n Electronics',  narrative: 'online purchase' },
  { label: 'Phishing',      merchant: 'PayPa1 Checkout',     narrative: 'urgent wire transfer needed' },
  { label: 'Legitimate',    merchant: 'Amazon Canada',        narrative: 'online shopping' },
  { label: 'Crypto scam',   merchant: 'BitCoinExpress',       narrative: 'buy bitcoin prize winner claim' },
]

const TYPO_PATTERNS = [/amaz[o0]n/i,/paypa[l1]/i,/app[l1]e/i,/g[o0]{2}gle/i,/micr[o0]soft/i]
const RISK_WORDS    = ['urgent','wire','bitcoin','crypto','prize','winner','lottery','transfer','offshore']

function HighlightedText({ text, typoMatches, riskWords }) {
  if (!text) return null
  const all = [
    ...(typoMatches||[]).map(m => ({ match:m, type:'typo' })),
    ...(riskWords||[]).map(m => ({ match:m, type:'risk' })),
  ]
  if (!all.length) return <span className="font-mono text-sm text-ink">{text}</span>
  const regex = new RegExp(`(${all.map(p=>p.match.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')})`, 'gi')
  return (
    <span className="font-mono text-sm text-ink leading-relaxed">
      {text.split(regex).map((part, i) => {
        const isTypo = (typoMatches||[]).some(m => m.toLowerCase()===part.toLowerCase())
        const isRisk = (riskWords||[]).some(m => m.toLowerCase()===part.toLowerCase())
        if (isTypo) return <span key={i} className="bg-danger/20 text-danger border-b border-danger font-bold">{part}</span>
        if (isRisk) return <span key={i} className="bg-amber/20 text-amber border-b border-amber">{part}</span>
        return <span key={i}>{part}</span>
      })}
    </span>
  )
}

// compute highlights without running full simulation
function getHighlights(merchant, narrative) {
  const all = `${merchant} ${narrative}`.toLowerCase()
  const typoMatches = []
  const riskWordMatches = []
  for (const pat of TYPO_PATTERNS) {
    if (pat.test(merchant)) { const m = merchant.match(pat); if(m) typoMatches.push(m[0]) }
  }
  for (const word of RISK_WORDS) {
    if (all.includes(word)) riskWordMatches.push(word)
  }
  return { typoMatches, riskWords: riskWordMatches }
}

export default function NLPStreamPanel({ enabled, onChange, result }) {
  const [merchant,  setMerchant]  = useState('')
  const [narrative, setNarrative] = useState('')

  useEffect(() => {
    if (!enabled) { onChange?.(null); return }
    if (!merchant && !narrative) { onChange?.(null); return }
    // pass raw fields up — parent simulates on button click
    onChange?.({ merchant, narrative })
  }, [merchant, narrative, enabled])

  const highlights = (merchant || narrative) ? getHighlights(merchant, narrative) : null

  if (!enabled) {
    return (
      <div className="flex flex-col items-center justify-center py-8 gap-2 text-center">
        <div className="w-10 h-10 rounded-lg bg-bg-3 grid place-items-center opacity-40">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-ink-muted">
            <rect x="2" y="4" width="12" height="2" rx="1" fill="currentColor"/>
            <rect x="2" y="7.5" width="8" height="2" rx="1" fill="currentColor" opacity=".5"/>
            <rect x="2" y="11" width="10" height="2" rx="1" fill="currentColor" opacity=".3"/>
          </svg>
        </div>
        <p className="text-[10px] font-mono text-ink-dim">NLP unavailable for this channel</p>
        <p className="text-[9px] text-ink-dim opacity-60">Switch to Online, Phone, or In Person</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">

      {/* Quick presets */}
      <div>
        <label className="form-label">Quick scenarios</label>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map(e => (
            <button key={e.label}
              onClick={() => { setMerchant(e.merchant); setNarrative(e.narrative) }}
              className="text-[10px] font-mono px-2.5 py-1 rounded border border-border bg-bg-3 text-ink-muted hover:text-ink hover:border-border-strong transition-colors">
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="form-label">Merchant name / text</label>
        <input className="form-input font-mono" placeholder='e.g. "Amaz0n Electronics"'
          value={merchant} onChange={e => setMerchant(e.target.value)} />
      </div>

      <div>
        <label className="form-label">Transaction narrative</label>
        <input className="form-input font-mono" placeholder='e.g. "urgent wire transfer"'
          value={narrative} onChange={e => setNarrative(e.target.value)} />
      </div>

      {/* Text highlighting — always live (visual only, no scores) */}
      {highlights && (merchant || narrative) && (
        <div className="border border-border rounded-lg overflow-hidden">
          <div className="px-3 py-2 bg-bg-3 border-b border-border flex items-center justify-between">
            <span className="text-[9px] font-mono uppercase tracking-wider text-ink-muted">
              Text preview
            </span>
            {highlights.typoMatches.length > 0
              ? <span className="text-[9px] font-mono text-danger">typo detected</span>
              : highlights.riskWords.length > 0
                ? <span className="text-[9px] font-mono text-amber">risk language</span>
                : <span className="text-[9px] font-mono text-teal">clean</span>}
          </div>
          <div className="p-2.5 space-y-1.5">
            {merchant && (
              <div className="bg-bg-3 rounded p-2.5 border border-border">
                <HighlightedText text={merchant} typoMatches={highlights.typoMatches} riskWords={highlights.riskWords} />
              </div>
            )}
            {narrative && (
              <div className="bg-bg-3 rounded p-2.5 border border-border">
                <HighlightedText text={narrative} typoMatches={[]} riskWords={highlights.riskWords} />
              </div>
            )}
            <div className="flex gap-3 text-[9px] font-mono text-ink-dim">
              <span><span className="inline-block w-2 h-2 bg-danger/30 rounded-sm mr-1"/>typosquatting</span>
              <span><span className="inline-block w-2 h-2 bg-amber/30 rounded-sm mr-1"/>risk language</span>
            </div>
          </div>
        </div>
      )}

      {/* Scores — only shown after Analyze is clicked */}
      {result && (
        <div className="border border-border rounded-lg overflow-hidden animate-fade-in">
          <div className="px-3 py-2 bg-bg-3 border-b border-border flex items-center justify-between">
            <span className="text-[9px] font-mono uppercase tracking-wider text-ink-muted">
              Luis · FinBERT + spaCy NER
            </span>
            <span className="text-[9px] font-mono" style={{
              color: result.fraud_probability>=0.71?'#ff4757':result.fraud_probability>=0.31?'#f5a623':'#00e5c3'
            }}>
              {(result.fraud_probability*100).toFixed(1)}% fraud prob.
            </span>
          </div>
          <div className="p-3 space-y-2">
            <div>
              <div className="flex justify-between text-[10px] font-mono mb-1">
                <span className="text-ink-muted">Fraud probability</span>
                <span style={{ color:result.fraud_probability>=0.71?'#ff4757':result.fraud_probability>=0.31?'#f5a623':'#00e5c3' }}>
                  {(result.fraud_probability*100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-bg-4 rounded overflow-hidden">
                <div className="h-full rounded transition-all duration-700"
                  style={{ width:`${(result.fraud_probability*100).toFixed(1)}%`,
                    background:result.fraud_probability>=0.71?'#ff4757':result.fraud_probability>=0.31?'#f5a623':'#00e5c3' }} />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(result.signals).map(([k,v]) => (
                <div key={k} className="bg-bg-3 rounded p-2 text-center">
                  <div className="text-[9px] font-mono text-ink-dim mb-1">{k.replace(/_/g,' ').replace('risk','').trim()}</div>
                  <div className="font-mono text-xs font-medium" style={{ color:v>=0.6?'#ff4757':v>=0.3?'#f5a623':'#6b7494' }}>
                    {(v*100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
            {result.flags.filter(Boolean).map((f,i) => (
              <div key={i} className={`text-[10px] font-mono px-2.5 py-1.5 rounded border
                ${f.severity==='high'?'text-danger bg-danger-dim border-danger/30':'text-amber bg-amber-dim border-amber/30'}`}>
                <span className="font-medium">{f.label}:</span>{' '}
                <span className="opacity-80">{f.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}