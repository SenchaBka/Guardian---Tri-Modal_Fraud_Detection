import { useState } from 'react'
import StatCard     from '../components/StatCard'
import ResultPanel  from '../components/ResultPanel'
import HistoryTable from '../components/HistoryTable'
import { simulateNumerical, simulateNLP, simulateVoice, simulateFusion } from '../api/fusionApi'

const CHANNELS = [
  { value:'online',    label:'Online',      streams:{ nlp:true,  voice:false } },
  { value:'phone',     label:'Phone Call',  streams:{ nlp:true,  voice:true  } },
  { value:'in_person', label:'In Person',   streams:{ nlp:true,  voice:true  } },
  { value:'atm',       label:'ATM',         streams:{ nlp:false, voice:false } },
]

const TYPO_PATTERNS = [/amaz[o0]n/i,/paypa[l1]/i,/app[l1]e/i,/g[o0]{2}gle/i,/micr[o0]soft/i]
const RISK_WORDS    = ['urgent','wire','bitcoin','crypto','prize','winner','lottery','transfer','offshore']

const SAMPLES = [
  { label:'Deepfake A',  name:'deepfake_call_001.wav',    isDeepfake:true  },
  { label:'Deepfake B',  name:'synthetic_voice_002.wav',  isDeepfake:true  },
  { label:'Genuine A',   name:'genuine_call_003.wav',     isDeepfake:false },
  { label:'Genuine B',   name:'bonafide_speaker_004.wav', isDeepfake:false },
]

function HighlightedText({ text }) {
  if (!text) return null
  const typoM=[], riskM=[]
  for (const p of TYPO_PATTERNS) { const m=text.match(p); if(m) typoM.push(m[0]) }
  for (const w of RISK_WORDS) { if(text.toLowerCase().includes(w)) riskM.push(w) }
  if (!typoM.length && !riskM.length) return <span className="font-mono text-xs text-ink">{text}</span>
  const regex = new RegExp(`(${[...typoM,...riskM].map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')})`, 'gi')
  return (
    <span className="font-mono text-xs text-ink">
      {text.split(regex).map((p,i) => {
        if (typoM.some(m=>m.toLowerCase()===p.toLowerCase())) return <span key={i} className="bg-danger/20 text-danger border-b border-danger font-bold">{p}</span>
        if (riskM.some(m=>m.toLowerCase()===p.toLowerCase())) return <span key={i} className="bg-amber/20 text-amber border-b border-amber">{p}</span>
        return <span key={i}>{p}</span>
      })}
    </span>
  )
}

function StreamRow({ icon, name, owner, required, enabled, onToggle, children }) {
  return (
    <div className={`rounded-lg border transition-colors ${enabled ? 'border-teal/20 bg-bg-3' : 'border-border bg-bg-3'}`}>
      <div className="flex items-center gap-2.5 px-3 py-2.5">
        <div className={`w-6 h-6 rounded flex-shrink-0 grid place-items-center
          ${name==='Numerical'?'bg-sky-dim':name==='NLP'?'bg-teal-dim':'bg-amber-dim'}`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium">{name}</p>
          <p className="text-[9px] font-mono text-ink-muted">{owner}</p>
        </div>
        {required ? (
          <span className="text-[9px] font-mono text-teal">ALWAYS ON</span>
        ) : (
          <label className="relative w-8 h-[18px] cursor-pointer flex-shrink-0">
            <input type="checkbox" className="sr-only" checked={enabled} onChange={onToggle}/>
            <div className={`absolute inset-0 rounded-full border transition-all ${enabled?'bg-teal border-teal':'bg-bg-4 border-border'}`}/>
            <div className={`absolute top-0.5 w-3 h-3 rounded-full transition-all ${enabled?'left-[17px] bg-bg-0':'left-0.5 bg-ink-muted'}`}/>
          </label>
        )}
      </div>
      {/* Only show children when enabled */}
      {enabled && children && (
        <div className="px-3 pb-3 border-t border-border pt-2.5 space-y-2">
          {children}
        </div>
      )}
    </div>
  )
}

export default function AnalysisPage({ stats, onNewResult, onModeChange }) {
  const [txnId,    setTxnId]    = useState('TXN-2024-001')
  const [amount,   setAmount]   = useState('4850')
  const [channel,  setChannel]  = useState('online')
  const [country,  setCountry]  = useState('US')
  const [category, setCategory] = useState('retail')
  const [merchant, setMerchant] = useState('Amazon Purchase')
  const [narrative,setNarrative]= useState('')
  const [voiceSample,setVoiceSample] = useState(null)
  const [nlpEnabled,  setNlpEnabled]   = useState(true)
  const [voiceEnabled,setVoiceEnabled] = useState(false)
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [analyzed, setAnalyzed] = useState(false)
  const [history,  setHistory]  = useState([])

  const handleChannelChange = val => {
    setChannel(val)
    const ch = CHANNELS.find(c=>c.value===val)
    setNlpEnabled(ch.streams.nlp)
    setVoiceEnabled(ch.streams.voice)
    if (!ch.streams.voice) setVoiceSample(null)
    setResult(null); setAnalyzed(false)
    onModeChange?.(null)
  }

  const handleAnalyze = async () => {
    if (!amount) return
    setLoading(true)
    await new Promise(r=>setTimeout(r,700))
    const txnData = { amount, currency:'USD', channel, country, merchant_category:category }
    const num   = simulateNumerical(txnData)
    const nlp   = (nlpEnabled && merchant) ? simulateNLP(merchant, narrative) : null
    const voice = (voiceEnabled && voiceSample) ? simulateVoice(voiceSample.name, voiceSample.isDeepfake) : null
    const fusion = simulateFusion(num, nlp, voice)
    setResult(fusion)
    setAnalyzed(true)
    setLoading(false)
    onNewResult?.(fusion)
    onModeChange?.(fusion.ensemble_mode)
    setHistory(prev=>[
      { id:txnId||'TXN-001', amount, decision:fusion.decision, score:fusion.fraud_score, mode:fusion.ensemble_mode },
      ...prev
    ].slice(0,50))
  }

  const handleClear = () => {
    setResult(null); setAnalyzed(false)
    setAmount(''); setMerchant(''); setNarrative(''); setVoiceSample(null)
    onModeChange?.(null)
  }

  return (
    <div className="h-full flex overflow-hidden">

      {/* ── LEFT ── */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 min-w-0">

        {/* Stat cards */}
        <div className="grid grid-cols-4 gap-3 flex-shrink-0">
          <StatCard label="Analyzed" value={stats.total}    color="sky"    />
          <StatCard label="Approved" value={stats.approved} color="teal"   />
          <StatCard label="Review"   value={stats.review}   color="amber"  />
          <StatCard label="Blocked"  value={stats.blocked}  color="danger" />
        </div>

        {/* Analyze form */}
        <div className="card flex-shrink-0">
          <div className="card-header">
            <span className="card-title">Analyze Transaction</span>
            <span className="tag tag-teal">Fusion Layer</span>
          </div>
          <div className="card-body space-y-4">

            {/* Transaction fields */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="form-label">Transaction ID</label>
                <input className="form-input" value={txnId} onChange={e=>setTxnId(e.target.value)} placeholder="TXN-001"/>
              </div>
              <div>
                <label className="form-label">Amount (USD)</label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted text-xs font-mono">$</span>
                  <input className="form-input pl-6" type="number" value={amount}
                    onChange={e=>{ setAmount(e.target.value); setResult(null); setAnalyzed(false); onModeChange?.(null) }}
                    placeholder="0.00"/>
                </div>
              </div>
              <div>
                <label className="form-label">Channel</label>
                <select className="form-select" value={channel} onChange={e=>handleChannelChange(e.target.value)}>
                  {CHANNELS.map(c=><option key={c.value} value={c.value}>{c.label}</option>)}
                </select>
              </div>
              <div>
                <label className="form-label">Country</label>
                <input className="form-input font-mono uppercase" value={country}
                  onChange={e=>setCountry(e.target.value)} placeholder="US" maxLength={3}/>
              </div>
              <div>
                <label className="form-label">Merchant Category</label>
                <select className="form-select" value={category} onChange={e=>setCategory(e.target.value)}>
                  {['retail','electronics','gambling','crypto','wire_transfer','cash_advance','forex','restaurants','travel','healthcare']
                    .map(c=><option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="form-label">Merchant Text</label>
                <input className="form-input font-mono" value={merchant}
                  onChange={e=>setMerchant(e.target.value)} placeholder="merchant name"/>
                {merchant && nlpEnabled && (
                  <div className="mt-1.5 px-2 py-1.5 bg-bg-3 rounded border border-border">
                    <HighlightedText text={merchant}/>
                  </div>
                )}
              </div>
            </div>

            {/* Stream configuration */}
            <div>
              <p className="form-label mb-2">Stream Configuration</p>
              <div className="space-y-2">

                {/* Numerical */}
                <StreamRow name="Numerical" owner="Ivan · XGBoost + RF · required" required enabled
                  icon={<svg width="13" height="13" viewBox="0 0 16 16" fill="none" className="text-sky"><path d="M2 11L5.5 7l3 3.5L12 4l2 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/></svg>}
                />

                {/* NLP — expands to show quick scenarios + narrative */}
                <StreamRow name="NLP" owner="Luis · FinBERT + spaCy NER"
                  enabled={nlpEnabled} onToggle={()=>setNlpEnabled(v=>!v)}
                  icon={<svg width="13" height="13" viewBox="0 0 16 16" fill="none" className="text-teal"><rect x="2" y="4" width="12" height="2" rx="1" fill="currentColor"/><rect x="2" y="7.5" width="8" height="2" rx="1" fill="currentColor" opacity=".6"/><rect x="2" y="11" width="10" height="2" rx="1" fill="currentColor" opacity=".3"/></svg>}
                >
                  <div className="flex flex-wrap gap-1">
                    {[['Typosquatting','Amaz0n Electronics','online purchase'],
                      ['Phishing','PayPa1 Checkout','urgent wire transfer'],
                      ['Legitimate','Amazon Canada','online shopping'],
                      ['Crypto scam','BitCoinExpress','buy bitcoin winner']].map(([l,m,n])=>(
                      <button key={l} onClick={()=>{ setMerchant(m); setNarrative(n) }}
                        className="text-[9px] font-mono px-2 py-0.5 rounded border border-border bg-bg-4 text-ink-dim hover:text-ink hover:border-border-strong transition-colors">
                        {l}
                      </button>
                    ))}
                  </div>
                  <div>
                    <label className="form-label">Transaction narrative</label>
                    <input className="form-input font-mono text-xs" value={narrative}
                      onChange={e=>setNarrative(e.target.value)} placeholder='e.g. "urgent wire transfer"'/>
                  </div>
                </StreamRow>

                {/* Voice — expands to show audio selector */}
                <StreamRow name="Voice" owner="Arsenii · WavLM deepfake detection"
                  enabled={voiceEnabled} onToggle={()=>setVoiceEnabled(v=>!v)}
                  icon={<svg width="13" height="13" viewBox="0 0 16 16" fill="none" className="text-amber"><path d="M8 2C6 2 4 3.5 4 5.5C4 7.5 5 9 8 11C11 9 12 7.5 12 5.5C12 3.5 10 2 8 2Z" stroke="currentColor" strokeWidth="1.3" fill="none"/><line x1="8" y1="11" x2="8" y2="14" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>}
                >
                  <label className="form-label">Select audio sample</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    {SAMPLES.map(s=>(
                      <button key={s.name} onClick={()=>setVoiceSample(s)}
                        className={`text-[10px] font-mono px-2.5 py-1.5 rounded border text-left transition-all
                          ${voiceSample?.name===s.name
                            ? s.isDeepfake ? 'border-danger bg-danger-dim text-danger' : 'border-teal bg-teal-dim text-teal'
                            : 'border-border bg-bg-4 text-ink-muted hover:border-border-strong hover:text-ink'}`}>
                        <div className="font-medium truncate">{s.label}</div>
                        <div className="text-[9px] opacity-60 truncate mt-0.5">{s.name}</div>
                      </button>
                    ))}
                  </div>
                  {voiceSample && (
                    <p className="text-[9px] font-mono text-ink-dim">
                      Selected: <span className={voiceSample.isDeepfake?'text-danger':'text-teal'}>{voiceSample.name}</span>
                    </p>
                  )}
                </StreamRow>

              </div>
            </div>

            {/* Buttons */}
            <div className="flex items-center gap-2">
              <button onClick={handleAnalyze} disabled={!amount||loading}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md
                  font-display text-xs font-bold uppercase tracking-wider transition-all
                  ${amount&&!loading ? 'bg-teal text-bg-0 hover:brightness-110 cursor-pointer' : 'bg-bg-3 text-ink-dim cursor-not-allowed opacity-50'}`}>
                {loading
                  ? <><div className="w-3.5 h-3.5 border-2 border-bg-0/30 border-t-bg-0 rounded-full animate-spin-slow"/>Analyzing…</>
                  : <><svg width="11" height="11" viewBox="0 0 16 16" fill="none"><polygon points="4,2 14,8 4,14" fill="currentColor"/></svg>Run Fusion Analysis</>}
              </button>
              {analyzed && (
                <button onClick={handleClear}
                  className="text-[10px] font-mono px-3 py-2.5 rounded-md border border-border bg-bg-3 text-ink-muted hover:text-ink hover:border-border-strong transition-colors">
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Recent analyses */}
        <div className="card flex-1 min-h-0">
          <div className="card-header">
            <span className="card-title">Recent Analyses</span>
            <span className="tag tag-dim">{history.length} transaction{history.length!==1?'s':''}</span>
          </div>
          <HistoryTable history={history}/>
        </div>

      </div>

      {/* ── RIGHT: result panel ── */}
      <div className="w-80 flex-shrink-0 border-l border-border bg-bg-1 overflow-y-auto">
        <ResultPanel result={result} loading={loading}/>
      </div>

    </div>
  )
}