import { useState, useEffect, useCallback } from 'react'
import NumericalStreamPanel from '../components/NumericalStreamPanel'
import NLPStreamPanel        from '../components/NLPStreamPanel'
import VoiceStreamPanel      from '../components/VoiceStreamPanel'
import FusionDecision        from '../components/FusionDecision'
import { simulateFusion }    from '../api/fusionApi'

const streamHeader = (name, icon, owner, active, enabled) => (
  <div className="card-header">
    <div className="flex items-center gap-2">
      <div className={`w-6 h-6 rounded flex-shrink-0 grid place-items-center ${active ? 'bg-teal-dim' : 'bg-bg-3'}`}>
        {icon}
      </div>
      <div>
        <span className="card-title">{name}</span>
        <span className="text-[9px] font-mono text-ink-dim ml-2">{owner}</span>
      </div>
    </div>
    <div className="flex items-center gap-2">
      {active && enabled && <span className="text-[9px] font-mono text-teal animate-pulse">live</span>}
      {!enabled && <span className="text-[9px] font-mono text-ink-dim">channel disabled</span>}
    </div>
  </div>
)

export default function LiveAnalysisPage({ onNewResult }) {
  const [numResult,   setNumResult]   = useState(null)
  const [nlpResult,   setNlpResult]   = useState(null)
  const [voiceResult, setVoiceResult] = useState(null)
  const [fusionResult,setFusionResult]= useState(null)
  const [nlpEnabled,  setNlpEnabled]  = useState(true)
  const [voiceEnabled,setVoiceEnabled]= useState(false)
  const [channel,     setChannel]     = useState('online')

  const handleChannelChange = useCallback((ch, streams) => {
    setChannel(ch)
    setNlpEnabled(streams.nlp)
    setVoiceEnabled(streams.voice)
    if (!streams.nlp)   { setNlpResult(null) }
    if (!streams.voice) { setVoiceResult(null) }
  }, [])

  // Run fusion whenever any stream result changes
  useEffect(() => {
    if (!numResult) { setFusionResult(null); return }
    const result = simulateFusion(
      numResult,
      nlpEnabled   ? nlpResult   : null,
      voiceEnabled ? voiceResult : null,
    )
    setFusionResult(result)
    onNewResult?.(result, channel)
  }, [numResult, nlpResult, voiceResult, nlpEnabled, voiceEnabled])

  const numActive   = !!numResult
  const nlpActive   = nlpEnabled  && !!nlpResult
  const voiceActive = voiceEnabled && !!voiceResult

  return (
    <div className="h-full overflow-y-auto p-4">

      <div className="mb-4">
        <h2 className="font-display text-base font-semibold tracking-tight">Live Transaction Analysis</h2>
        <p className="text-[10px] font-mono text-ink-muted mt-0.5">
          Fill in transaction data below. Each stream analyzes in real time. The Fusion Layer combines them at the bottom.
        </p>
      </div>

      {/* ── Three stream panels side by side ── */}
      <div className="grid grid-cols-3 gap-4 mb-6">

        {/* Numerical */}
        <div className="card">
          {streamHeader(
            'Numerical Stream',
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" className={numActive ? 'text-sky' : 'text-ink-dim'}>
              <path d="M2 11L5.5 7l3 3.5L12 4l2 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            </svg>,
            'Ivan · XGBoost + RF',
            numActive, true
          )}
          <div className="card-body">
            <NumericalStreamPanel onChange={setNumResult} onChannelChange={handleChannelChange} />
          </div>
        </div>

        {/* NLP */}
        <div className="card">
          {streamHeader(
            'NLP Stream',
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" className={nlpActive ? 'text-teal' : 'text-ink-dim'}>
              <rect x="2" y="4" width="12" height="2" rx="1" fill="currentColor"/>
              <rect x="2" y="7.5" width="8" height="2" rx="1" fill="currentColor" opacity=".6"/>
              <rect x="2" y="11" width="10" height="2" rx="1" fill="currentColor" opacity=".3"/>
            </svg>,
            'Luis · FinBERT + spaCy',
            nlpActive, nlpEnabled
          )}
          <div className="card-body">
            <NLPStreamPanel enabled={nlpEnabled} onChange={setNlpResult} />
          </div>
        </div>

        {/* Voice */}
        <div className="card">
          {streamHeader(
            'Voice Stream',
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" className={voiceActive ? 'text-amber' : 'text-ink-dim'}>
              <path d="M8 2C6 2 4 3.5 4 5.5C4 7.5 5 9 8 11C11 9 12 7.5 12 5.5C12 3.5 10 2 8 2Z" stroke="currentColor" strokeWidth="1.3" fill="none"/>
              <line x1="8" y1="11" x2="8" y2="14" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
            </svg>,
            'Arsenii · WavLM',
            voiceActive, voiceEnabled
          )}
          <div className="card-body">
            <VoiceStreamPanel enabled={voiceEnabled} onChange={setVoiceResult} />
          </div>
        </div>
      </div>

      {/* ── Stream status bar ── */}
      <div className="flex items-center gap-3 mb-6 px-4 py-2.5 bg-bg-2 border border-border rounded-lg">
        <span className="text-[10px] font-mono text-ink-dim uppercase tracking-wider">Fusion status</span>
        <div className="flex-1 flex items-center gap-2">
          {[
            { label: 'Numerical', active: numActive,   color: 'bg-sky'    },
            { label: 'NLP',       active: nlpActive,   color: 'bg-teal',  disabled: !nlpEnabled   },
            { label: 'Voice',     active: voiceActive, color: 'bg-amber', disabled: !voiceEnabled },
          ].map(s => (
            <div key={s.label} className="flex items-center gap-1.5">
              <div className={`w-1.5 h-1.5 rounded-full ${s.disabled ? 'bg-ink-dim opacity-30' : s.active ? s.color : 'bg-ink-dim'} ${s.active ? 'animate-pulse' : ''}`} />
              <span className={`text-[9px] font-mono ${s.disabled ? 'text-ink-dim opacity-40' : s.active ? 'text-ink-muted' : 'text-ink-dim'}`}>
                {s.label}{s.disabled ? ' (off)' : ''}
              </span>
            </div>
          ))}
        </div>
        {fusionResult && (
          <div className="flex items-center gap-2">
            <span className={`mode-pill ${fusionResult.ensemble_mode === 'full' ? 'mode-pill-full' : fusionResult.ensemble_mode === 'partial' ? 'mode-pill-partial' : 'mode-pill-fallback'}`}>
              {fusionResult.ensemble_mode}
            </span>
          </div>
        )}
      </div>

      {/* ── Fusion Decision ── */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-teal-dim grid place-items-center flex-shrink-0">
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none" className="text-teal">
                <path d="M7 1L12.196 4V10L7 13L1.804 10V4L7 1Z" fill="currentColor"/>
              </svg>
            </div>
            <span className="card-title">Fusion Layer Decision</span>
            <span className="text-[9px] font-mono text-ink-dim">Sherwayne · orchestrator.py</span>
          </div>
          {fusionResult && (
            <span className="text-[9px] font-mono text-teal">
              {fusionResult.audit.processing_time_ms.toFixed(1)} ms
            </span>
          )}
        </div>
        <div className="card-body">
          <FusionDecision result={fusionResult} numResult={numResult} nlpResult={nlpResult} voiceResult={voiceResult} />
        </div>
      </div>

    </div>
  )
}
