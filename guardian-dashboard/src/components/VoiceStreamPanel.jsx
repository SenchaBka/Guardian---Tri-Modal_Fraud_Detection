import { useState, useRef, useEffect } from 'react'

const SAMPLES = [
  { label: 'Deepfake sample A',  name: 'deepfake_call_001.wav',    isDeepfake: true  },
  { label: 'Deepfake sample B',  name: 'synthetic_voice_002.wav',  isDeepfake: true  },
  { label: 'Genuine customer A', name: 'genuine_call_003.wav',     isDeepfake: false },
  { label: 'Genuine customer B', name: 'bonafide_speaker_004.wav', isDeepfake: false },
]

function WaveformDisplay({ result, selected }) {
  const bars = Array.from({ length: 32 }, (_, i) => 0.2 + 0.6 * Math.abs(Math.sin(i * 0.8)))
  const color = result
    ? result.is_deepfake ? '#ff4757' : '#00e5c3'
    : selected
      ? selected.isDeepfake ? '#ff475740' : '#00e5c340'
      : 'rgba(58,63,85,0.5)'
  return (
    <div className="flex items-center justify-center gap-0.5 h-10">
      {bars.map((h, i) => (
        <div key={i} className="w-1 rounded-full"
          style={{ height: `${h*40}px`, background: color, opacity: result ? (0.6 + h*0.4) : 0.5 }} />
      ))}
    </div>
  )
}

export default function VoiceStreamPanel({ enabled, onSelect, result }) {
  const [selected, setSelected] = useState(null)
  const [dragOver,  setDragOver] = useState(false)
  const inputRef = useRef()

  useEffect(() => {
    if (!enabled) { setSelected(null); onSelect?.(null) }
  }, [enabled])

  const pick = (sample) => { setSelected(sample); onSelect?.(sample) }

  const handleFile = (f) => {
    const isDeepfake = /deep|syn|fake/i.test(f.name)
    pick({ label: f.name, name: f.name, isDeepfake })
  }

  const handleDrop = e => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  if (!enabled) {
    return (
      <div className="flex flex-col items-center justify-center py-8 gap-2 text-center">
        <div className="w-10 h-10 rounded-lg bg-bg-3 grid place-items-center opacity-40">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-ink-muted">
            <path d="M8 2C6 2 4 3.5 4 5.5C4 7.5 5 9 8 11C11 9 12 7.5 12 5.5C12 3.5 10 2 8 2Z"
              stroke="currentColor" strokeWidth="1.3" fill="none"/>
            <line x1="8" y1="11" x2="8" y2="14" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
          </svg>
        </div>
        <p className="text-[10px] font-mono text-ink-dim">Voice unavailable for this channel</p>
        <p className="text-[9px] text-ink-dim opacity-60">Switch to Phone or In Person</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">

      {/* Sample selector */}
      <div>
        <label className="form-label">Select audio sample</label>
        <div className="grid grid-cols-2 gap-1.5">
          {SAMPLES.map(s => {
            const isSelected = selected?.name === s.name
            return (
              <button key={s.name} onClick={() => pick(s)}
                className={`text-[10px] font-mono px-2.5 py-2 rounded border text-left transition-all
                  ${isSelected
                    ? s.isDeepfake
                      ? 'border-danger bg-danger-dim text-danger'
                      : 'border-teal bg-teal-dim text-teal'
                    : 'border-border bg-bg-3 text-ink-muted hover:border-border-strong hover:text-ink'}`}>
                <div className="flex items-center gap-1.5">
                  {isSelected && <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${s.isDeepfake?'bg-danger':'bg-teal'}`} />}
                  <span className="font-medium truncate">{s.label}</span>
                </div>
                <div className="opacity-60 text-[9px] mt-0.5 truncate">{s.name}</div>
              </button>
            )
          })}
        </div>
      </div>

      {/* File upload */}
      <div>
        <label className="form-label">Or upload audio file</label>
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg px-4 py-3 text-center cursor-pointer transition-all
            ${dragOver ? 'border-teal bg-teal-dim' : 'border-border hover:border-border-strong hover:bg-bg-3'}`}>
          <input ref={inputRef} type="file" accept="audio/*,.wav,.mp3,.flac,.m4a" className="hidden"
            onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />
          <p className="text-[10px] font-mono text-ink-muted">
            {selected && !SAMPLES.find(s => s.name === selected.name) ? selected.name : 'Drop audio or click to upload'}
          </p>
          <p className="text-[9px] text-ink-dim mt-0.5">WAV · MP3 · FLAC · M4A</p>
        </div>
      </div>

      {/* Waveform + result (result only shown after Analyze is clicked) */}
      {selected && (
        <div className="border border-border rounded-lg overflow-hidden">
          <div className="px-3 py-2 bg-bg-3 border-b border-border flex items-center justify-between">
            <span className="text-[9px] font-mono uppercase tracking-wider text-ink-muted">
              Arsenii · WavLM deepfake detection
            </span>
            <span className={`text-[9px] font-mono ${
              result ? (result.is_deepfake ? 'text-danger' : 'text-teal')
              : selected.isDeepfake ? 'text-danger' : 'text-teal'}`}>
              {result ? (result.is_deepfake ? 'deepfake detected' : 'genuine voice')
                : selected.isDeepfake ? 'deepfake selected' : 'genuine selected'}
            </span>
          </div>
          <div className="p-3 space-y-2.5">
            <div className="bg-bg-3 rounded px-3 py-2 border border-border">
              <WaveformDisplay result={result} selected={selected} />
              <p className="text-[9px] font-mono text-ink-dim text-center mt-1">{selected.name}</p>
            </div>

            {/* Scores only after Analyze */}
            {result ? (
              <>
                <div className={`rounded-lg p-3 border text-center ${result.is_deepfake ? 'bg-danger-dim border-danger/30' : 'bg-teal-dim border-teal/30'}`}>
                  <div className={`text-base font-bold font-display ${result.is_deepfake ? 'text-danger' : 'text-teal'}`}>
                    {result.is_deepfake ? '⊗ DEEPFAKE DETECTED' : '✓ GENUINE VOICE'}
                  </div>
                  <div className={`text-[10px] font-mono mt-1 ${result.is_deepfake ? 'text-danger/70' : 'text-teal/70'}`}>
                    {(result.deepfake_probability*100).toFixed(1)}% synthetic · {(result.bonafide_prob*100).toFixed(1)}% bonafide
                  </div>
                </div>
                <div className="space-y-1.5">
                  {[
                    { label: 'Deepfake probability', value: result.deepfake_probability, color: '#ff4757' },
                    { label: 'Bonafide probability', value: result.bonafide_prob,         color: '#00e5c3' },
                  ].map(({ label, value, color }) => (
                    <div key={label}>
                      <div className="flex justify-between text-[10px] font-mono mb-1">
                        <span className="text-ink-muted">{label}</span>
                        <span style={{ color }}>{(value*100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 bg-bg-4 rounded overflow-hidden">
                        <div className="h-full rounded transition-all duration-700" style={{ width:`${(value*100).toFixed(1)}%`, background: color }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-[9px] font-mono text-ink-dim text-center">
                Click <span className="text-teal">Run Fusion Analysis</span> to process
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}