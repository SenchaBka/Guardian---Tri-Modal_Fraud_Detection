import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const decisionConfig = {
  block:   { label: 'BLOCKED',  color: '#ff4757', bg: 'bg-danger-dim',  border: 'border-danger/30',  icon: '⊗' },
  review:  { label: 'REVIEW',   color: '#f5a623', bg: 'bg-amber-dim',   border: 'border-amber/30',   icon: '◈' },
  approve: { label: 'APPROVED', color: '#00e5c3', bg: 'bg-teal-dim',    border: 'border-teal/30',    icon: '✓' },
}

const modeClass = { full:'mode-pill-full', partial:'mode-pill-partial', fallback:'mode-pill-fallback' }

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-bg-3 border border-border rounded px-2.5 py-1.5 text-[10px] font-mono text-ink-muted">
      {payload[0].payload.feature}: <span className="text-ink">{payload[0].value.toFixed(4)}</span>
    </div>
  )
}

export default function FusionDecision({ result, numResult, nlpResult, voiceResult }) {
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
        <div className="w-14 h-14 rounded-2xl bg-bg-3 border border-border grid place-items-center">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-ink-dim">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <div>
          <p className="text-sm font-medium text-ink-muted">Awaiting analysis</p>
          <p className="text-[10px] font-mono text-ink-dim mt-1">Complete stream inputs above to see the fusion result</p>
        </div>
      </div>
    )
  }

  const dc  = decisionConfig[result.decision]
  const sc  = result.fraud_score
  const exp = result.explanation
  const ca  = result.confidence_adjustment
  const ms  = result.modality_scores

  const chartData = exp.top_factors.slice(0, 5).map(f => ({
    feature: f.feature.split(' ').slice(-2).join(' '),
    impact:  f.impact,
    color:   f.direction === 'positive' ? '#ff4757' : '#00e5c3',
  }))

  return (
    <div className="animate-slide-up space-y-4">

      {/* ── Main verdict ── */}
      <div className={`rounded-xl p-5 border ${dc.bg} ${dc.border}`}>
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest mb-1" style={{ color: dc.color, opacity: 0.7 }}>
              Guardian fusion decision
            </div>
            <div className="font-display text-3xl font-bold" style={{ color: dc.color }}>
              {dc.icon} {dc.label}
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-3xl font-medium" style={{ color: dc.color }}>
              {(sc * 100).toFixed(1)}%
            </div>
            <div className="text-[9px] font-mono uppercase tracking-wider mt-0.5" style={{ color: dc.color, opacity: 0.6 }}>
              fraud score
            </div>
          </div>
        </div>

        {/* Mode + risk + confidence row */}
        <div className="flex flex-wrap items-center gap-2">
          <span className={`mode-pill ${modeClass[result.ensemble_mode]}`}>{result.ensemble_mode} mode</span>
          <span className="tag tag-dim">{result.risk_level} risk</span>
          {result.decision === 'review' && <span className="tag tag-amber">human review required</span>}
          <span className="tag tag-dim font-mono">{(result.confidence*100).toFixed(0)}% confidence</span>
          {ca.availability_penalty > 0 && (
            <span className="tag tag-danger">−{(ca.availability_penalty*100).toFixed(0)}% penalty</span>
          )}
        </div>

        {/* Override note */}
        {ca.availability_penalty > 0 && (
          <div className="mt-3 text-[10px] font-mono text-danger/70 bg-danger/10 rounded px-2.5 py-1.5 border border-danger/20">
            Conservative override active — {result.ensemble_mode} mode lowers decision thresholds due to missing streams
          </div>
        )}
      </div>

      {/* ── Stream weight breakdown ── */}
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(ms).map(([name, detail]) => {
          const colors = { numerical: '#4a9eff', nlp: '#00e5c3', voice: '#f5a623' }
          const col = colors[name]
          const unavail = !detail.used
          return (
            <div key={name} className={`rounded-lg p-3 border border-border bg-bg-2 ${unavail ? 'opacity-40' : ''}`}>
              <div className="flex justify-between items-start mb-2">
                <div className="text-xs font-medium capitalize" style={{ color: unavail ? '#6b7494' : col }}>{name}</div>
                <div className="text-[9px] font-mono text-ink-dim">{(detail.weight*100).toFixed(0)}% weight</div>
              </div>
              <div className="font-mono text-base font-medium" style={{ color: unavail ? '#3a3f55' : (detail.score >= 0.71 ? '#ff4757' : detail.score >= 0.31 ? '#f5a623' : '#00e5c3') }}>
                {unavail ? 'N/A' : `${(detail.score * 100).toFixed(1)}%`}
              </div>
              {!unavail && (
                <div className="mt-1.5 h-1 bg-bg-4 rounded overflow-hidden">
                  <div className="h-full rounded" style={{ width:`${(detail.score*100).toFixed(1)}%`, background: col, opacity: 0.7 }} />
                </div>
              )}
              <div className="text-[9px] font-mono text-ink-dim mt-1">
                {unavail ? 'unavailable' : detail.status}
              </div>
            </div>
          )
        })}
      </div>

      {/* ── SHAP Feature Impact ── */}
      {chartData.length > 0 && (
        <div className="bg-bg-2 border border-border rounded-lg p-4">
          <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim mb-3">Feature impact (SHAP)</p>
          <ResponsiveContainer width="100%" height={chartData.length * 28 + 20}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16, top: 0, bottom: 0 }}>
              <XAxis type="number" domain={[0,'auto']} hide />
              <YAxis type="category" dataKey="feature" width={130}
                tick={{ fontSize: 10, fontFamily: 'DM Mono', fill: '#6b7494' }}
                axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="impact" radius={[0, 3, 3, 0]}>
                {chartData.map((e, i) => <Cell key={i} fill={e.color} opacity={0.85} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Narrative ── */}
      <div className="bg-bg-2 border border-border rounded-lg p-4">
        <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim mb-2">Explanation</p>
        <p className="text-xs font-mono text-ink-muted leading-relaxed">{exp.narrative}</p>
        {exp.missing_modality_note && (
          <p className="text-[10px] font-mono text-amber/70 mt-2 pt-2 border-t border-border">
            {exp.missing_modality_note}
          </p>
        )}
      </div>

      {/* ── Audit info ── */}
      <div className="bg-bg-2 border border-border rounded-lg p-4">
        <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim mb-2.5">Audit record</p>
        <div className="grid grid-cols-2 gap-1.5 mb-3">
          {[
            ['Processing time', `${result.audit.processing_time_ms.toFixed(1)} ms`],
            ['Model version',   `v${result.model_version}`],
            ['Streams used',    result.audit.modalities_used.join(', ')],
            ['Confidence',      `${(result.confidence*100).toFixed(1)}%`],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between text-[10px] font-mono">
              <span className="text-ink-dim">{k}</span>
              <span className="text-ink-muted">{v}</span>
            </div>
          ))}
        </div>
        <div className="text-[9px] font-mono text-ink-dim break-all px-2.5 py-1.5 bg-bg-3 rounded border border-border">
          {result.audit.request_id}
        </div>
      </div>

    </div>
  )
}