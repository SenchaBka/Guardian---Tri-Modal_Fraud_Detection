import { useState } from 'react'

const decisionColor = { approve: 'tag-teal', review: 'tag-amber', block: 'tag-danger' }
const modeClass     = { full: 'mode-pill-full', partial: 'mode-pill-partial', fallback: 'mode-pill-fallback' }
const scoreColor    = s => s >= 0.71 ? 'text-danger' : s >= 0.31 ? 'text-amber' : 'text-teal'

export default function AuditLogPage({ auditLog }) {
  const [filter,   setFilter]  = useState('all')
  const [expanded, setExpanded]= useState(null)

  const filtered = filter === 'all' ? auditLog : auditLog.filter(r => r.decision === filter)

  return (
    <div className="h-full overflow-y-auto p-4">

      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="font-display text-base font-semibold tracking-tight">Audit Log</h2>
          <p className="text-[10px] font-mono text-ink-muted mt-0.5">
            Full decision history with reasoning — every transaction is recorded and explainable
          </p>
        </div>
        <span className="tag tag-dim">{auditLog.length} records</span>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-4">
        {['all','block','review','approve'].map(f => (
          <button key={f}
            onClick={() => setFilter(f)}
            className={`text-[10px] font-mono px-3 py-1.5 rounded border transition-all
              ${filter === f
                ? f==='block'   ? 'bg-danger-dim text-danger border-danger/30'
                  : f==='review'  ? 'bg-amber-dim text-amber border-amber/30'
                  : f==='approve' ? 'bg-teal-dim text-teal border-teal/30'
                  : 'bg-bg-3 text-ink border-border-strong'
                : 'bg-bg-2 text-ink-muted border-border hover:border-border-strong'}`}>
            {f === 'all' ? `All (${auditLog.length})` : `${f} (${auditLog.filter(r=>r.decision===f).length})`}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-16 text-[10px] font-mono text-ink-dim">
          {auditLog.length === 0
            ? 'No transactions analyzed yet — run an analysis from the Live Analysis tab'
            : `No ${filter} decisions recorded`}
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((r, i) => (
            <div key={i} className="card cursor-pointer hover:border-border-strong transition-colors"
              onClick={() => setExpanded(expanded === i ? null : i)}>

              {/* Summary row */}
              <div className="p-3 flex items-center gap-3">
                {/* Decision */}
                <div className={`w-20 text-center py-1.5 rounded font-display text-xs font-bold border
                  ${r.decision==='block'   ? 'bg-danger-dim text-danger border-danger/30'
                  : r.decision==='review'  ? 'bg-amber-dim  text-amber  border-amber/30'
                  : 'bg-teal-dim text-teal border-teal/30'}`}>
                  {r.decision.toUpperCase()}
                </div>

                {/* Transaction ID */}
                <div className="flex-1 min-w-0">
                  <div className="font-mono text-xs text-ink truncate">{r.transaction_id}</div>
                  <div className="font-mono text-[9px] text-ink-dim mt-0.5">{r.audit?.request_id}</div>
                </div>

                {/* Score */}
                <div className={`font-mono text-sm font-medium ${scoreColor(r.fraud_score)} min-w-[48px] text-right`}>
                  {(r.fraud_score*100).toFixed(1)}%
                </div>

                {/* Mode */}
                <span className={`mode-pill ${modeClass[r.ensemble_mode]}`}>{r.ensemble_mode}</span>

                {/* Confidence */}
                <div className="text-[10px] font-mono text-ink-muted min-w-[50px] text-right">
                  {(r.confidence*100).toFixed(0)}% conf
                </div>

                {/* Expand arrow */}
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className={`text-ink-dim transition-transform ${expanded===i?'rotate-180':''}`}>
                  <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>

              {/* Expanded detail */}
              {expanded === i && (
                <div className="border-t border-border p-4 space-y-4 animate-fade-in">

                  {/* Stream scores */}
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim mb-2">Stream scores</p>
                    <div className="grid grid-cols-3 gap-2">
                      {Object.entries(r.modality_scores).map(([name, detail]) => (
                        <div key={name} className={`bg-bg-3 rounded p-2.5 border border-border ${!detail.used?'opacity-40':''}`}>
                          <div className="text-[9px] font-mono text-ink-dim capitalize mb-1">{name}</div>
                          <div className={`font-mono text-sm font-medium ${detail.score!=null?scoreColor(detail.score):'text-ink-dim'}`}>
                            {detail.score!=null ? `${(detail.score*100).toFixed(1)}%` : 'N/A'}
                          </div>
                          <div className="text-[9px] font-mono text-ink-dim mt-1">w={detail.weight.toFixed(3)} · {detail.status}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Top risk factors */}
                  {r.explanation?.top_factors?.length > 0 && (
                    <div>
                      <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim mb-2">Risk factors</p>
                      <div className="space-y-1.5">
                        {r.explanation.top_factors.map((f, fi) => (
                          <div key={fi} className="flex items-center gap-2.5">
                            <div className="flex-1 min-w-0">
                              <div className="text-[10px] font-mono text-ink-muted truncate">{f.feature}</div>
                            </div>
                            <div className="w-24 h-1 bg-bg-4 rounded overflow-hidden flex-shrink-0">
                              <div className="h-full rounded bg-danger" style={{ width:`${(f.impact/0.5*100).toFixed(0)}%`, maxWidth:'100%' }} />
                            </div>
                            <div className="text-[9px] font-mono text-ink-dim w-10 text-right">{f.impact.toFixed(3)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Narrative */}
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim mb-1.5">Explanation</p>
                    <p className="text-[10px] font-mono text-ink-muted leading-relaxed">{r.explanation?.narrative}</p>
                    {r.explanation?.missing_modality_note && (
                      <p className="text-[9px] font-mono text-amber/70 mt-1.5">{r.explanation.missing_modality_note}</p>
                    )}
                    {r.confidence_adjustment?.availability_penalty > 0 && (
                      <p className="text-[9px] font-mono text-danger/70 mt-1">
                        Conservative override applied — {r.ensemble_mode} mode penalty: −{(r.confidence_adjustment.availability_penalty*100).toFixed(0)}%
                      </p>
                    )}
                  </div>

                  {/* Audit metadata */}
                  <div className="grid grid-cols-3 gap-2 text-[9px] font-mono text-ink-dim">
                    <div>Processed in {r.audit?.processing_time_ms?.toFixed(1)} ms</div>
                    <div>Model v{r.model_version}</div>
                    <div>Streams: {r.audit?.modalities_used?.join(', ')}</div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
