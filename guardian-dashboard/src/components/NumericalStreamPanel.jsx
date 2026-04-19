import { useState, useEffect } from 'react'

const CHANNELS = [
  { value: 'online',    label: 'Online',     streams: { nlp: true,  voice: false } },
  { value: 'phone',     label: 'Phone Call', streams: { nlp: true,  voice: true  } },
  { value: 'in_person', label: 'In Person',  streams: { nlp: true,  voice: true  } },
  { value: 'atm',       label: 'ATM',        streams: { nlp: false, voice: false } },
]

export default function NumericalStreamPanel({ onChange, onChannelChange, result }) {
  const [fields, setFields] = useState({
    amount: '', currency: 'USD', channel: 'online',
    country: 'US', merchant_category: 'retail',
  })

  const set = k => e => setFields(f => ({ ...f, [k]: e.target.value }))

  // notify parent when channel changes
  useEffect(() => {
    const ch = CHANNELS.find(c => c.value === fields.channel)
    onChannelChange?.(fields.channel, ch?.streams || { nlp: true, voice: false })
  }, [fields.channel])

  // pass raw fields up — parent simulates on button click
  useEffect(() => {
    if (!fields.amount) { onChange?.(null); return }
    onChange?.(fields)
  }, [fields.amount, fields.country, fields.channel, fields.merchant_category, fields.currency])

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2.5">
        <div>
          <label className="form-label">Transaction amount</label>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted text-xs font-mono">$</span>
            <input className="form-input pl-6" type="number" placeholder="0.00"
              value={fields.amount} onChange={set('amount')} />
          </div>
        </div>
        <div>
          <label className="form-label">Currency</label>
          <select className="form-select" value={fields.currency} onChange={set('currency')}>
            {['USD','CAD','EUR','GBP','NGN'].map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className="form-label">Channel</label>
          <select className="form-select" value={fields.channel} onChange={set('channel')}>
            {CHANNELS.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
          </select>
        </div>
        <div>
          <label className="form-label">Country</label>
          <input className="form-input font-mono uppercase" placeholder="US"
            value={fields.country} onChange={set('country')} maxLength={3} />
        </div>
        <div className="col-span-2">
          <label className="form-label">Merchant category</label>
          <select className="form-select" value={fields.merchant_category} onChange={set('merchant_category')}>
            {['retail','electronics','gambling','crypto','wire_transfer','cash_advance','forex','restaurants','travel','healthcare'].map(c =>
              <option key={c} value={c}>{c}</option>
            )}
          </select>
        </div>
      </div>

      {/* Results — only shown after Analyze is clicked */}
      {result && (
        <div className="border border-border rounded-lg overflow-hidden animate-fade-in">
          <div className="px-3 py-2 bg-bg-3 border-b border-border flex items-center justify-between">
            <span className="text-[9px] font-mono uppercase tracking-wider text-ink-muted">
              Ivan · XGBoost + Random Forest
            </span>
            <span className="text-[9px] font-mono" style={{
              color: result.fraud_probability >= 0.71 ? '#ff4757'
                : result.fraud_probability >= 0.31 ? '#f5a623' : '#00e5c3'
            }}>
              {(result.fraud_probability * 100).toFixed(1)}% fraud prob.
            </span>
          </div>
          <div className="p-3 space-y-2">
            {/* Score bar */}
            <div>
              <div className="flex justify-between text-[10px] font-mono mb-1">
                <span className="text-ink-muted">Fraud probability</span>
                <span style={{ color: result.fraud_probability>=0.71?'#ff4757':result.fraud_probability>=0.31?'#f5a623':'#00e5c3' }}>
                  {(result.fraud_probability*100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-bg-4 rounded overflow-hidden">
                <div className="h-full rounded transition-all duration-700"
                  style={{ width:`${(result.fraud_probability*100).toFixed(1)}%`,
                    background: result.fraud_probability>=0.71?'#ff4757':result.fraud_probability>=0.31?'#f5a623':'#00e5c3' }} />
              </div>
            </div>
            {/* Signal grid */}
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(result.signals).map(([k, v]) => (
                <div key={k} className="bg-bg-3 rounded p-2">
                  <div className="text-[9px] font-mono text-ink-dim uppercase tracking-wider mb-1">
                    {k.replace(/_/g, ' ')}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="flex-1 h-1 bg-bg-4 rounded overflow-hidden">
                      <div className="h-full rounded"
                        style={{ width:`${(v*100).toFixed(0)}%`, background:v>=0.6?'#ff4757':v>=0.3?'#f5a623':'#00e5c3' }} />
                    </div>
                    <span className="font-mono text-[10px] text-ink-muted">{(v*100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
            {/* Flags */}
            {result.flags.length > 0 && (
              <div className="space-y-1.5">
                {result.flags.map((f, i) => (
                  <div key={i} className={`text-[10px] font-mono px-2.5 py-1.5 rounded border
                    ${f.severity==='high'?'text-danger bg-danger-dim border-danger/30':'text-amber bg-amber-dim border-amber/30'}`}>
                    <span className="font-medium">{f.label}:</span>{' '}
                    <span className="opacity-80">{f.detail}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}