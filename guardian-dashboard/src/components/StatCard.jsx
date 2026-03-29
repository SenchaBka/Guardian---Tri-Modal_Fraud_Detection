const accentColors = {
  teal:   'bg-teal',
  amber:  'bg-amber',
  danger: 'bg-danger',
  sky:    'bg-sky',
}
const valueColors = {
  teal:   'text-teal',
  amber:  'text-amber',
  danger: 'text-danger',
  sky:    'text-sky',
}

export default function StatCard({ label, value, color = 'teal' }) {
  return (
    <div className="stat-card">
      <div className={`absolute top-0 left-0 right-0 h-0.5 ${accentColors[color]}`} />
      <p className="text-[9px] font-mono uppercase tracking-widest text-ink-muted mb-1.5">
        {label}
      </p>
      <p className={`font-mono text-2xl font-medium tracking-tight ${valueColors[color]}`}>
        {value}
      </p>
    </div>
  )
}