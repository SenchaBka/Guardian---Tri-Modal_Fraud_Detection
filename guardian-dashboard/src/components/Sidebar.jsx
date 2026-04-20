import { useApiStatus } from '../hooks/useApiStatus'

const HexIcon = () => (
  <div className="w-7 h-7 bg-teal rounded-md grid place-items-center flex-shrink-0">
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
      <path d="M7 1L12.196 4V10L7 13L1.804 10V4L7 1Z" fill="#07080b" />
    </svg>
  </div>
)

const NavItem = ({ icon, label, right, active, onClick }) => (
  <div className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}>
    <span className="opacity-70 flex-shrink-0">{icon}</span>
    <span className="flex-1">{label}</span>
    {right && <span className="text-[9px] font-mono text-ink-dim">{right}</span>}
  </div>
)

const SectionLabel = ({ children }) => (
  <p className="text-[9px] font-mono uppercase tracking-widest text-ink-dim px-2 pt-4 pb-1">{children}</p>
)

export default function Sidebar({ activePage, onNavigate }) {
  const { status, version } = useApiStatus()

  return (
    <aside className="bg-bg-1 border-r border-border flex flex-col overflow-hidden">

      {/* Logo */}
      <div className="px-4 py-5 border-b border-border flex-shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <HexIcon />
          <span className="font-display text-base font-bold tracking-tight">Guardian</span>
        </div>
        <p className="text-[9px] font-mono uppercase tracking-widest text-ink-muted">
          Tri-Modal Fraud Detection
        </p>
      </div>

      <nav className="p-2 flex-1 overflow-y-auto">

        <SectionLabel>Detection</SectionLabel>

        <NavItem active={activePage==='analysis'} onClick={()=>onNavigate('analysis')} label="Dashboard"
          icon={<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="6" height="6" rx="1.5" fill="currentColor"/><rect x="9" y="1" width="6" height="6" rx="1.5" fill="currentColor" opacity=".4"/><rect x="1" y="9" width="6" height="6" rx="1.5" fill="currentColor" opacity=".4"/><rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor"/></svg>}
        />
        <NavItem active={activePage==='audit'} onClick={()=>onNavigate('audit')} label="Audit Log"
          icon={<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="5.5" stroke="currentColor" strokeWidth="1.3"/><path d="M8 5v3.5l2.5 1.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>}
        />

        <SectionLabel>Streams</SectionLabel>

        <NavItem label="Numerical" right="Ivan"
          icon={<svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="text-sky"><path d="M2 11L5.5 7l3 3.5L12 4l2 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" fill="none"/></svg>}
        />
        <NavItem label="NLP" right="Luis"
          icon={<svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="text-teal"><rect x="2" y="4" width="12" height="2" rx="1" fill="currentColor"/><rect x="2" y="7.5" width="8" height="2" rx="1" fill="currentColor" opacity=".5"/><rect x="2" y="11" width="10" height="2" rx="1" fill="currentColor" opacity=".3"/></svg>}
        />
        <NavItem label="Voice" right="Arsenii"
          icon={<svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="text-amber"><path d="M8 2C6 2 4 3.5 4 5.5C4 7.5 5 9 8 11C11 9 12 7.5 12 5.5C12 3.5 10 2 8 2Z" stroke="currentColor" strokeWidth="1.2" fill="none"/><line x1="8" y1="11" x2="8" y2="14" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>}
        />

      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0
            ${status==='online' ? 'bg-teal animate-pulse-dot' : status==='offline' ? 'bg-amber' : 'bg-ink-dim'}`} />
          <span className="font-mono text-[10px] text-ink-muted">
            API <span className={status==='online'?'text-teal':status==='offline'?'text-amber':'text-ink-dim'}>
              {status==='checking'?'connecting…':status}
            </span>
          </span>
        </div>
        <p className="font-mono text-[9px] text-ink-dim mt-1">v{version||'2.0.0'} · localhost:8000</p>
      </div>

    </aside>
  )
}