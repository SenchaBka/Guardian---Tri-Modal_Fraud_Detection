import { useState, useCallback } from 'react'
import Sidebar      from './components/Sidebar'
import AnalysisPage from './pages/AnalysisPage'
import AuditLogPage from './pages/AuditLogPage'
import { useApiStatus } from './hooks/useApiStatus'

const modeTag = {
  full:     'tag-teal',
  partial:  'tag-amber',
  fallback: 'tag-danger',
}

export default function App() {
  const { status } = useApiStatus()
  const [page,     setPage]    = useState('analysis')
  const [auditLog, setAuditLog]= useState([])
  const [stats,    setStats]   = useState({ total:0, approved:0, review:0, blocked:0 })
  const [mode,     setMode]    = useState(null) // current ensemble mode for topbar

  const handleNewResult = useCallback((result) => {
    setAuditLog(prev => [result, ...prev].slice(0, 200))
    setStats(prev => ({
      total:    prev.total + 1,
      approved: prev.approved + (result.decision==='approve' ? 1 : 0),
      review:   prev.review   + (result.decision==='review'  ? 1 : 0),
      blocked:  prev.blocked  + (result.decision==='block'   ? 1 : 0),
    }))
  }, [])

  return (
    <div className="h-screen flex overflow-hidden bg-bg-0">
      <div className="w-52 flex-shrink-0">
        <Sidebar activePage={page} onNavigate={setPage}/>
      </div>
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Top bar */}
        <div className="h-12 bg-bg-1 border-b border-border flex items-center justify-between px-5 flex-shrink-0">
          <span className="text-xs font-medium text-ink">
            {page === 'analysis' ? 'Fraud Analysis Dashboard' : 'Audit Log'}
          </span>
          <div className="flex items-center gap-3">
            {/* Ensemble mode badge — shows after first analysis */}
            {mode && (
              <span className={`tag ${modeTag[mode]} uppercase tracking-wider`}>
                {mode} mode
              </span>
            )}
            <span className="text-[10px] font-mono text-ink-muted">COMP386 · Winter 2026</span>
          </div>
        </div>

        {/* Pages */}
        <div className="flex-1 overflow-hidden">
          {page === 'analysis'
            ? <AnalysisPage
                stats={stats}
                onNewResult={handleNewResult}
                onModeChange={setMode}
              />
            : <AuditLogPage auditLog={auditLog}/>
          }
        </div>

      </div>
    </div>
  )
}