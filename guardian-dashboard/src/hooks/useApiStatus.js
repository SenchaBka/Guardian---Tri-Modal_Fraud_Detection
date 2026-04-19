import { useState, useEffect, useCallback } from 'react'

export function useApiStatus() {
  const [status,  setStatus]  = useState('checking')
  const [version, setVersion] = useState(null)

  const ping = useCallback(async () => {
    try {
      const r = await fetch('/api/v1/fusion/health', { signal: AbortSignal.timeout(2500) })
      if (!r.ok) throw new Error()
      const data = await r.json()
      setStatus('online')
      setVersion(data.version)
    } catch {
      setStatus('offline')
    }
  }, [])

  useEffect(() => {
    ping()
    const id = setInterval(ping, 10000)
    return () => clearInterval(id)
  }, [ping])

  return { status, version, ping }
}
