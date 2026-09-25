import { useEffect, useState } from 'react'
import { getJob, type JobStatus } from './api'

// Mirrors the server's job state; the server is the source of truth, this just polls it.
export function useJob(jobId: string | null, intervalMs = 1500) {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setJob(null); setError(null)
    if (!jobId) return
    let stop = false
    let timer: number | undefined
    const tick = async () => {
      try {
        const j = await getJob(jobId)
        if (stop) return
        setJob(j)
        if (j.status === 'done' || j.status === 'failed') return
      } catch (e) {
        if (stop) return
        setError((e as Error).message)
      }
      timer = window.setTimeout(tick, intervalMs)
    }
    tick()
    return () => { stop = true; if (timer) window.clearTimeout(timer) }
  }, [jobId, intervalMs])

  return { job, error }
}
