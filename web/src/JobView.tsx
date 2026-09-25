import type { JobStatus } from './api'

// Shared status + result rendering for both pipelines.
export function JobView({ job, error }: { job: JobStatus | null; error: string | null }) {
  if (error) return <p className="err">{error}</p>
  if (!job) return <p className="muted">submitting…</p>
  const m = job.result?.metrics as { total_s?: number; total_gen_s?: number } | undefined
  return (
    <div className="job">
      <div className="status">
        {job.status === 'queued' && <span>queued (position {job.position})</span>}
        {job.status === 'running' && <span>running · {job.stage ?? 'finishing'} · {job.elapsed_s}s</span>}
        {job.status === 'done' && <span>done in {job.elapsed_s}s</span>}
        {job.status === 'failed' && <span className="err">failed: {job.error}</span>}
      </div>
      {job.stages_done.length > 0 && (
        <table className="stages">
          <tbody>
            {job.stages_done.map((s, i) => (
              <tr key={i}><td>{s.stage}</td><td className="muted">{s.model}</td><td>{(s.time_s ?? s.gen_s ?? 0).toFixed(1)} s</td><td className="muted">{s.gpu_peak_gb ? `${s.gpu_peak_gb} GB` : ''}</td></tr>
            ))}
            {m && <tr><td>total</td><td /><td>{(m.total_s ?? m.total_gen_s ?? 0).toFixed(1)} s</td><td /></tr>}
          </tbody>
        </table>
      )}
      {job.result && (
        <>
          <a href={job.result.output_url} target="_blank" rel="noreferrer"><img className="result" src={job.result.output_url} alt="output" /></a>
          <div className="thumbs">
            {job.result.intermediates.map((f) => (
              <a key={f.name} href={f.url} target="_blank" rel="noreferrer" title={f.name}><img src={f.url} alt={f.name} /><span>{f.name}</span></a>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
