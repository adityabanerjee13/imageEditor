import { useState } from 'react'
import { createJob, uploadImage, type Finish, type Upload } from './api'
import { JobView } from './JobView'
import { useJob } from './useJob'

export function FloorEdit() {
  const [scene, setScene] = useState<Upload | null>(null)
  const [pattern, setPattern] = useState<Upload | null>(null)
  const [material, setMaterial] = useState<Finish>('smooth-matte')
  // A pattern PNG carries no scale, so the tile size comes from this: how much floor 1000 px of it
  // covers. Held as the raw string so a half-typed "5." stays editable; parsed on submit.
  const [scale, setScale] = useState('5')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const { job, error } = useJob(jobId)
  const running = busy || job?.status === 'running' || job?.status === 'queued'
  const mPer1000px = Number(scale)
  const scaleOk = scale.trim() !== '' && Number.isFinite(mPer1000px) && mPer1000px > 0 && mPer1000px <= 100

  const pick = (kind: 'scene' | 'pattern') => async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setErr(null)
    try { const u = await uploadImage(f, kind); (kind === 'scene' ? setScene : setPattern)(u) } catch (x) { setErr((x as Error).message) }
  }

  const run = async () => {
    if (!scene || !pattern || !scaleOk) return
    setBusy(true); setErr(null); setJobId(null)
    try {
      const r = await createJob({ kind: 'floor', scene_id: scene.id, pattern_id: pattern.id, material, m_per_1000px: mPer1000px })
      setJobId(r.job_id)
    }
    catch (x) { setErr((x as Error).message) }
    finally { setBusy(false) }
  }

  return (
    <section>
      <div className="row">
        <label className="file">Room photo <input type="file" accept="image/*" onChange={pick('scene')} />
          {scene && <><img src={scene.original_url} alt="" title="original upload" /><span className="muted">→</span><img src={scene.url} alt="" title="pre-cleaned (DDRM, 5 DDIM steps) - this is what the pipeline uses" /></>}</label>
        <label className="file">Tile pattern <input type="file" accept="image/*" onChange={pick('pattern')} />{pattern && <img src={pattern.url} alt="" />}</label>
        <label title="Surface finish. Both are the same dielectric (4% reflectance head-on); roughness sets how wide the reflection lobe is, so matte scatters the room into a soft haze toward the horizon while glossy mirrors it.">
          Finish <select value={material} onChange={(e) => setMaterial(e.target.value as Finish)} disabled={running}>
            <option value="smooth-matte">Matte (honed / matte-finish tile)</option>
            <option value="smooth-glossy">Glossy (polished / sealed)</option>
          </select></label>
        <label title="A pattern image has no physical size of its own, so this sets the scale: how much floor 1000 px of the pattern covers. One repeat is then (pattern width / 1000) x this, and its depth follows the pattern's aspect ratio.">
          1000 px = <input type="text" inputMode="decimal" value={scale} size={4}
            onChange={(e) => setScale(e.target.value)} disabled={running}
            aria-label="metres of floor per 1000 pixels of pattern" /> m of floor
        </label>
        {pattern && scaleOk && (
          <span className="muted" title="one repeat of this pattern on the floor">
            tile {(pattern.width / 1000 * mPer1000px).toFixed(2)} x {(pattern.height / 1000 * mPer1000px).toFixed(2)} m
          </span>
        )}
        <button onClick={run} disabled={!scene || !pattern || !scaleOk || running}>Generate</button>
      </div>
      {err && <p className="err">{err}</p>}
      {jobId && <JobView job={job} error={error} />}
    </section>
  )
}
