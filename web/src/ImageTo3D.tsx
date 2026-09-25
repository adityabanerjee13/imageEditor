import { useEffect, useRef, useState } from 'react'
import { createJob, samProposals, uploadImage, type Box, type Proposal, type Upload } from './api'
import { BoxCanvas } from './BoxCanvas'
import { JobView } from './JobView'
import { MeshViewer } from './MeshViewer'
import { SplatViewer } from './SplatViewer'
import { useJob } from './useJob'

// Upload -> box -> pick a SAM mask -> reconstruct (imageto3D / TRELLIS) -> rotate the mesh in 3 axes.
// Deliberately no brush step: the selected proposal goes straight to the reconstruction.

type Step = 'upload' | 'select' | 'mask' | 'running' | 'view'

const HINT: Record<Step, string> = {
  upload: 'Upload a photo.',
  select: 'Drag a box around the object you want in 3D.',
  mask: 'Pick the mask that best fits the object.',
  running: 'Reconstructing — about two minutes on the iGPU.',
  view: 'Drag the model to orbit. The sliders rotate the object itself.',
}

const ZERO = { x: 0, y: 0, z: 0 }

export function ImageTo3D() {
  const [scene, setScene] = useState<Upload | null>(null)
  const [step, setStep] = useState<Step>('upload')
  const [box, setBox] = useState<Box | null>(null)
  const [proposals, setProposals] = useState<Proposal[] | null>(null)
  const [chosen, setChosen] = useState<Proposal | null>(null)
  const [hover, setHover] = useState<Proposal | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [seed, setSeed] = useState(42)
  const [jobId, setJobId] = useState<string | null>(null)
  const [rot, setRot] = useState(ZERO)
  const [spin, setSpin] = useState(false)
  const [rep, setRep] = useState<'splat' | 'mesh'>('splat')
  const capture = useRef<((scale?: number) => string) | null>(null)
  const { job, error } = useJob(jobId)

  const fail = (e: unknown) => setErr((e as Error).message)

  /** Save what the viewer is showing, at 2x resolution, named after the pose it was taken at. */
  const savePng = () => {
    const png = capture.current?.(2)
    if (!png) return
    const deg = (v: number) => Math.round(v)
    const a = document.createElement('a')
    a.href = png
    a.download = `${jobId?.slice(0, 8) ?? 'view'}-${rep}-x${deg(rot.x)}y${deg(rot.y)}z${deg(rot.z)}.png`
    a.click()
  }
  const done = job?.status === 'done'
  const modelUrl = done ? job.result?.model_url ?? null : null
  const splatUrl = done ? job.result?.splat_url ?? null : null

  useEffect(() => { if (modelUrl || splatUrl) setStep('view') }, [modelUrl, splatUrl])
  useEffect(() => { if (job?.status === 'failed') setStep('mask') }, [job?.status])

  const pick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setErr(null); setBusy('uploading')
    try {
      setScene(await uploadImage(f, 'scene'))
      setBox(null); setProposals(null); setChosen(null); setJobId(null); setRot(ZERO); setStep('select')
    } catch (x) { fail(x) } finally { setBusy(null) }
  }

  const onBox = async (b: Box) => {
    if (!scene || step !== 'select') return
    setBox(b); setErr(null); setBusy('segmenting')
    try {
      const r = await samProposals(scene.id, b)
      setProposals(r.proposals); setStep('mask')
    } catch (x) { fail(x); setBox(null) } finally { setBusy(null) }
  }

  const generate = async (p: Proposal) => {
    if (!scene) return
    setChosen(p); setBusy('submitting'); setErr(null)
    try {
      const r = await createJob({ kind: 'image3d', scene_id: scene.id, mask_id: p.mask_id, seed })
      setJobId(r.job_id); setRot(ZERO); setStep('running')
    } catch (x) { fail(x) } finally { setBusy(null) }
  }

  const running = job?.status === 'queued' || job?.status === 'running'
  const mode = step === 'select' ? 'box' : 'view'
  const overlays = box ? [{ box, color: '#ffffff', label: 'object' }] : []

  return (
    <section>
      <div className="row">
        <label className="file">Photo <input type="file" accept="image/*" onChange={pick} /></label>
        {scene && <span className="muted">{scene.width}×{scene.height}</span>}
        {busy && <span className="muted">{busy}…</span>}
      </div>
      {err && <p className="err">{err}</p>}

      {scene && (
        <>
          <p className="hint">{HINT[step]}</p>

          {step !== 'view' && (
            <BoxCanvas src={scene.url} width={scene.width} height={scene.height} mode={mode}
              overlays={overlays} maskUrl={hover?.mask_url ?? chosen?.mask_url ?? null} onBox={onBox} />
          )}

          {step === 'mask' && proposals && (
            <div className="row proposals">
              {proposals.map((p, i) => (
                <button key={p.mask_id} className="thumb" disabled={!!busy}
                  onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} onClick={() => generate(p)}>
                  <MaskThumb scene={scene} box={box!} maskUrl={p.mask_url} />
                  <span>{i + 1} · IoU {p.iou.toFixed(2)} · {Math.round(p.area / 1000)}k px</span>
                </button>
              ))}
              <button className="ghost" onClick={() => { setBox(null); setProposals(null); setStep('select') }}>Redo box</button>
              <label>seed <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} style={{ width: 70 }} /></label>
            </div>
          )}

          {step === 'view' && (modelUrl || splatUrl) && (
            <>
              {rep === 'splat' && splatUrl
                ? <SplatViewer url={splatUrl} rot={rot} autoRotate={spin} captureRef={capture} />
                : modelUrl && <MeshViewer url={modelUrl} rot={rot} autoRotate={spin} captureRef={capture} />}
              <div className="row">
                <span title="TRELLIS decodes both from the same latent. DIRECT renders the Gaussian; the mesh decoder's per-vertex colours measure ~37% less saturated.">
                  <button className={rep === 'splat' ? 'active' : ''} onClick={() => setRep('splat')} disabled={!splatUrl}>Gaussian splat</button>{' '}
                  <button className={rep === 'mesh' ? 'active' : ''} onClick={() => setRep('mesh')} disabled={!modelUrl}>Mesh</button>
                </span>
              </div>
              <div className="row">
                {(['x', 'y', 'z'] as const).map((ax) => (
                  <label key={ax}>
                    {ax.toUpperCase()} {Math.round(rot[ax])}°
                    <input type="range" min={-180} max={180} value={rot[ax]}
                      onChange={(e) => setRot({ ...rot, [ax]: +e.target.value })} />
                  </label>
                ))}
                <button className={spin ? 'active' : ''} onClick={() => setSpin(!spin)}>{spin ? 'Stop' : 'Spin'}</button>
                <button className="ghost" onClick={() => setRot(ZERO)}>Reset</button>
                <button onClick={savePng}
                  title={rep === 'splat'
                    ? 'Save this view as a PNG at 2x resolution (black background)'
                    : 'Save this view as a PNG at 2x resolution (transparent background)'}>
                  Save PNG
                </button>
                {modelUrl && <a href={modelUrl} download>Download .glb</a>}
                {splatUrl && <a href={splatUrl} download>Download .ply</a>}
                <button className="ghost" onClick={() => { setJobId(null); setBox(null); setProposals(null); setChosen(null); setStep('select') }}>
                  Pick another object
                </button>
              </div>
            </>
          )}
        </>
      )}

      {jobId && (running || job?.status === 'failed') && <JobView job={job} error={error} />}
    </section>
  )
}

/** Crop of the scene around `box` with the mask tinted on top (same thumb as the object-move panel). */
function MaskThumb({ scene, box, maskUrl, size = 110 }: { scene: Upload; box: Box; maskUrl: string; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    let live = true
    const img = new Image(); img.src = scene.url
    const msk = new Image(); msk.src = maskUrl
    Promise.all([img.decode(), msk.decode()]).then(() => {
      if (!live || !ref.current) return
      const pad = 0.25 * Math.max(box[2] - box[0], box[3] - box[1])
      const x0 = Math.max(0, box[0] - pad), y0 = Math.max(0, box[1] - pad)
      const w = Math.min(scene.width, box[2] + pad) - x0, h = Math.min(scene.height, box[3] + pad) - y0
      const s = size / Math.max(w, h)
      const c = ref.current; c.width = Math.round(w * s); c.height = Math.round(h * s)
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, x0, y0, w, h, 0, 0, c.width, c.height)
      const t = document.createElement('canvas'); t.width = c.width; t.height = c.height
      const tc = t.getContext('2d')!
      tc.drawImage(msk, x0, y0, w, h, 0, 0, t.width, t.height)
      const d = tc.getImageData(0, 0, t.width, t.height); const p = d.data
      for (let i = 0; i < p.length; i += 4) { const on = p[i] > 127; p[i] = 255; p[i + 1] = 64; p[i + 2] = 64; p[i + 3] = on ? 140 : 0 }
      tc.putImageData(d, 0, 0)
      ctx.drawImage(t, 0, 0)
    })
    return () => { live = false }
  }, [scene, box, maskUrl, size])
  return <canvas ref={ref} />
}
