import { useEffect, useRef, useState } from 'react'
import { createJob, samProposals, uploadImage, uploadMask, type Box, type Proposal, type Region, type Upload } from './api'
import { BoxCanvas, type BoxCanvasHandle, type Overlay } from './BoxCanvas'
import { JobView } from './JobView'
import { useJob } from './useJob'

type Step = 'upload' | 'source' | 'mask' | 'brush' | 'target' | 'review' | 'running'
interface Move { src_box: Box; dst_box: Box; mask_id: string; mask_url: string }
interface Draft { src_box?: Box; proposals?: Proposal[]; chosen?: { mask_id: string; mask_url: string }; decode_ms?: number }

const HINT: Record<Step, string> = {
  upload: 'Upload a photo.',
  source: 'Drag a box around the object you want to move.',
  mask: 'Pick the mask that fits the object, or draw your own.',
  brush: 'Paint the object (red). Switch to erase to remove parts.',
  target: 'Drag a box where the object should go (its centre is the destination).',
  review: 'Add another object or generate.',
  running: '',
}

export function ObjectEdit() {
  const [scene, setScene] = useState<Upload | null>(null)
  const [step, setStep] = useState<Step>('upload')
  const [moves, setMoves] = useState<Move[]>([])
  const [draft, setDraft] = useState<Draft>({})
  const [hover, setHover] = useState<Proposal | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [brushSize, setBrushSize] = useState(24)
  const [erase, setErase] = useState(false)
  const [removal, setRemoval] = useState('omnipaint')
  const [insertion, setInsertion] = useState('omnipaint')
  const [omniMode, setOmniMode] = useState<'window' | 'full'>('full')
  const [srcRegion, setSrcRegion] = useState<Region>('full')
  const [dstRegion, setDstRegion] = useState<Region>('full')
  const [margin, setMargin] = useState(0.35)
  const [srFactor, setSrFactor] = useState<1 | 2 | 3 | 4>(1)
  const [showOriginal, setShowOriginal] = useState(false)
  const [omniSteps, setOmniSteps] = useState(28)
  const [jobId, setJobId] = useState<string | null>(null)
  const canvas = useRef<BoxCanvasHandle>(null)
  const { job, error } = useJob(jobId)

  const fail = (e: unknown) => setErr((e as Error).message)

  const pick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setErr(null); setBusy('uploading')
    try { setScene(await uploadImage(f, 'scene')); setMoves([]); setDraft({}); setJobId(null); setStep('source') }
    catch (x) { fail(x) } finally { setBusy(null) }
  }

  const onBox = async (b: Box) => {
    if (!scene) return
    if (step === 'source') {
      setDraft({ src_box: b }); setErr(null); setBusy('segmenting')
      try {
        const r = await samProposals(scene.id, b)
        setDraft({ src_box: b, proposals: r.proposals, decode_ms: r.decode_ms }); setStep('mask')
      } catch (x) { fail(x); setDraft({}) } finally { setBusy(null) }
    } else if (step === 'target' && draft.src_box && draft.chosen) {
      setMoves([...moves, { src_box: draft.src_box, dst_box: b, ...draft.chosen }])
      setDraft({}); setHover(null); setStep('review')
    }
  }

  const choose = (p: Proposal) => { setDraft({ ...draft, chosen: { mask_id: p.mask_id, mask_url: p.mask_url } }); setHover(null); setStep('target') }

  const startBrush = async (from?: Proposal) => {
    setStep('brush'); setHover(null)
    await canvas.current?.loadMask(from?.mask_url ?? null)
  }
  const finishBrush = async () => {
    if (!scene) return
    setBusy('saving mask'); setErr(null)
    try {
      const blob = await canvas.current!.exportMask()
      const m = await uploadMask(scene.id, blob)
      setDraft({ ...draft, chosen: { mask_id: m.id, mask_url: m.url } }); setStep('target')
    } catch (x) { fail(x) } finally { setBusy(null) }
  }

  const generate = async () => {
    if (!scene || !moves.length) return
    setBusy('submitting'); setErr(null)
    try {
      const r = await createJob({ kind: 'object', scene_id: scene.id, removal, insertion,
        omnipaint_mode: omniMode, omnipaint_res: 1024, omnipaint_steps: omniSteps,
        src_region: srcRegion, dst_region: dstRegion, region_margin: margin, sr_factor: srFactor,
        moves: moves.map((m) => ({ src_box: m.src_box, dst_box: m.dst_box, mask_id: m.mask_id })) })
      setJobId(r.job_id); setStep('running')
    } catch (x) { fail(x) } finally { setBusy(null) }
  }

  useEffect(() => { if (job && (job.status === 'done' || job.status === 'failed')) setStep('review') }, [job])

  const overlays: Overlay[] = moves.flatMap((m, i) => [
    { box: m.src_box, color: '#ff5050', label: `#${i + 1}` }, { box: m.dst_box, color: '#40e070', label: `#${i + 1} →` }])
  if (draft.src_box) overlays.push({ box: draft.src_box, color: '#ffffff', label: 'source' })
  const maskUrl = hover?.mask_url ?? draft.chosen?.mask_url ?? null
  const mode = step === 'source' || step === 'target' ? 'box' : step === 'brush' ? 'brush' : 'view'
  const running = job?.status === 'queued' || job?.status === 'running'

  return (
    <section>
      <div className="row">
        <label className="file">Room photo <input type="file" accept="image/*" onChange={pick} /></label>
        {scene && <span className="muted">{scene.width}×{scene.height}</span>}
        {scene?.precleaned && (
          <span className="muted" title="Every stage (SAM masks, removal, insertion) runs on the pre-cleaned image: DDRM identity, noise 0.005, 5 DDIM steps">
            pre-cleaned (DDRM, 5 DDIM steps) ·{' '}
            <button className={showOriginal ? 'ghost' : 'active'} onClick={() => setShowOriginal(false)}>cleaned</button>{' '}
            <button className={showOriginal ? 'active' : 'ghost'} onClick={() => setShowOriginal(true)}>original</button>
          </span>
        )}
        {busy && <span className="muted">{busy === 'uploading' ? 'uploading + pre-cleaning (DDRM)' : busy}…</span>}
      </div>
      {err && <p className="err">{err}</p>}
      {scene && (
        <>
          <p className="hint">{HINT[step]}</p>
          <BoxCanvas ref={canvas} src={showOriginal ? scene.original_url : scene.url} width={scene.width} height={scene.height} mode={mode} overlays={overlays}
            arrows={moves.map((m) => [m.src_box, m.dst_box])} maskUrl={maskUrl} brushSize={brushSize} erase={erase} onBox={onBox} />

          {step === 'mask' && draft.proposals && (
            <div className="row proposals">
              {draft.proposals.map((p, i) => (
                <button key={p.mask_id} className="thumb" onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} onClick={() => choose(p)}>
                  <MaskThumb scene={scene} box={draft.src_box!} maskUrl={p.mask_url} />
                  <span>{i + 1} · IoU {p.iou.toFixed(2)} · {Math.round(p.area / 1000)}k px</span>
                </button>
              ))}
              <button onClick={() => startBrush(draft.proposals?.[0])}>Draw my own</button>
              <button className="ghost" onClick={() => { setDraft({}); setStep('source') }}>Redo box</button>
              {draft.decode_ms !== undefined && <span className="muted">SAM decode {draft.decode_ms} ms</span>}
            </div>
          )}

          {step === 'brush' && (
            <div className="row">
              <label>Brush {brushSize}px <input type="range" min={4} max={120} value={brushSize} onChange={(e) => setBrushSize(+e.target.value)} /></label>
              <button className={erase ? '' : 'active'} onClick={() => setErase(false)}>Paint</button>
              <button className={erase ? 'active' : ''} onClick={() => setErase(true)}>Erase</button>
              <button className="ghost" onClick={() => canvas.current?.clearMask()}>Clear</button>
              <button onClick={finishBrush} disabled={!!busy}>Use this mask</button>
              <button className="ghost" onClick={() => setStep('mask')}>Back</button>
            </div>
          )}

          {step === 'target' && <div className="row"><button className="ghost" onClick={() => setStep('mask')}>Back to masks</button></div>}

          {(step === 'review' || step === 'running') && (
            <div className="review">
              <ul>
                {moves.map((m, i) => (
                  <li key={i}>
                    <MaskThumb scene={scene} box={m.src_box} maskUrl={m.mask_url} size={48} />
                    #{i + 1}: ({m.src_box.join(', ')}) → ({m.dst_box.join(', ')})
                    {!running && <button className="ghost" onClick={() => setMoves(moves.filter((_, j) => j !== i))}>✕</button>}
                  </li>
                ))}
              </ul>
              <div className="row">
                <label title="What the remover may repaint at the source. mask = only the object (old shadow stays); dilated / box = a band around it so the shadow can be removed; full = everything the model renders (OmniPaint / FreeFine only)">
                  Source edit <RegionSelect value={srcRegion} onChange={setSrcRegion} disabled={running} /></label>
                <label title="What the inserter may repaint at the target. mask = only the pasted object; dilated / box = a band where a new shadow can be rendered; full = everything the model renders">
                  Target edit <RegionSelect value={dstRegion} onChange={setDstRegion} disabled={running} /></label>
                {(srcRegion === 'dilated' || srcRegion === 'box' || dstRegion === 'dilated' || dstRegion === 'box') && (
                  <label title="dilated / box margin as a fraction of the object size">margin <input type="number" min={0} max={2} step={0.05} value={margin} onChange={(e) => setMargin(+e.target.value)} disabled={running} style={{ width: 60 }} /></label>
                )}
              </div>
              <div className="row">
                <label title="Blur + subsample the photo by this factor before removal / insertion (faster, less memory), then GS-PnP / GSDRUNet super-resolution back to native; only edited pixels come from the SR image">
                  Processing resolution <select value={srFactor} onChange={(e) => setSrFactor(+e.target.value as 1 | 2 | 3 | 4)} disabled={running}>
                    <option value={1}>native{scene ? ` (${scene.width}×${scene.height})` : ''}</option>
                    {([2, 3, 4] as const).map((f) => <option key={f} value={f}>1/{f}{scene ? ` (${Math.floor(scene.width / f)}×${Math.floor(scene.height / f)}) + SR` : ''}</option>)}
                  </select></label>
              </div>
              <div className="row">
                <label>Removal <select value={removal} onChange={(e) => setRemoval(e.target.value)} disabled={running}>
                  <option value="omnipaint">OmniPaint / FLUX (~9 min per hole)</option><option value="lama">LaMa + refiner (~1 min)</option><option value="lama-plain">LaMa (~5 s)</option><option value="freefine">FreeFine bg-gen (~1.5 min)</option></select></label>
                <label>Insertion <select value={insertion} onChange={(e) => setInsertion(e.target.value)} disabled={running}>
                  <option value="omnipaint">OmniPaint / FLUX (~12 min per object)</option><option value="freefine">FreeFine (~1.5 min per object)</option><option value="paste">Paste only</option></select></label>
                {(removal === 'omnipaint' || insertion === 'omnipaint') && (
                  <>
                    <label>OmniPaint canvas <select value={omniMode} onChange={(e) => setOmniMode(e.target.value as 'window' | 'full')} disabled={running}>
                      <option value="full">full image (1024 long side)</option><option value="window">per-object 512 window (faster)</option></select></label>
                    <label>steps <input type="number" min={1} max={50} value={omniSteps} onChange={(e) => setOmniSteps(+e.target.value)} disabled={running} style={{ width: 52 }} /></label>
                  </>
                )}
                <button className="ghost" onClick={() => { setDraft({}); setStep('source') }} disabled={running}>Add another object</button>
                <button onClick={generate} disabled={running || !moves.length || !!busy}>Generate</button>
              </div>
            </div>
          )}
        </>
      )}
      {jobId && <JobView job={job} error={error} />}
    </section>
  )
}

function RegionSelect({ value, onChange, disabled }: { value: Region; onChange: (r: Region) => void; disabled: boolean }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value as Region)} disabled={disabled}>
      <option value="mask">SAM mask only</option>
      <option value="dilated">dilated mask (shadow band)</option>
      <option value="box">bounding box + margin</option>
      <option value="full">full image</option>
    </select>
  )
}

/** Crop of the scene around `box` with the mask tinted on top. */
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
      // tint: draw the mask into a temp canvas, keep only its white pixels as red
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
