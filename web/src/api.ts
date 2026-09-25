// The only module that talks to the server. Coordinates are always natural (image) pixels.

export type Box = [number, number, number, number] // x0, y0, x1, y1
export type Region = 'mask' | 'dilated' | 'box' | 'full'  // how far around the object a stage may repaint

export interface Upload { id: string; url: string; original_url: string; width: number; height: number; precleaned?: boolean }
export interface Proposal { mask_id: string; mask_url: string; iou: number; area: number }
export interface MoveSpec { src_box: Box; dst_box: Box; mask_id: string }
export interface StageRow { stage: string; model?: string; time_s?: number; gen_s?: number; gpu_peak_gb?: number }
export interface JobResult { output_url: string; intermediates: { name: string; url: string }[]; metrics: Record<string, unknown>; model_url?: string | null; splat_url?: string | null }
export interface JobStatus {
  job_id: string
  kind: 'floor' | 'object' | 'image3d'
  status: 'queued' | 'running' | 'done' | 'failed'
  stage: string | null
  stages_done: StageRow[]
  elapsed_s: number
  position: number
  result: JobResult | null
  error: string | null
}

async function check<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* not json */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

export async function uploadImage(file: File, kind: 'scene' | 'pattern'): Promise<Upload> {
  const fd = new FormData()
  fd.append('file', file)
  return check(await fetch(`/api/uploads?kind=${kind}`, { method: 'POST', body: fd }))
}

export async function samProposals(image_id: string, box: Box): Promise<{ proposals: Proposal[]; decode_ms: number }> {
  return check(await fetch('/api/sam/proposals', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id, box }),
  }))
}

export async function uploadMask(image_id: string, png: Blob): Promise<{ id: string; url: string }> {
  const fd = new FormData()
  fd.append('file', png, 'mask.png')
  return check(await fetch(`/api/masks?image_id=${encodeURIComponent(image_id)}`, { method: 'POST', body: fd }))
}

export type Finish = 'smooth-matte' | 'smooth-glossy'

export async function createJob(body:
  | { kind: 'floor'; scene_id: string; pattern_id: string; material?: Finish; m_per_1000px?: number }
  | { kind: 'object'; scene_id: string; moves: MoveSpec[]; removal?: string; insertion?: string;
      omnipaint_mode?: 'window' | 'full'; omnipaint_res?: number; omnipaint_steps?: number;
      src_region?: Region; dst_region?: Region; region_margin?: number; sr_factor?: 1 | 2 | 3 | 4 }
  | { kind: 'image3d'; scene_id: string; mask_id: string; seed?: number; ss_steps?: number; slat_steps?: number },
): Promise<{ job_id: string; status: string; position: number }> {
  return check(await fetch('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }))
}

export async function getJob(id: string): Promise<JobStatus> {
  return check(await fetch(`/api/jobs/${id}`))
}
