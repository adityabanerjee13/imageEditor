import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react'
import type { Box } from './api'

// Image + one overlay canvas. All coordinates handed in/out are natural image pixels; the component owns the
// display <-> natural conversion (naturalWidth / clientWidth). In 'brush' mode it keeps an offscreen RGBA mask
// canvas at natural resolution that the parent exports through the imperative handle.

export interface Overlay { box: Box; color: string; label?: string }
export interface BoxCanvasHandle {
  exportMask(): Promise<Blob>
  loadMask(url: string | null): Promise<void>
  clearMask(): void
}
interface Props {
  src: string
  width: number            // natural
  height: number
  mode: 'view' | 'box' | 'brush'
  overlays?: Overlay[]
  arrows?: [Box, Box][]    // centre -> centre
  maskUrl?: string | null  // shown tinted in view/box mode
  brushSize?: number       // natural px
  erase?: boolean
  onBox?: (b: Box) => void
  maxWidth?: number
}

const MASK_RGB = [255, 64, 64] as const

function centre(b: Box): [number, number] { return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] }

/** Load a white-on-black (or RGBA) mask PNG into an RGBA canvas: alpha = mask. */
async function maskToCanvas(url: string, w: number, h: number): Promise<HTMLCanvasElement> {
  const img = new Image()
  img.src = url
  await img.decode()
  const c = document.createElement('canvas'); c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  ctx.drawImage(img, 0, 0, w, h)
  const d = ctx.getImageData(0, 0, w, h)
  const p = d.data
  for (let i = 0; i < p.length; i += 4) {
    const on = p[i + 3] > 127 && (p[i] > 127 || p[i + 1] > 127 || p[i + 2] > 127)
    p[i] = MASK_RGB[0]; p[i + 1] = MASK_RGB[1]; p[i + 2] = MASK_RGB[2]; p[i + 3] = on ? 255 : 0
  }
  ctx.putImageData(d, 0, 0)
  return c
}

export const BoxCanvas = forwardRef<BoxCanvasHandle, Props>(function BoxCanvas(
  { src, width, height, mode, overlays = [], arrows = [], maskUrl = null, brushSize = 20, erase = false, onBox, maxWidth = 900 }, ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskRef = useRef<HTMLCanvasElement | null>(null)      // brush mask, natural size
  const viewMaskRef = useRef<HTMLCanvasElement | null>(null)  // tinted maskUrl for display
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const painting = useRef(false)
  const last = useRef<[number, number] | null>(null)
  const [cursor, setCursor] = useState<[number, number] | null>(null)
  const [, force] = useState(0)
  const redraw = () => force((n) => n + 1)

  const dispW = Math.min(maxWidth, width)
  const scale = dispW / width
  const dispH = Math.round(height * scale)

  const ensureMask = useCallback(() => {
    if (!maskRef.current) {
      const c = document.createElement('canvas'); c.width = width; c.height = height
      maskRef.current = c
    }
    return maskRef.current
  }, [width, height])

  useImperativeHandle(ref, () => ({
    async exportMask() {
      const c = ensureMask()
      return new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/png'))
    },
    async loadMask(url) {
      maskRef.current = url ? await maskToCanvas(url, width, height) : null
      ensureMask(); redraw()
    },
    clearMask() { maskRef.current = null; ensureMask(); redraw() },
  }), [ensureMask, width, height])

  // tinted copy of maskUrl for view/box modes
  useEffect(() => {
    let live = true
    if (!maskUrl) { viewMaskRef.current = null; redraw(); return }
    maskToCanvas(maskUrl, width, height).then((c) => { if (live) { viewMaskRef.current = c; redraw() } })
    return () => { live = false }
  }, [maskUrl, width, height])

  // natural coords of a pointer event
  const toNat = (e: React.PointerEvent): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect()
    return [
      Math.max(0, Math.min(width, (e.clientX - r.left) * (width / r.width))),
      Math.max(0, Math.min(height, (e.clientY - r.top) * (height / r.height))),
    ]
  }

  const paint = (from: [number, number] | null, to: [number, number]) => {
    const ctx = ensureMask().getContext('2d')!
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    ctx.strokeStyle = ctx.fillStyle = `rgb(${MASK_RGB.join(',')})`
    ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
    ctx.beginPath()
    if (from) { ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke() }
    else { ctx.arc(to[0], to[1], brushSize / 2, 0, Math.PI * 2); ctx.fill() }
  }

  const onDown = (e: React.PointerEvent) => {
    if (mode === 'view') return
    canvasRef.current!.setPointerCapture(e.pointerId)
    const [x, y] = toNat(e)
    if (mode === 'box') setDrag({ x0: x, y0: y, x1: x, y1: y })
    else { painting.current = true; last.current = [x, y]; paint(null, [x, y]); redraw() }
  }
  const onMove = (e: React.PointerEvent) => {
    const p = toNat(e)
    if (mode === 'brush') setCursor(p)
    if (mode === 'box' && drag) setDrag({ ...drag, x1: p[0], y1: p[1] })
    else if (mode === 'brush' && painting.current) { paint(last.current, p); last.current = p; redraw() }
  }
  const onUp = () => {
    if (mode === 'box' && drag) {
      const b: Box = [Math.round(Math.min(drag.x0, drag.x1)), Math.round(Math.min(drag.y0, drag.y1)),
        Math.round(Math.max(drag.x0, drag.x1)), Math.round(Math.max(drag.y0, drag.y1))]
      setDrag(null)
      if (b[2] - b[0] >= 4 && b[3] - b[1] >= 4) onBox?.(b)
    }
    painting.current = false; last.current = null
  }

  // draw everything
  useLayoutEffect(() => {
    const c = canvasRef.current; if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    ctx.save(); ctx.scale(scale, scale)
    ctx.globalAlpha = 0.55
    if (mode === 'brush' && maskRef.current) ctx.drawImage(maskRef.current, 0, 0)
    else if (mode !== 'brush' && viewMaskRef.current) ctx.drawImage(viewMaskRef.current, 0, 0)
    ctx.globalAlpha = 1
    ctx.lineWidth = 2 / scale
    for (const [a, b] of arrows) {
      const [x0, y0] = centre(a), [x1, y1] = centre(b)
      ctx.strokeStyle = '#ffd400'; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke()
    }
    ctx.font = `${13 / scale}px system-ui`
    for (const o of overlays) {
      ctx.strokeStyle = o.color; ctx.strokeRect(o.box[0], o.box[1], o.box[2] - o.box[0], o.box[3] - o.box[1])
      if (o.label) { ctx.fillStyle = o.color; ctx.fillText(o.label, o.box[0] + 3 / scale, o.box[1] + 14 / scale) }
    }
    if (drag) {
      ctx.strokeStyle = '#fff'; ctx.setLineDash([6 / scale, 4 / scale])
      ctx.strokeRect(Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.abs(drag.x1 - drag.x0), Math.abs(drag.y1 - drag.y0))
      ctx.setLineDash([])
    }
    if (mode === 'brush' && cursor) {
      ctx.strokeStyle = erase ? '#fff' : `rgb(${MASK_RGB.join(',')})`
      ctx.beginPath(); ctx.arc(cursor[0], cursor[1], brushSize / 2, 0, Math.PI * 2); ctx.stroke()
    }
    ctx.restore()
  })

  return (
    <div className="canvas-wrap" style={{ width: dispW, height: dispH }}>
      <img src={src} width={dispW} height={dispH} draggable={false} alt="" />
      <canvas
        ref={canvasRef} width={dispW} height={dispH}
        style={{ cursor: mode === 'view' ? 'default' : mode === 'box' ? 'crosshair' : 'none', touchAction: 'none' }}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={() => setCursor(null)}
      />
    </div>
  )
})
