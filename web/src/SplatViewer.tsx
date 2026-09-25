import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { DropInViewer } from '@mkkellogg/gaussian-splats-3d'

// Renders the TRELLIS Gaussian (the 3DGS .ply that `Gaussian.save_ply` writes) rather than the mesh.
// This is the representation DIRECT actually displays - its demo decodes formats=["gaussian"] and
// feeds the splats to viser, using the mesh only for normal maps. The mesh decoder's per-vertex
// colour channel measures ~37% less saturated than the input image, which is why the mesh viewer
// looked washed out.
//
// DropInViewer is a THREE.Group, so it drops into an ordinary three.js scene: the splat sort runs
// off the camera via an onBeforeRender hook on an internal callback mesh. Drag orbits the camera;
// the X/Y/Z props rotate the object, matching DIRECT's viser gizmo.

interface Props {
  url: string
  rot: { x: number; y: number; z: number }   // degrees
  autoRotate?: boolean
  height?: number
  // Filled with a function that renders one frame and returns it as a PNG data URL, so the panel
  // above can offer a "Save PNG" button. Cleared when the viewer tears down.
  captureRef?: React.MutableRefObject<((scale?: number) => string) | null>
}

export function SplatViewer({ url, rot, autoRotate = false, height = 460, captureRef }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const pivot = useRef<THREE.Group | null>(null)
  const controls = useRef<OrbitControls | null>(null)
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [detail, setDetail] = useState('')

  useEffect(() => {
    const el = host.current
    if (!el) return
    const w = el.clientWidth || 640
    setStatus('loading'); setDetail('')

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(w, height)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, w / height, 0.05, 100)
    camera.position.set(0, 0, 2.6)
    camera.up.set(0, 1, 0)

    const ctl = new OrbitControls(camera, renderer.domElement)
    ctl.enableDamping = true
    controls.current = ctl

    const group = new THREE.Group()
    scene.add(group)
    pivot.current = group

    // save_ply writes the splats in the same Y-up frame as the .glb, already centred on the origin
    // in roughly a unit box, so no extra fitting is needed here.
    const viewer = new DropInViewer({
      gpuAcceleratedSort: false,      // the iGPU is busy; the CPU sort is plenty at ~325k splats
      sharedMemoryForWorkers: false,  // avoids needing COOP/COEP cross-origin isolation headers
    })
    group.add(viewer)

    let disposed = false
    viewer
      .addSplatScene(url, { showLoadingUI: false, splatAlphaRemovalThreshold: 5 })
      .then(() => { if (!disposed) setStatus('ready') })
      .catch((e: unknown) => {
        if (disposed) return
        setStatus('error'); setDetail((e as Error)?.message ?? String(e))
        console.error('splat load failed', e)
      })

    let raf = 0
    const tick = () => {
      ctl.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    tick()

    // Grab the canvas at `scale` x the on-screen resolution. The buffer is not preserved between
    // frames, so the read has to follow a render in the same task; raising the pixel ratio for that
    // one frame gives a print-size PNG without changing what the viewer looks like.
    if (captureRef) {
      captureRef.current = (scale = 2) => {
        const size = renderer.getSize(new THREE.Vector2())
        const dpr = renderer.getPixelRatio()
        renderer.setPixelRatio(dpr * scale)
        renderer.setSize(size.x, size.y, false)
        renderer.render(scene, camera)
        const png = renderer.domElement.toDataURL('image/png')
        renderer.setPixelRatio(dpr)
        renderer.setSize(size.x, size.y)
        renderer.render(scene, camera)
        return png
      }
    }

    const onResize = () => {
      const nw = el.clientWidth || w
      camera.aspect = nw / height
      camera.updateProjectionMatrix()
      renderer.setSize(nw, height)
    }
    window.addEventListener('resize', onResize)

    return () => {
      disposed = true
      if (captureRef) captureRef.current = null
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      ctl.dispose()
      // dispose() tears down the splat workers; it is async and may reject if a load is still in flight
      Promise.resolve(viewer.dispose()).catch(() => {})
      renderer.dispose()
      el.removeChild(renderer.domElement)
      pivot.current = null
      controls.current = null
    }
  }, [url, height, captureRef])

  useEffect(() => {
    const d = Math.PI / 180
    pivot.current?.rotation.set(rot.x * d, rot.y * d, rot.z * d)
  }, [rot.x, rot.y, rot.z])

  useEffect(() => {
    if (!controls.current) return
    controls.current.autoRotate = autoRotate
    controls.current.autoRotateSpeed = 2.0
  }, [autoRotate])

  return (
    <div className="viewer3d" style={{ height, position: 'relative' }} ref={host}>
      {status !== 'ready' && (
        <span className="viewer-note">
          {status === 'loading' ? 'loading splats…' : `splat load failed: ${detail}`}
        </span>
      )}
    </div>
  )
}
