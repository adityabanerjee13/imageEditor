import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// Interactive view of a reconstructed mesh (.glb). Drag to orbit the camera; the X/Y/Z props rotate
// the *object*, which is what DIRECT's viser gizmo does before it renders a pose-conditioned view.
// The mesh carries per-vertex colours (FlexiCubes output), so it renders unlit - MeshBasicMaterial
// would flatten it, so a light rig plus vertexColors on a standard material keeps the form readable.

interface Props {
  url: string
  rot: { x: number; y: number; z: number }   // degrees
  autoRotate?: boolean
  height?: number
  // Filled with a function that renders one frame and returns it as a PNG data URL, so the panel
  // above can offer a "Save PNG" button. The capture frame clears with alpha 0, so the PNG has a
  // transparent background even though the live canvas is cleared opaque.
  captureRef?: React.MutableRefObject<((scale?: number) => string) | null>
}

export function MeshViewer({ url, rot, autoRotate = false, height = 460, captureRef }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const pivot = useRef<THREE.Group | null>(null)
  const controls = useRef<OrbitControls | null>(null)

  // Scene is built once per url; rotation/autoRotate are pushed in by the effects below so that
  // dragging a slider never rebuilds the renderer.
  useEffect(() => {
    const el = host.current
    if (!el) return
    const w = el.clientWidth || 640
    const h = height

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(w, h)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = null
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 100)
    camera.position.set(0, 0, 3)

    scene.add(new THREE.AmbientLight(0xffffff, 1.4))
    const key = new THREE.DirectionalLight(0xffffff, 1.6)
    key.position.set(2, 3, 4)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xffffff, 0.7)
    fill.position.set(-3, -1, -2)
    scene.add(fill)

    const ctl = new OrbitControls(camera, renderer.domElement)
    ctl.enableDamping = true
    ctl.target.set(0, 0, 0)
    controls.current = ctl

    const group = new THREE.Group()
    scene.add(group)
    pivot.current = group

    let disposed = false
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (disposed) return
        const obj = gltf.scene
        obj.traverse((o) => {
          const m = o as THREE.Mesh
          if (!m.isMesh) return
          m.material = new THREE.MeshStandardMaterial({
            vertexColors: true, roughness: 0.95, metalness: 0.0, side: THREE.DoubleSide,
          })
        })
        // Centre on the origin and scale into a unit box so any object frames the same way.
        const box = new THREE.Box3().setFromObject(obj)
        const size = box.getSize(new THREE.Vector3())
        const centre = box.getCenter(new THREE.Vector3())
        obj.position.sub(centre)
        const s = 1.6 / Math.max(size.x, size.y, size.z, 1e-6)
        obj.scale.setScalar(s)
        group.add(obj)
      },
      undefined,
      (e) => console.error('GLB load failed', e),
    )

    let raf = 0
    const tick = () => {
      ctl.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    tick()

    // Grab the canvas at `scale` x the on-screen resolution; see SplatViewer for why the read has
    // to follow a render in the same task.
    if (captureRef) {
      captureRef.current = (scale = 2) => {
        const size = renderer.getSize(new THREE.Vector2())
        const dpr = renderer.getPixelRatio()
        const alpha = renderer.getClearAlpha()
        renderer.setPixelRatio(dpr * scale)
        renderer.setSize(size.x, size.y, false)
        renderer.setClearAlpha(0)          // alpha:true alone still clears opaque: clearAlpha is 1
        renderer.render(scene, camera)
        const png = renderer.domElement.toDataURL('image/png')
        renderer.setClearAlpha(alpha)
        renderer.setPixelRatio(dpr)
        renderer.setSize(size.x, size.y)
        renderer.render(scene, camera)
        return png
      }
    }

    const onResize = () => {
      const nw = el.clientWidth || w
      camera.aspect = nw / h
      camera.updateProjectionMatrix()
      renderer.setSize(nw, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      disposed = true
      if (captureRef) captureRef.current = null
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      ctl.dispose()
      renderer.dispose()
      scene.traverse((o) => {
        const m = o as THREE.Mesh
        if (m.isMesh) {
          m.geometry?.dispose()
          const mat = m.material as THREE.Material | THREE.Material[]
          Array.isArray(mat) ? mat.forEach((x) => x.dispose()) : mat?.dispose()
        }
      })
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

  return <div className="viewer3d" ref={host} style={{ height }} />
}
