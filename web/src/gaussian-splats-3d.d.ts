// The package ships no types. Only what this app calls is declared.
declare module '@mkkellogg/gaussian-splats-3d' {
  import * as THREE from 'three'

  export interface SplatSceneOptions {
    splatAlphaRemovalThreshold?: number
    showLoadingUI?: boolean
    position?: number[]
    rotation?: number[]     // quaternion
    scale?: number[]
    onProgress?: (percent: number, label: string, stage: unknown) => void
  }

  export interface DropInViewerOptions {
    gpuAcceleratedSort?: boolean
    sharedMemoryForWorkers?: boolean
    integerBasedSort?: boolean
    halfPrecisionCovariancesOnGPU?: boolean
    antialiased?: boolean
loadingSpinner?: boolean
  }

  export class DropInViewer extends THREE.Group {
    constructor(options?: DropInViewerOptions)
    addSplatScene(path: string, options?: SplatSceneOptions): Promise<void> & { abort?: () => void }
    dispose(): Promise<void>
    splatMesh: THREE.Object3D | null
  }
}
