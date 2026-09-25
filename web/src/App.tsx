import { useState } from 'react'
import { FloorEdit } from './FloorEdit'
import { ImageTo3D } from './ImageTo3D'
import { ObjectEdit } from './ObjectEdit'

type Tab = 'floor' | 'object' | 'image3d'

export default function App() {
  const [tab, setTab] = useState<Tab>('floor')
  return (
    <main>
      <header>
        <h1>imageEditor</h1>
        <nav>
          <button className={tab === 'floor' ? 'active' : ''} onClick={() => setTab('floor')}>Floor edit</button>
          <button className={tab === 'object' ? 'active' : ''} onClick={() => setTab('object')}>Object move</button>
          <button className={tab === 'image3d' ? 'active' : ''} onClick={() => setTab('image3d')}>Image to 3D</button>
        </nav>
      </header>
      {tab === 'floor' && <FloorEdit />}
      {tab === 'object' && <ObjectEdit />}
      {tab === 'image3d' && <ImageTo3D />}
    </main>
  )
}
