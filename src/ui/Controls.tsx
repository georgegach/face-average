import { useStore } from '../state/store'
import { EditPanel } from './panels/EditPanel'
import { BabyPanel } from './panels/BabyPanel'
import { AveragePanel } from './panels/AveragePanel'
import { MorphPanel } from './panels/MorphPanel'
import { ReplacePanel } from './panels/ReplacePanel'
import { EnhancePanel } from './panels/EnhancePanel'

export function Controls() {
  const mode = useStore((s) => s.mode)
  if (mode === 'edit') return <EditPanel />
  if (mode === 'baby') return <BabyPanel />
  if (mode === 'average') return <AveragePanel />
  if (mode === 'morph') return <MorphPanel />
  if (mode === 'replace') return <ReplacePanel />
  return <EnhancePanel />
}
