import type { RecognitionMode } from '../types'
import { Hash, Type, ALargeSmall, FileText } from 'lucide-react'

const modes: { id: RecognitionMode; label: string; icon: typeof Hash; desc: string }[] = [
  { id: 'digit', label: 'Digits', icon: Hash, desc: '0-9' },
  { id: 'letter', label: 'Letters', icon: Type, desc: 'A-Z, a-z' },
  { id: 'all', label: 'All', icon: ALargeSmall, desc: '47 classes' },
  { id: 'word', label: 'Word', icon: FileText, desc: 'Multi-char' },
]

interface Props {
  mode: RecognitionMode
  onModeChange: (mode: RecognitionMode) => void
}

export default function ModeSelector({ mode, onModeChange }: Props) {
  return (
    <div className="flex gap-2 mb-8">
      {modes.map((m) => {
        const Icon = m.icon
        const active = mode === m.id
        return (
          <button
            key={m.id}
            onClick={() => onModeChange(m.id)}
            className={`
              flex-1 flex flex-col items-center gap-1 py-3.5 px-2 rounded-xl border-2 transition-all duration-200
              ${active
                ? 'border-themed bg-themed text-themed-inv'
                : 'border-themed-muted hover:border-themed-subtle'
              }
            `}
          >
            <Icon className="w-5 h-5" />
            <span className="text-[11px] font-bold uppercase tracking-wider">{m.label}</span>
            <span className={`text-[9px] font-medium ${active ? 'opacity-50' : 'opacity-25'}`}>{m.desc}</span>
          </button>
        )
      })}
    </div>
  )
}
