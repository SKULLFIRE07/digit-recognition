import { useEffect, useState } from 'react'
import { Sun, Moon, Brain } from 'lucide-react'
import { fetchModelInfo } from '../api/predict'
import { ModelInfo } from '../types'

export default function Header() {
  const [dark, setDark] = useState(true)
  const [info, setInfo] = useState<ModelInfo | null>(null)

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(() => {})
  }, [])

  const toggleTheme = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle('dark', next)
  }

  return (
    <header className="flex items-end justify-between mb-12 flex-wrap gap-4">
      <div className="flex items-center gap-3">
        <Brain className="w-10 h-10 opacity-80" />
        <div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter leading-none">
            CharacterAI
          </h1>
          <p className="text-xs uppercase tracking-[3px] opacity-30 mt-1">
            Handwriting Recognition
          </p>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {info && (
          <div className="flex gap-5">
            <Stat label="Accuracy" value={`${info.accuracy}%`} />
            <Stat label="Model" value={info.model_type} />
            <Stat label="Classes" value={String(info.total_labels)} />
          </div>
        )}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg border border-white/10 dark:border-white/10 hover:border-white/30 transition-colors"
        >
          {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-right">
      <div className="text-[10px] font-semibold uppercase tracking-[2px] opacity-30">
        {label}
      </div>
      <div className="font-mono text-lg font-bold">{value}</div>
    </div>
  )
}
