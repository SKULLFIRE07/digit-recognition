import { useEffect, useState } from 'react'
import { Sun, Moon, Brain, Zap } from 'lucide-react'
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
      <div className="flex items-center gap-4">
        <div className="relative">
          <Brain className="w-11 h-11 opacity-90" />
          <Zap className="w-4 h-4 absolute -bottom-0.5 -right-0.5 opacity-60" />
        </div>
        <div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-[-0.04em] leading-none">
            CharacterAI
          </h1>
          <p className="text-[10px] uppercase tracking-[4px] opacity-25 mt-1.5 font-medium">
            Handwriting Recognition
          </p>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {info && (
          <div className="hidden sm:flex gap-5">
            <Stat label="Accuracy" value={`${info.accuracy}%`} highlight />
            <Stat label="Model" value="CNN" />
            <Stat label="Classes" value={String(info.total_labels)} />
          </div>
        )}
        <button
          onClick={toggleTheme}
          className="p-2.5 rounded-lg border-themed-muted border hover:opacity-70 transition-opacity"
          aria-label="Toggle theme"
        >
          {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  )
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="text-right">
      <div className="text-[10px] font-semibold uppercase tracking-[2px] opacity-25">
        {label}
      </div>
      <div className={`font-mono text-lg font-bold ${highlight ? 'opacity-100' : 'opacity-70'}`}>
        {value}
      </div>
    </div>
  )
}
