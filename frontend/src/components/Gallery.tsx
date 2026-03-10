import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { fetchRandomSamples } from '../api/predict'
import type { SampleItem } from '../types'

export default function Gallery() {
  const [samples, setSamples] = useState<SampleItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  const load = () => {
    setLoading(true)
    setError(false)
    fetchRandomSamples()
      .then(setSamples)
      .catch(() => { setSamples([]); setError(true) })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <div className="mt-14 pt-8 border-t-2 border-themed">
      <div className="flex items-center justify-between mb-5">
        <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-35">
          Test Samples
        </div>
        <button
          onClick={load}
          className="flex items-center gap-2 px-4 py-2 border border-themed-muted rounded-lg bg-transparent text-[10px] font-bold uppercase tracking-[2px] hover:bg-themed hover:text-themed-inv transition-all"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {loading && !error ? (
        <div className="text-center opacity-15 py-10 text-sm uppercase tracking-wider">Loading...</div>
      ) : error ? (
        <div className="text-center opacity-20 py-10 text-sm">
          Start the backend to see test samples
        </div>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3">
          {samples.map((s, i) => (
            <div
              key={i}
              className="border border-themed-muted rounded-lg p-2.5 text-center hover:border-themed transition-colors animate-fade-in"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <img
                src={s.image}
                alt={s.true_label}
                className="w-full block mb-2 rounded"
                style={{ imageRendering: 'pixelated' }}
              />
              <div className={`font-mono text-xs font-semibold ${s.correct ? '' : 'opacity-40 line-through'}`}>
                {s.correct ? s.predicted : `${s.predicted} → ${s.true_label}`}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
