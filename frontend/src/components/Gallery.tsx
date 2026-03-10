import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { fetchRandomSamples } from '../api/predict'
import { SampleItem } from '../types'

export default function Gallery() {
  const [samples, setSamples] = useState<SampleItem[]>([])
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    fetchRandomSamples()
      .then(setSamples)
      .catch(() => setSamples([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <div className="mt-14 pt-8 border-t-2 border-white dark:border-white">
      <div className="flex items-center justify-between mb-5">
        <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-40">
          Test Samples
        </div>
        <button
          onClick={load}
          className="flex items-center gap-2 px-4 py-2 border border-white/20 dark:border-white/20 bg-transparent text-[10px] font-bold uppercase tracking-[2px] hover:border-white hover:bg-white hover:text-black dark:hover:border-white dark:hover:bg-white dark:hover:text-black transition-all"
        >
          <RefreshCw className="w-3 h-3" />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="text-center opacity-20 py-10 text-sm">Loading...</div>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3">
          {samples.map((s, i) => (
            <div
              key={i}
              className="border border-white/15 dark:border-white/15 p-2.5 text-center hover:border-white/50 dark:hover:border-white/50 transition-colors"
            >
              <img
                src={s.image}
                alt={s.true_label}
                className="w-full block mb-2"
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
