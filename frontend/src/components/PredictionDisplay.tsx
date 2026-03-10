import { PredictionResult, WordResult } from '../types'

interface Props {
  result: PredictionResult | null
  wordResult: WordResult | null
  loading: boolean
  isWordMode: boolean
}

export default function PredictionDisplay({ result, wordResult, loading, isWordMode }: Props) {
  return (
    <div className="text-center py-5 pb-8 border-b-2 border-white dark:border-white mb-7 min-h-[190px] flex flex-col items-center justify-center">
      {loading && (
        <div className="w-5 h-5 border-2 border-white/10 border-t-white rounded-full animate-spin" />
      )}

      {!loading && !result && !wordResult && (
        <span className="text-sm opacity-20 uppercase tracking-wider">
          {isWordMode ? 'Draw a word' : 'Draw a character'}
        </span>
      )}

      {!loading && isWordMode && wordResult && (
        <div className="animate-scale-in">
          <div className="font-mono text-6xl md:text-7xl font-bold tracking-wider">
            {wordResult.word || '—'}
          </div>
          <div className="mt-3 flex gap-1 justify-center flex-wrap">
            {wordResult.characters.map((ch, i) => (
              <span
                key={i}
                className="inline-flex flex-col items-center px-2 py-1 border border-white/10 rounded text-xs"
              >
                <span className="font-mono font-bold text-lg">{ch.label}</span>
                <span className="opacity-40 text-[10px]">{ch.confidence.toFixed(0)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {!loading && !isWordMode && result && (
        <>
          <div className="font-mono text-[150px] font-bold leading-none animate-scale-in">
            {result.label}
          </div>
          <div className="mt-1 font-mono text-sm opacity-50">
            <span className="font-bold opacity-100">{result.confidence}%</span> confidence
          </div>
        </>
      )}
    </div>
  )
}
