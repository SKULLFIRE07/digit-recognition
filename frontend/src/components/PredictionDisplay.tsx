import { PredictionResult, WordResult } from '../types'

interface Props {
  result: PredictionResult | null
  wordResult: WordResult | null
  loading: boolean
  isWordMode: boolean
}

export default function PredictionDisplay({ result, wordResult, loading, isWordMode }: Props) {
  return (
    <div className="text-center py-6 pb-8 border-b-2 border-themed mb-7 min-h-[200px] flex flex-col items-center justify-center">
      {loading && (
        <div className="w-6 h-6 border-2 border-themed-muted border-t-current rounded-full animate-spin" />
      )}

      {!loading && !result && !wordResult && (
        <span className="text-sm opacity-15 uppercase tracking-[3px] font-medium">
          {isWordMode ? 'Draw a word' : 'Draw a character'}
        </span>
      )}

      {!loading && isWordMode && wordResult && (
        <div className="animate-scale-in">
          <div className="font-mono text-5xl sm:text-6xl md:text-7xl font-bold tracking-wider leading-tight">
            {wordResult.word || '—'}
          </div>
          <div className="mt-4 flex gap-1.5 justify-center flex-wrap">
            {wordResult.characters.map((ch, i) => (
              <span
                key={i}
                className="inline-flex flex-col items-center px-2.5 py-1.5 border border-themed-muted rounded-lg text-xs animate-fade-in"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <span className="font-mono font-bold text-xl">{ch.label}</span>
                <span className="opacity-35 text-[10px] font-medium">{ch.confidence.toFixed(0)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {!loading && !isWordMode && result && (
        <div className="animate-scale-in">
          <div className="font-mono text-[140px] sm:text-[160px] font-bold leading-none">
            {result.label}
          </div>
          <div className="mt-2 font-mono text-sm opacity-40">
            <span className="font-bold opacity-100">{result.confidence}%</span>
            <span className="ml-1">confidence</span>
          </div>
        </div>
      )}
    </div>
  )
}
