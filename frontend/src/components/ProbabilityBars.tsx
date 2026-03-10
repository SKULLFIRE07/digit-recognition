import { ProbabilityItem } from '../types'

interface Props {
  probabilities: ProbabilityItem[]
  predictedLabel: string
  maxBars?: number
}

export default function ProbabilityBars({ probabilities, predictedLabel, maxBars = 15 }: Props) {
  const topItems = probabilities.slice(0, maxBars)

  if (topItems.length === 0) {
    return (
      <div className="opacity-15 text-center py-10 text-sm uppercase tracking-wider">
        Predictions will appear here
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
      {topItems.map((item, i) => {
        const isTop = item.label === predictedLabel
        return (
          <div
            key={item.label}
            className="flex items-center gap-3 h-7 animate-fade-in"
            style={{ animationDelay: `${i * 25}ms` }}
          >
            <span
              className={`font-mono text-base font-bold w-6 text-right transition-opacity duration-300 ${
                isTop ? 'opacity-100' : 'opacity-12'
              }`}
            >
              {item.label}
            </span>
            <div className="flex-1 h-1.5 bg-bar-track overflow-hidden rounded-full">
              <div
                className="h-full bg-bar rounded-full transition-all duration-700"
                style={{
                  width: `${item.value}%`,
                  transitionTimingFunction: 'cubic-bezier(0.22, 1, 0.36, 1)',
                }}
              />
            </div>
            <span
              className={`font-mono text-xs font-semibold w-14 text-right transition-opacity duration-300 ${
                isTop ? 'opacity-100' : 'opacity-12'
              }`}
            >
              {item.value > 0.1 ? `${item.value.toFixed(1)}%` : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
