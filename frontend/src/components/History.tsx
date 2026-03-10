interface HistoryItem {
  label: string
  confidence: number
}

interface Props {
  items: HistoryItem[]
}

export default function History({ items }: Props) {
  if (items.length === 0) return null

  return (
    <div className="mt-8">
      <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-35 mb-3">History</div>
      <div className="flex gap-2 flex-wrap">
        {items.map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-themed-muted rounded-lg font-mono animate-fade-in"
          >
            <span className="text-lg font-bold">{item.label}</span>
            {item.confidence > 0 && (
              <span className="text-[10px] opacity-30">{item.confidence}%</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
