import { useCanvas } from '../hooks/useCanvas'
import { Eraser, Undo2, Sparkles } from 'lucide-react'

interface Props {
  onPredict: (dataUrl: string) => void
  canvasHook: ReturnType<typeof useCanvas>
}

export default function Canvas({ onPredict, canvasHook }: Props) {
  const { canvasRef, penSize, setPenSize, hasContent, clear, undo, getDataURL } = canvasHook

  const handlePredict = () => {
    if (!hasContent) return
    onPredict(getDataURL())
  }

  return (
    <div className="flex flex-col">
      <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-35 mb-3">Draw</div>

      <div className="flex gap-2 items-center mb-3">
        <label className="text-[10px] font-semibold uppercase tracking-[1.5px] opacity-25 whitespace-nowrap">
          Stroke
        </label>
        <input
          type="range"
          min={8}
          max={42}
          value={penSize}
          onChange={(e) => setPenSize(Number(e.target.value))}
          className="flex-1"
        />
      </div>

      <div className="border-2 border-themed mb-4 leading-[0] rounded-lg overflow-hidden shadow-lg">
        <canvas
          ref={canvasRef}
          width={420}
          height={420}
          className="block w-full h-[420px] bg-black"
        />
      </div>

      <div className="flex gap-2">
        <button
          onClick={handlePredict}
          className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-lg border-2 border-themed bg-themed text-themed-inv font-bold text-xs uppercase tracking-[2px] hover:opacity-80 transition-opacity"
        >
          <Sparkles className="w-4 h-4" />
          Predict
        </button>
        <button
          onClick={undo}
          title="Undo"
          className="px-4 py-3.5 rounded-lg border-2 border-themed-muted bg-transparent font-bold text-xs uppercase tracking-[2px] hover:bg-themed hover:text-themed-inv transition-all"
        >
          <Undo2 className="w-4 h-4" />
        </button>
        <button
          onClick={clear}
          title="Clear"
          className="px-4 py-3.5 rounded-lg border-2 border-themed-muted bg-transparent font-bold text-xs uppercase tracking-[2px] hover:bg-themed hover:text-themed-inv transition-all"
        >
          <Eraser className="w-4 h-4" />
        </button>
      </div>

      <div className="text-center text-[10px] opacity-15 mt-3 tracking-wide">
        <kbd className="font-mono text-[9px] border border-themed-muted px-1.5 py-0.5 rounded">Enter</kbd> predict
        {' · '}
        <kbd className="font-mono text-[9px] border border-themed-muted px-1.5 py-0.5 rounded">Esc</kbd> clear
        {' · '}
        <kbd className="font-mono text-[9px] border border-themed-muted px-1.5 py-0.5 rounded">Space</kbd> undo
      </div>
    </div>
  )
}
