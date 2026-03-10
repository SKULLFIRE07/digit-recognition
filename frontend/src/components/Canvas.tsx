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
      <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-40 mb-3">Draw</div>

      {/* Stroke slider */}
      <div className="flex gap-2 items-center mb-3">
        <label className="text-[10px] font-semibold uppercase tracking-[1.5px] opacity-30 whitespace-nowrap">
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

      {/* Canvas */}
      <div className="border-2 border-white dark:border-white border-opacity-100 mb-4 leading-[0] relative">
        <canvas
          ref={canvasRef}
          width={420}
          height={420}
          className="block w-full h-[420px] bg-black"
        />
      </div>

      {/* Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handlePredict}
          className="flex-1 flex items-center justify-center gap-2 py-3.5 border-2 border-white dark:border-white bg-white dark:bg-white text-black font-bold text-xs uppercase tracking-[2px] hover:bg-neutral-200 dark:hover:bg-neutral-200 transition-all"
        >
          <Sparkles className="w-4 h-4" />
          Predict
        </button>
        <button
          onClick={undo}
          className="px-4 py-3.5 border-2 border-white/30 dark:border-white/30 bg-transparent font-bold text-xs uppercase tracking-[2px] hover:bg-white hover:text-black dark:hover:bg-white dark:hover:text-black transition-all"
        >
          <Undo2 className="w-4 h-4" />
        </button>
        <button
          onClick={clear}
          className="px-4 py-3.5 border-2 border-white/30 dark:border-white/30 bg-transparent font-bold text-xs uppercase tracking-[2px] hover:bg-white hover:text-black dark:hover:bg-white dark:hover:text-black transition-all"
        >
          <Eraser className="w-4 h-4" />
        </button>
      </div>

      {/* Keyboard shortcuts */}
      <div className="text-center text-[10px] opacity-20 mt-2.5 tracking-wide">
        <kbd className="font-mono text-[9px] border border-white/20 px-1.5 py-0.5 rounded-sm">Enter</kbd> predict
        {' · '}
        <kbd className="font-mono text-[9px] border border-white/20 px-1.5 py-0.5 rounded-sm">Esc</kbd> clear
        {' · '}
        <kbd className="font-mono text-[9px] border border-white/20 px-1.5 py-0.5 rounded-sm">Space</kbd> undo
      </div>
    </div>
  )
}
