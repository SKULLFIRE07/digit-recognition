import { useState, useEffect, useCallback } from 'react'
import { RecognitionMode, PredictionResult, WordResult } from './types'
import { predictCharacter, predictWord } from './api/predict'
import { useCanvas } from './hooks/useCanvas'
import Header from './components/Header'
import ModeSelector from './components/ModeSelector'
import Canvas from './components/Canvas'
import PredictionDisplay from './components/PredictionDisplay'
import ProbabilityBars from './components/ProbabilityBars'
import History from './components/History'
import Gallery from './components/Gallery'
import './App.css'

interface HistoryItem {
  label: string
  confidence: number
}

function App() {
  const [mode, setMode] = useState<RecognitionMode>('all')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [wordResult, setWordResult] = useState<WordResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const canvasHook = useCanvas()

  const handlePredict = useCallback(async (dataUrl: string) => {
    setLoading(true)
    setResult(null)
    setWordResult(null)

    try {
      if (mode === 'word') {
        const res = await predictWord(dataUrl)
        setWordResult(res)
        if (res.word) {
          setHistory((prev) => [{ label: res.word, confidence: 0 }, ...prev].slice(0, 20))
        }
      } else {
        const res = await predictCharacter(dataUrl, mode)
        setResult(res)
        setHistory((prev) => [{ label: res.label, confidence: res.confidence }, ...prev].slice(0, 20))
      }
    } catch (e) {
      console.error('Prediction failed:', e)
    } finally {
      setLoading(false)
    }
  }, [mode])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        if (canvasHook.hasContent) {
          handlePredict(canvasHook.getDataURL())
        }
      }
      if (e.key === 'Escape') {
        canvasHook.clear()
        setResult(null)
        setWordResult(null)
      }
      if (e.key === ' ') {
        e.preventDefault()
        canvasHook.undo()
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [canvasHook, handlePredict])

  return (
    <div className="max-w-[1100px] mx-auto px-6 py-14">
      <Header />
      <ModeSelector mode={mode} onModeChange={setMode} />

      <div className="grid grid-cols-1 lg:grid-cols-[420px_1fr] gap-12 items-start">
        <div>
          <Canvas onPredict={handlePredict} canvasHook={canvasHook} />
          <History items={history} />
        </div>

        <div>
          <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-40 mb-3">
            Prediction
          </div>
          <PredictionDisplay
            result={result}
            wordResult={wordResult}
            loading={loading}
            isWordMode={mode === 'word'}
          />
          <div className="text-[10px] font-bold uppercase tracking-[3px] opacity-40 mb-3">
            Distribution
          </div>
          <ProbabilityBars
            probabilities={result?.probabilities || []}
            predictedLabel={result?.label || ''}
          />
        </div>
      </div>

      <Gallery />

      <footer className="mt-12 pt-6 border-t border-themed-subtle flex gap-8 flex-wrap">
        <FooterItem label="Model" value="CNN (PyTorch)" />
        <FooterItem label="Dataset" value="EMNIST Balanced" />
        <FooterItem label="Classes" value="47 (0-9, A-Z, a-z)" />
        <FooterItem label="Input" value="28 × 28 px" />
        <FooterItem label="Architecture" value="React + Flask" />
      </footer>
    </div>
  )
}

function FooterItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] font-semibold uppercase tracking-[2px] opacity-20">{label}</span>
      <span className="font-mono text-xs font-semibold opacity-50">{value}</span>
    </div>
  )
}

export default App
