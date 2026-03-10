import type { PredictionResult, WordResult, SampleItem, ModelInfo } from '../types'

export async function predictCharacter(
  imageDataUrl: string,
  mode: string
): Promise<PredictionResult> {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl, mode }),
  })
  if (!res.ok) throw new Error('Prediction failed')
  return res.json()
}

export async function predictWord(imageDataUrl: string): Promise<WordResult> {
  const res = await fetch('/api/predict-word', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl }),
  })
  if (!res.ok) throw new Error('Word prediction failed')
  return res.json()
}

export async function fetchRandomSamples(): Promise<SampleItem[]> {
  const res = await fetch('/api/random-samples')
  if (!res.ok) throw new Error('Failed to fetch samples')
  return res.json()
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch('/api/model-info')
  if (!res.ok) throw new Error('Failed to fetch model info')
  return res.json()
}
