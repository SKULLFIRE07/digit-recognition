export interface PredictionResult {
  prediction: number
  label: string
  confidence: number
  probabilities: ProbabilityItem[]
}

export interface ProbabilityItem {
  label: string
  value: number
  index: number
}

export interface WordResult {
  word: string
  characters: {
    label: string
    confidence: number
    bbox: number[]
  }[]
}

export interface SampleItem {
  image: string
  true_label: string
  predicted: string
  correct: boolean
}

export interface ModelInfo {
  accuracy: number
  num_classes: number
  epoch: number
  labels: string[]
  model_type: string
  dataset: string
  input_size: string
  total_labels: number
}

export type RecognitionMode = 'digit' | 'letter' | 'all' | 'word'
