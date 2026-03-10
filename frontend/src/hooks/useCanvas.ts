import { useRef, useEffect, useState, useCallback } from 'react'

interface Point {
  x: number
  y: number
  w: number
}

export function useCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [penSize, setPenSize] = useState(24)
  const [hasContent, setHasContent] = useState(false)
  const drawingRef = useRef(false)
  const lastPosRef = useRef({ x: 0, y: 0 })
  const strokesRef = useRef<Point[][]>([])
  const currentStrokeRef = useRef<Point[]>([])

  const getPos = useCallback((e: MouseEvent | TouchEvent) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const source = 'touches' in e ? e.touches[0] : e
    return {
      x: (source.clientX - rect.left) * (canvas.width / rect.width),
      y: (source.clientY - rect.top) * (canvas.height / rect.height),
    }
  }, [])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = '#fff'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    for (const stroke of strokesRef.current) {
      if (stroke.length === 1) {
        ctx.beginPath()
        ctx.arc(stroke[0].x, stroke[0].y, stroke[0].w / 2, 0, Math.PI * 2)
        ctx.fillStyle = '#fff'
        ctx.fill()
      } else {
        for (let i = 1; i < stroke.length; i++) {
          ctx.lineWidth = stroke[i].w
          ctx.beginPath()
          ctx.moveTo(stroke[i - 1].x, stroke[i - 1].y)
          ctx.lineTo(stroke[i].x, stroke[i].y)
          ctx.stroke()
        }
      }
    }
    setHasContent(strokesRef.current.length > 0)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#fff'

    const handleDown = (e: MouseEvent | TouchEvent) => {
      e.preventDefault()
      drawingRef.current = true
      currentStrokeRef.current = []
      const p = getPos(e)
      lastPosRef.current = p
      currentStrokeRef.current.push({ x: p.x, y: p.y, w: penSize })
      ctx.beginPath()
      ctx.arc(p.x, p.y, penSize / 2, 0, Math.PI * 2)
      ctx.fillStyle = '#fff'
      ctx.fill()
    }

    const handleMove = (e: MouseEvent | TouchEvent) => {
      e.preventDefault()
      if (!drawingRef.current) return
      const p = getPos(e)
      currentStrokeRef.current.push({ x: p.x, y: p.y, w: penSize })
      ctx.lineWidth = penSize
      ctx.strokeStyle = '#fff'
      ctx.beginPath()
      ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y)
      ctx.lineTo(p.x, p.y)
      ctx.stroke()
      lastPosRef.current = p
    }

    const handleUp = (e?: MouseEvent | TouchEvent) => {
      if (e) e.preventDefault()
      if (drawingRef.current && currentStrokeRef.current.length > 0) {
        strokesRef.current.push([...currentStrokeRef.current])
        setHasContent(true)
      }
      drawingRef.current = false
    }

    canvas.addEventListener('mousedown', handleDown)
    canvas.addEventListener('mousemove', handleMove)
    canvas.addEventListener('mouseup', handleUp)
    canvas.addEventListener('mouseleave', handleUp as EventListener)
    canvas.addEventListener('touchstart', handleDown)
    canvas.addEventListener('touchmove', handleMove)
    canvas.addEventListener('touchend', handleUp)

    return () => {
      canvas.removeEventListener('mousedown', handleDown)
      canvas.removeEventListener('mousemove', handleMove)
      canvas.removeEventListener('mouseup', handleUp)
      canvas.removeEventListener('mouseleave', handleUp as EventListener)
      canvas.removeEventListener('touchstart', handleDown)
      canvas.removeEventListener('touchmove', handleMove)
      canvas.removeEventListener('touchend', handleUp)
    }
  }, [getPos, penSize])

  const clear = useCallback(() => {
    strokesRef.current = []
    currentStrokeRef.current = []
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setHasContent(false)
  }, [])

  const undo = useCallback(() => {
    if (strokesRef.current.length === 0) return
    strokesRef.current.pop()
    redraw()
  }, [redraw])

  const getDataURL = useCallback(() => {
    return canvasRef.current?.toDataURL('image/png') || ''
  }, [])

  return { canvasRef, penSize, setPenSize, hasContent, clear, undo, getDataURL }
}
