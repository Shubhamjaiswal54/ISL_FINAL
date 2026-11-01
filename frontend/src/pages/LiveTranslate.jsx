import { useCallback, useEffect, useRef } from 'react'
import { useMediaPipe } from '../hooks/useMediaPipe'
import { useSocket } from '../hooks/useSocket'
import { useTranslationStore } from '../context/TranslationContext.jsx'
import { useSpeech } from '../hooks/useSpeech'

// Detection runs at display refresh rate; the server only needs ~20 fps to
// classify held poses. Empty frames let a lowered hand sign the same letter
// again. They do not end a word or a sentence.
const HAND_FRAME_INTERVAL_MS = 50
const EMPTY_FRAME_INTERVAL_MS = 200

export default function LiveTranslate() {
  const { emitFrame, emitControl, socket, mlStatus } = useSocket()
  const {
    setSentence, setSentences, setGlosses, setCurrentWord, setSuggestions, setPrediction,
    isListening, registerControl,
  } = useTranslationStore()
  const { speak } = useSpeech()
  const lastSentRef = useRef(0)
  const glossKeyRef = useRef('')
  const isListeningRef = useRef(isListening)
  isListeningRef.current = isListening

  useEffect(() => {
    registerControl(emitControl)
    return () => registerControl(null)
  }, [emitControl, registerControl])

  const handleDetection = useCallback((hands) => {
    if (!isListeningRef.current) return
    const now = performance.now()
    const interval = hands.length ? HAND_FRAME_INTERVAL_MS : EMPTY_FRAME_INTERVAL_MS
    if (now - lastSentRef.current < interval) return
    lastSentRef.current = now
    emitFrame(hands)
  }, [emitFrame])

  const { videoRef, canvasRef, isRunning, handCount } = useMediaPipe(handleDetection, isListening)

  useEffect(() => {
    if (!socket) return undefined

    const applyWords = (data) => {
      const nextGlosses = data.glosses || []
      const nextKey = nextGlosses.join('\u0000')
      if (glossKeyRef.current && glossKeyRef.current !== nextKey) setSentence('')
      glossKeyRef.current = nextKey
      setGlosses(nextGlosses)
      setCurrentWord(data.currentWord || '')
      setSuggestions(data.suggestions || [])
    }

    const handlePrediction = (data) => {
      if (data.type === 'sentence') {
        setSentence(data.sentence)
        if (data.sentence) {
          setSentences((items) => [...items, data.sentence].slice(-6))
          speak(data.sentence)
        }
        return
      }
      applyWords(data)
      if (data.type === 'glosses') return
      setPrediction({
        label: data.predictedLabel,
        confidence: data.confidence,
        settled: data.settled,
        accepted: data.accepted,
        handCount: data.handCount,
        threshold: data.threshold,
        control: data.control || null,
        controlFired: Boolean(data.controlFired),
      })
    }

    socket.on('prediction', handlePrediction)
    return () => socket.off('prediction', handlePrediction)
  }, [socket, setGlosses, setCurrentWord, setSuggestions, setSentence, setSentences, setPrediction, speak])

  const mlLabel = mlStatus.connected ? 'Model connected' : (mlStatus.error || 'Model offline')

  return (
    <section className="overflow-hidden rounded-3xl border border-white/10 bg-card">
      <div className="flex items-center justify-between px-5 py-4">
        <div>
          <h2 className="text-lg font-semibold">Camera</h2>
          <p className="text-sm text-muted">Front camera, mirrored so it matches a mirror.</p>
        </div>
      </div>
      <div className="relative mx-4 mb-4 aspect-[4/3] overflow-hidden rounded-2xl bg-black">
        <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover -scale-x-100" />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />
        {!isRunning && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/70 text-center">
            <p className="text-base font-medium">Camera is off</p>
            <p className="max-w-xs text-sm text-muted">Press Start. Press Space when a word is finished.</p>
          </div>
        )}
        <div className="absolute left-3 top-3 flex flex-wrap gap-2">
          <StatusPill live={isRunning} label={isRunning ? 'Camera on' : 'Camera off'} />
          <StatusPill live={handCount > 0} label={handCount === 1 ? '1 hand' : `${handCount} hands`} />
          <StatusPill live={mlStatus.connected} label={mlLabel} />
        </div>
      </div>
    </section>
  )
}

function StatusPill({ live, label }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-black/70 px-3 py-1 text-xs text-foreground backdrop-blur">
      <span className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-accent' : 'bg-muted'}`} />
      {label}
    </span>
  )
}
