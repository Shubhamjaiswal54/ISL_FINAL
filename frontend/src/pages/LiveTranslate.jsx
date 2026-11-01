import { useCallback, useEffect, useRef } from 'react'
import { useMediaPipe } from '../hooks/useMediaPipe'
import { useSocket } from '../hooks/useSocket'
import { useTranslationStore } from '../context/TranslationContext.jsx'
import { useSpeech } from '../hooks/useSpeech'

// Detection runs at display refresh rate; the server only needs ~20 fps to
// classify held poses. No-hand frames only drive the word/sentence gap
// timers, so a lower rate is enough for those.
const HAND_FRAME_INTERVAL_MS = 50
const EMPTY_FRAME_INTERVAL_MS = 200

export default function LiveTranslate() {
  const { emitFrame, emitControl, socket, mlStatus } = useSocket()
  const { setSentence, setGlosses, setPrediction, isListening } = useTranslationStore()
  const { speak } = useSpeech()
  const lastSentRef = useRef(0)
  const isListeningRef = useRef(isListening)
  const wasListeningRef = useRef(isListening)
  isListeningRef.current = isListening

  const handleDetection = useCallback((hands) => {
    if (!isListeningRef.current) return
    const now = performance.now()
    const interval = hands.length ? HAND_FRAME_INTERVAL_MS : EMPTY_FRAME_INTERVAL_MS
    if (now - lastSentRef.current < interval) return
    lastSentRef.current = now
    emitFrame(hands)
  }, [emitFrame])

  const { videoRef, canvasRef, isRunning, handCount } = useMediaPipe(handleDetection, isListening)

  // Stop turns the camera off, so no more "hands lowered" frames will arrive
  // to close the sentence. Ask the server to finish it now.
  useEffect(() => {
    if (wasListeningRef.current && !isListening) {
      emitControl('finish')
      setGlosses([])
      setPrediction(null)
    }
    wasListeningRef.current = isListening
  }, [isListening, emitControl, setGlosses, setPrediction])

  useEffect(() => {
    if (!socket) return undefined

    const handlePrediction = (data) => {
      if (data.type === 'sentence') {
        setSentence(data.sentence)
        // A sentence that finishes while the next word is already being
        // spelled must not wipe those new letters.
        setGlosses((current) => (
          current.join('\u0000') === (data.glosses || []).join('\u0000') ? [] : current
        ))
        if (data.sentence) speak(data.sentence)
        return
      }
      setGlosses(data.glosses || [])
      setPrediction({
        label: data.predictedLabel,
        confidence: data.confidence,
        settled: data.settled,
        accepted: data.accepted,
        handCount: data.handCount,
      })
    }

    socket.on('prediction', handlePrediction)
    return () => socket.off('prediction', handlePrediction)
  }, [socket, setGlosses, setSentence, setPrediction, speak])

  return (
    <div>
      {/* Mirrored for a natural selfie view; landmarks are sent unmirrored. */}
      <div className="relative">
        <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-lg -scale-x-100" />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none -scale-x-100" />
      </div>
      <p className="mt-2 text-sm text-gray-500">
        Camera: {isRunning ? 'on' : 'off'} · Hands: {handCount} · ML service: {mlStatus.connected ? 'connected' : (mlStatus.error || 'disconnected')}
      </p>
    </div>
  )
}
