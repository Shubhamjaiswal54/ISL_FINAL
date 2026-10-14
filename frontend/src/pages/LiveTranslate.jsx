import { useEffect } from 'react'
import { useMediaPipe } from '../hooks/useMediaPipe'
import { useSocket } from '../hooks/useSocket'
import { useTranslationStore } from '../context/TranslationContext.jsx'
import { useSpeech } from '../hooks/useSpeech'
import { normalizeLandmarks } from '../services/landmarkNormalizer.js'
import PredictionDisplay from '../components/PredictionDisplay.jsx'

export default function LiveTranslate() {
  const { startCamera, stopCamera, landmarks, handLabel, isRunning } = useMediaPipe()
  const { emitLandmark, socket } = useSocket()
  const { sentence, setSentence, glosses, setGlosses, isListening, setIsListening } = useTranslationStore()
  const { speak } = useSpeech()

  useEffect(() => {
    if (!socket) {
      console.log('[Prediction] Waiting for Socket.IO connection')
      return undefined
    }

    const handlePrediction = ({ sentence: nextSentence, glosses: nextGlosses, predictedLabel, accepted, word }) => {
      console.log('[Backend -> Frontend] Received prediction:', {
        predictedLabel,
        accepted,
        word,
        sentence: nextSentence,
        glosses: nextGlosses,
      })
      if (nextSentence) setSentence(nextSentence)
      if (nextGlosses) setGlosses(nextGlosses)
    }

    socket.on('prediction', handlePrediction)
    return () => socket.off('prediction', handlePrediction)
  }, [socket, setGlosses, setSentence])

  useEffect(() => {
    let animationId
    let frameCount = 0
    const handleFrame = () => {
      frameCount += 1
      if (frameCount % 60 === 0) {
        console.log('[Frame loop] Status:', {
          listening: isListening,
          landmarks: landmarks?.length ?? 0,
          socketConnected: socket?.connected ?? false,
          cameraRunning: isRunning,
        })
      }
      if (landmarks && isListening) {
        const normalizedLandmarks = normalizeLandmarks(landmarks, handLabel)
        console.log('[Landmarks -> Socket] Normalized vector:', {
          rawLength: landmarks.length,
          normalizedLength: normalizedLandmarks?.length ?? 0,
          hand: handLabel,
          firstValues: normalizedLandmarks?.slice(0, 4),
        })
        if (normalizedLandmarks) emitLandmark(normalizedLandmarks)
      } else if (frameCount % 60 === 0) {
        console.warn('[Frame loop] No data sent:', {
          hasLandmarks: Boolean(landmarks),
          listening: isListening,
        })
      }
      animationId = requestAnimationFrame(handleFrame)
    }
    handleFrame()
    return () => cancelAnimationFrame(animationId)
  }, [landmarks, handLabel, isListening, emitLandmark, socket, isRunning])

  return (
    <div>
      <video ref={startCamera} autoPlay playsInline className="w-full rounded-t-lg" />
      <canvas ref={stopCamera} className="w-full rounded-b-lg h-64" />
      <PredictionDisplay sentence={sentence} setSentence={setSentence} />
    </div>
  )
}