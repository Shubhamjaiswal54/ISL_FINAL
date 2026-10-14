import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'

export const useMediaPipe = () => {
  const [landmarks, setLandmarks] = useState(null)
  const [handLabel, setHandLabel] = useState('Right')
  const [isRunning, setIsRunning] = useState(false)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn('MediaDevices not supported')
      return
    }

    let stream
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then(nextStream => {
        stream = nextStream
          if (!videoRef.current) {
            console.error('[Frontend -> Camera] Video element is not mounted')
            return
          }
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => {
            console.log('[Frontend -> Camera] Video metadata loaded:', {
              width: videoRef.current.videoWidth,
              height: videoRef.current.videoHeight,
              readyState: videoRef.current.readyState,
            })
            videoRef.current.play().catch(error => console.error('[Frontend -> Camera] Video play failed:', error.message))
          }
        setIsRunning(true)
        console.log('[Frontend -> Camera] Webcam stream attached')
      })
      .catch(err => console.error('Webcam access error:', err))

    return () => stream?.getTracks().forEach(track => track.stop())
  }, [])

  useEffect(() => {
    let detector
    let cancelled = false
    let animationId
    let detectionCount = 0

    const initializeDetector = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
      )
      detector = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: import.meta.env.VITE_HAND_LANDMARKER_MODEL_URL ||
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        },
        numHands: 1,
        runningMode: 'VIDEO',
        // Defaults (0.5) let stale/low-confidence tracked detections through
        // when the hand leaves frame or is mid-motion, which the classifier
        // then confidently (mis)labels as some letter. Raise the bar instead
        // of trusting a shaky detection.
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.7,
        minTrackingConfidence: 0.7,
      })
      console.log('[Frontend -> MediaPipe] HandLandmarker initialized')
      if (cancelled) {
        detector.unload()
        return
      }

      const detectFrame = () => {
        const video = videoRef.current
        if (!cancelled && detector && video?.readyState >= 2 && video.currentTime > 0) {
          detectionCount += 1
          const output = detector.detectForVideo(video, performance.now())
          const nextLandmarks = output.landmarks?.[0] || null
          const nextHandLabel = output.handedness?.[0]?.[0]?.categoryName || 'Right'
          if (nextLandmarks) {
            console.log('[MediaPipe -> Frontend] Landmarks detected:', {
              count: nextLandmarks.length,
              hand: nextHandLabel,
              sample: nextLandmarks[0],
            })
          } else if (detectionCount === 1 || detectionCount % 60 === 0) {
            console.warn('[MediaPipe -> Frontend] Video frame received, but no hand detected:', {
              detectionCount,
              currentTime: video.currentTime,
              readyState: video.readyState,
            })
          }
          setHandLabel(nextHandLabel)
          setLandmarks(nextLandmarks)
        }
        if (!cancelled) animationId = requestAnimationFrame(detectFrame)
      }
      detectFrame()
    }

    initializeDetector().catch(error => console.error('MediaPipe initialization error:', error))

    return () => {
      cancelled = true
      cancelAnimationFrame(animationId)
      detector?.unload()
    }
  }, [])

  return { startCamera: videoRef, stopCamera: canvasRef, landmarks, handLabel, isRunning }
}