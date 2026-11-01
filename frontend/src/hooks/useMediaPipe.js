import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
]
const HAND_COLORS = ['#00a000', '#0050c8']

const drawHands = (canvas, video, hands) => {
  if (!canvas) return
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
  }
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const ordered = [...hands].sort((a, b) => a[0][0] - b[0][0])
  ordered.forEach((hand, handIdx) => {
    ctx.strokeStyle = HAND_COLORS[handIdx % HAND_COLORS.length]
    ctx.lineWidth = 3
    HAND_CONNECTIONS.forEach(([a, b]) => {
      ctx.beginPath()
      ctx.moveTo(hand[a][0], hand[a][1])
      ctx.lineTo(hand[b][0], hand[b][1])
      ctx.stroke()
    })
    ctx.fillStyle = '#ff0000'
    hand.forEach(([x, y]) => {
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, 2 * Math.PI)
      ctx.fill()
    })
  })
}

// Calls onDetection(hands) once per processed video frame. hands is a list of
// 0-2 hands, each 21 [x, y, z] points in video pixels (z shares x's scale) -
// the same units the training script stores, so the ML service can render
// them with the training renderer unchanged.
export const useMediaPipe = (onDetection, enabled = true) => {
  const [isRunning, setIsRunning] = useState(false)
  const [handCount, setHandCount] = useState(0)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const onDetectionRef = useRef(onDetection)
  onDetectionRef.current = onDetection

  useEffect(() => {
    if (!enabled) return undefined
    if (!navigator.mediaDevices?.getUserMedia) {
      console.warn('MediaDevices not supported')
      return undefined
    }

    let stream
    let cancelled = false
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      .then((nextStream) => {
        stream = nextStream
        if (cancelled || !videoRef.current) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch((error) => console.error('Video play failed:', error.message))
        }
        setIsRunning(true)
      })
      .catch((error) => console.error('Webcam access error:', error))

    return () => {
      cancelled = true
      stream?.getTracks().forEach((track) => track.stop())
      setIsRunning(false)
    }
  }, [enabled])

  useEffect(() => {
    if (!enabled) return undefined
    let detector
    let cancelled = false
    let animationId
    let lastVideoTime = -1
    let lastHandCount = 0

    const detectFrame = () => {
      const video = videoRef.current
      if (detector && video?.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime
        const output = detector.detectForVideo(video, performance.now())
        const width = video.videoWidth
        const height = video.videoHeight
        const hands = (output.landmarks || []).map((hand) =>
          hand.map((lm) => [lm.x * width, lm.y * height, lm.z * width])
        )
        drawHands(canvasRef.current, video, hands)
        if (hands.length !== lastHandCount) {
          lastHandCount = hands.length
          setHandCount(hands.length)
        }
        // The preview is CSS-mirrored. Phone front-camera photos are mirrored
        // the same way, so the landmarks sent to the model are mirrored to
        // match them. The overlay stays in raw video coordinates.
        const mirrored = hands.map((hand) => hand.map(([x, y, z]) => [width - x, y, z]))
        onDetectionRef.current?.(mirrored)
      }
      if (!cancelled) animationId = requestAnimationFrame(detectFrame)
    }

    const initializeDetector = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
      )
      detector = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: import.meta.env.VITE_HAND_LANDMARKER_MODEL_URL ||
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        },
        numHands: 2,
        runningMode: 'VIDEO',
        // Same threshold as scripts/process_original_images.py, so a hand the
        // training photos kept is not dropped live.
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })
      if (cancelled) {
        detector.close()
        return
      }
      detectFrame()
    }

    initializeDetector().catch((error) => console.error('MediaPipe initialization error:', error))

    return () => {
      cancelled = true
      cancelAnimationFrame(animationId)
      detector?.close()
    }
  }, [enabled])

  return { videoRef, canvasRef, isRunning, handCount }
}
