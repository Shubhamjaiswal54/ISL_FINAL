import { useMediaPipe } from '../hooks/useMediaPipe'
import { useSocket } from '../hooks/useSocket'

export default function CameraFeed() {
  const { startCamera, stopCamera, landmarks, isRunning } = useMediaPipe()
  const { emitLandmark } = useSocket()

  const handleFrame = () => {
    if (landmarks && isRunning) {
      emitLandmark(landmarks)
    }
    requestAnimationFrame(handleFrame)
  }

  return (
    <div className="relative">
      <video ref={startCamera} autoPlay playsInline className="hidden" />
      <canvas ref={stopCamera} className="absolute inset-0" />
      {isRunning && <div className="absolute inset-0 bg-black/50 opacity-50" />}
    </div>
  )
}