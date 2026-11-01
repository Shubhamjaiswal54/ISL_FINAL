import { io } from 'socket.io-client'
import { useCallback, useEffect, useRef, useState } from 'react'

export const useSocket = () => {
  const socketRef = useRef(null)
  const [socket, setSocket] = useState(null)
  const [mlStatus, setMlStatus] = useState({ connected: false })

  useEffect(() => {
    const nextSocket = io(import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000', {
      withCredentials: true,
    })
    socketRef.current = nextSocket
    setSocket(nextSocket)

    nextSocket.on('connect', () => console.log('[Socket] Connected:', nextSocket.id))
    nextSocket.on('disconnect', (reason) => {
      console.log('[Socket] Disconnected:', reason)
      setMlStatus({ connected: false })
    })
    nextSocket.on('connect_error', (error) => console.error('Socket connection error:', error.message))
    nextSocket.on('ml-status', setMlStatus)

    return () => {
      nextSocket.disconnect()
      socketRef.current = null
      setSocket(null)
    }
  }, [])

  // hands: 0-2 hands of 21 [x, y, z] points. An empty list tells the ML
  // service no hand is visible, which is what ends words and sentences.
  // t is the capture time; the server judges holds and gaps on it, so a
  // frame processed late still counts for the time it was actually signed.
  const emitFrame = useCallback((hands) => {
    const currentSocket = socketRef.current
    if (currentSocket?.connected) currentSocket.emit('landmark-data', { hands, t: performance.now() })
  }, [])

  const emitControl = useCallback((action, extra) => {
    const currentSocket = socketRef.current
    if (currentSocket?.connected) currentSocket.emit('translation-control', { action, ...extra })
  }, [])

  return { socket, emitFrame, emitControl, mlStatus }
}
