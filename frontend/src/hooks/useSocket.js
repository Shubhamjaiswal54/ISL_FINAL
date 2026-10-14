import { io } from 'socket.io-client'
import { useCallback, useEffect, useRef, useState } from 'react'

export const useSocket = () => {
  const socketRef = useRef(null)
  const sentCountRef = useRef(0)
  const [socket, setSocket] = useState(null)

  useEffect(() => {
    const socket = io(import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000', {
      withCredentials: true,
    })
    socketRef.current = socket
    setSocket(socket)

    socket.on('connect', () => console.log('[Socket] Connected:', {
      id: socket.id,
      connected: socket.connected,
      transport: socket.io.engine.transport.name,
    }))
    socket.on('disconnect', (reason) => console.log('[Frontend <- Backend] Socket disconnected:', reason))
    socket.on('connect_error', (error) => console.error('Socket connection error:', error.message))

    return () => {
      socket.disconnect()
      socketRef.current = null
      setSocket(null)
    }
  }, [])

  const emitLandmark = useCallback((landmarks) => {
    const currentSocket = socketRef.current
    sentCountRef.current += 1
    if (sentCountRef.current === 1 || sentCountRef.current % 30 === 0) {
      console.log('[Frontend -> Backend] Sending landmark-data:', {
        count: sentCountRef.current,
        length: landmarks?.length,
        sample: landmarks?.slice(0, 4),
        connected: currentSocket?.connected ?? false,
        socketId: currentSocket?.id ?? null,
      })
    }
    if (!currentSocket?.connected) {
      console.warn('[Frontend -> Backend] Landmark not sent: socket is not connected')
      return
    }
    if (!Array.isArray(landmarks) || landmarks.length !== 126) {
      console.error('[Frontend -> Backend] Landmark not sent: expected 126 values', landmarks?.length)
      return
    }
    currentSocket.emit('landmark-data', landmarks)
  }, [])

  return { emitLandmark, socket }
}