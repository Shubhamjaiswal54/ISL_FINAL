import WebSocket from 'ws'

const ML_SERVICE_WS_URL = process.env.ML_SERVICE_WS_URL || 'ws://localhost:8000/ws/predict'
const MAX_RECONNECT_DELAY_MS = 5000

// One ML WebSocket per browser client, so each client gets its own smoothing
// and sentence state on the ML side. Frames are {hands: [...]} - an empty
// list means no hand is visible, which is what ends words and sentences.
export const translationSocket = (io) => {
  io.on('connection', (socket) => {
    console.log('Client connected:', socket.id)
    let mlSocket = null
    let reconnectDelay = 500
    let reconnectTimer = null
    let closedByClient = false
    let droppedFrames = 0

    const connectToML = () => {
      mlSocket = new WebSocket(ML_SERVICE_WS_URL)

      mlSocket.on('open', () => {
        console.log('ML socket connected for client:', socket.id)
        reconnectDelay = 500
        socket.emit('ml-status', { connected: true })
      })

      mlSocket.on('message', (message) => {
        try {
          const data = JSON.parse(message.toString())
          if (data.error) {
            console.error('ML service error:', data.error, data.detail || '')
            socket.emit('ml-status', { connected: false, error: data.error, detail: data.detail })
          } else if (data.type === 'sentence') {
            console.log('[ML -> Frontend] Sentence:', { socketId: socket.id, sentence: data.sentence })
            socket.emit('prediction', data)
          } else {
            socket.emit('prediction', data)
          }
        } catch (error) {
          console.error('Invalid ML response:', error.message)
        }
      })

      mlSocket.on('error', (error) => console.error('ML socket error:', error.message))

      mlSocket.on('close', (code, reason) => {
        if (closedByClient) return
        const why = reason?.toString?.() || ''
        // 1011: the service is up but refused the session (no model loaded).
        // Reconnecting would just repeat that refusal.
        if (code === 1011) {
          console.warn('ML service refused the session:', why)
          socket.emit('ml-status', { connected: false, error: why || 'Model not loaded' })
          return
        }
        console.warn(`ML socket closed (${code} ${why}), retrying in ${reconnectDelay}ms`)
        socket.emit('ml-status', { connected: false })
        reconnectTimer = setTimeout(connectToML, reconnectDelay)
        reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY_MS)
      })
    }

    connectToML()

    const sendToML = (payload) => {
      if (mlSocket?.readyState === WebSocket.OPEN) {
        mlSocket.send(JSON.stringify(payload))
        return true
      }
      return false
    }

    socket.on('landmark-data', (frame) => {
      if (!frame || !Array.isArray(frame.hands)) return
      if (!sendToML({ hands: frame.hands, t: frame.t })) {
        if (++droppedFrames % 100 === 1) {
          console.warn(`[Backend -> ML] Dropping frames: ML socket not open (${droppedFrames} so far)`)
        }
      }
    })

    socket.on('translation-control', (message) => {
      const action = message?.action
      if (action === 'finish' || action === 'reset' || action === 'backspace' || action === 'space' || action === 'speak') {
        const payload = { type: action }
        if (Number.isInteger(message.index)) payload.index = message.index
        if (typeof message.word === 'string') payload.word = message.word
        sendToML(payload)
      }
    })

    socket.on('disconnect', () => {
      console.log('Client disconnected:', socket.id)
      closedByClient = true
      clearTimeout(reconnectTimer)
      mlSocket?.close()
    })
  })
}
