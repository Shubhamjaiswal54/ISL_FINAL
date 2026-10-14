import WebSocket from 'ws'

export const translationSocket = (io) => {
  io.on('connection', (socket) => {
    console.log('Client connected:', socket.id)
    const mlSocket = new WebSocket(process.env.ML_SERVICE_WS_URL || 'ws://localhost:8000/ws/predict')

    mlSocket.on('open', () => console.log('ML socket connected for client:', socket.id))
    mlSocket.on('message', (message) => {
      try {
        const prediction = JSON.parse(message.toString())
        console.log('[ML -> Backend -> Frontend] Prediction:', {
          socketId: socket.id,
          predictedLabel: prediction.predictedLabel,
          accepted: prediction.accepted,
          word: prediction.word,
          confidence: prediction.confidence,
          sentence: prediction.sentence,
          glosses: prediction.glosses,
        })
        socket.emit('prediction', prediction)
      } catch (error) {
        console.error('Invalid ML response:', error.message)
      }
    })
    mlSocket.on('error', (error) => console.error('ML socket error:', error.message))

    socket.on('landmark-data', (landmarks) => {
      console.log('[Frontend -> Backend] landmark-data received:', {
        socketId: socket.id,
        length: landmarks?.length,
        sample: landmarks?.slice?.(0, 4),
        mlState: mlSocket.readyState,
      })
      if (mlSocket.readyState === WebSocket.OPEN) {
        const payload = JSON.stringify({ landmarks })
        mlSocket.send(payload)
        console.log('[Backend -> ML] Landmark vector sent:', { socketId: socket.id, bytes: payload.length })
      } else {
        console.warn('[Backend -> ML] Landmark dropped because ML socket is not open')
      }
    })

    socket.on('disconnect', () => {
      console.log('Client disconnected:', socket.id)
      mlSocket.close()
    })
  })
}