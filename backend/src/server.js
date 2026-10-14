import express from 'express'
import cors from 'cors'
import 'dotenv/config'
import { Server } from 'socket.io'
import { connectDB } from './config/db.js'
import authRoutes from './routes/authRoutes.js'
import translationRoutes from './routes/translationRoutes.js'
import { translationSocket } from './sockets/translationSocket.js'

const app = express()
const port = Number(process.env.PORT || 5000)
const configuredOrigins = (process.env.CORS_ORIGIN || 'http://localhost:5173')
  .split(',')
  .map((origin) => origin.trim())
const isAllowedOrigin = (origin, callback) => {
  const allowed = !origin || configuredOrigins.includes(origin) || /^http:\/\/localhost:\d+$/.test(origin)
  callback(null, allowed)
}

app.use(cors({ origin: isAllowedOrigin, credentials: true }))
app.use(express.json())
app.use((req, res, next) => {
  console.log('[Backend REST request]', req.method, req.originalUrl)
  next()
})

connectDB().catch((error) => {
  console.error('Database startup error:', error.message)
})

app.use('/api/auth', authRoutes)
app.use('/api/translations', translationRoutes)

const server = app.listen(port, () =>
  console.log(`Backend running on port ${port}`)
)

const io = new Server(server, {
  cors: { origin: isAllowedOrigin, credentials: true }
})

translationSocket(io)

export { io }