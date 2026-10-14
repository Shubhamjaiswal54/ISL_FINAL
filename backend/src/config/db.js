import mongoose from 'mongoose'

export const db = mongoose.connection

db.on('error', (error) => console.error('MongoDB connection error:', error))
db.once('open', () => console.log('Connected to MongoDB'))

export const connectDB = async () => {
  await mongoose.connect(process.env.MONGO_URI)
}