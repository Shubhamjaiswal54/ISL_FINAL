import mongoose from 'mongoose'

const translationHistorySchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, index: true },
  raw_glosses: { type: [String], default: [] },
  final_sentence: { type: String, default: '' },
  timestamp: { type: Date, default: Date.now },
})

export const TranslationHistory = mongoose.model('TranslationHistory', translationHistorySchema)
