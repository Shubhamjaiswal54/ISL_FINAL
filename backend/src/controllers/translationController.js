import { TranslationHistory } from '../models/TranslationHistory.js'

export const saveTranslation = async (req, res) => {
  try {
    console.log('[Frontend -> Backend REST] Save translation:', { userId: req.body.userId, glossCount: req.body.raw_glosses?.length })
    const { userId, raw_glosses, final_sentence } = req.body
    const history = await TranslationHistory.create({ userId, raw_glosses, final_sentence })
    console.log('[Backend -> Database] Translation saved:', history._id.toString())
    res.status(201).json(history)
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
}

export const fetchHistory = async (req, res) => {
  try {
    const { userId } = req.params
    const history = await TranslationHistory.find({ userId }).sort({ timestamp: -1 })
    console.log('[Database -> Backend -> Frontend REST] History fetched:', { userId, count: history.length })
    res.json(history)
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
}