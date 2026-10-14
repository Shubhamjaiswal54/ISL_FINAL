import { Router } from 'express'
import { saveTranslation, fetchHistory } from '../controllers/translationController.js'
import { protect } from '../middleware/authMiddleware.js'

const translationRoutes = Router()

translationRoutes.post('/', protect, saveTranslation)
translationRoutes.get('/:userId', protect, fetchHistory)

export default translationRoutes