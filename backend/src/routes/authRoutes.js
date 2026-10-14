import { Router } from 'express'
import { signup, login, logout } from '../controllers/authController.js'

const authRoutes = Router()

authRoutes.post('/signup', signup)
authRoutes.post('/login', login)
authRoutes.post('/logout', logout)

export default authRoutes