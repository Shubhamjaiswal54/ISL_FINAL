import { User } from '../models/User.js'
import { createToken } from '../utils/token.js'
import { hashPassword, comparePassword } from '../utils/bcrypt.js'

export const signup = async (req, res) => {
  try {
    const { email, password } = req.body
    const existingUser = await User.findOne({ email })
    if (existingUser) return res.status(400).json({ error: 'User already exists' })

    const hashedPassword = await hashPassword(password)
    const user = await User.create({ email, password: hashedPassword })
    const token = createToken(user._id)

    res.cookie('jwt', token, { httpOnly: true, secure: false, sameSite: 'lax' })
    res.status(201).json({ user: { id: user._id, email: user.email } })
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
}

export const login = async (req, res) => {
  try {
    const { email, password } = req.body
    const user = await User.findOne({ email })
    if (!user) return res.status(404).json({ error: 'User not found' })

    const isMatch = await comparePassword(password, user.password)
    if (!isMatch) return res.status(401).json({ error: 'Invalid credentials' })

    const token = createToken(user._id)
    res.cookie('jwt', token, { httpOnly: true, secure: false, sameSite: 'lax' })
    res.json({ user: { id: user._id, email: user.email } })
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
}

export const logout = (req, res) => {
  res.cookie('jwt', '', { maxAge: 0, httpOnly: true })
  res.json({ message: 'Logged out' })
}