import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000',
  timeout: 10000,
  withCredentials: true
})

api.interceptors.request.use((config) => {
  console.log('[Frontend -> Backend REST]', config.method?.toUpperCase(), config.url, config.data)
  return config
})

api.interceptors.response.use((response) => {
  console.log('[Backend -> Frontend REST]', response.status, response.config.url, response.data)
  return response
}, (error) => {
  console.error('[Backend REST error]', error.response?.status, error.config?.url, error.response?.data || error.message)
  return Promise.reject(error)
})

export default api