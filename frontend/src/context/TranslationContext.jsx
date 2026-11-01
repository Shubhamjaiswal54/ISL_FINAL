import { createContext, useContext, useState } from 'react'

const TranslationContext = createContext()

export const useTranslationStore = () => useContext(TranslationContext)

export const TranslationProvider = ({ children }) => {
  const [sentence, setSentence] = useState('')
  const [glosses, setGlosses] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [isListening, setIsListening] = useState(false)

  return (
    <TranslationContext.Provider value={{
      sentence, setSentence,
      glosses, setGlosses,
      prediction, setPrediction,
      isListening, setIsListening,
    }}>
      {children}
    </TranslationContext.Provider>
  )
}
