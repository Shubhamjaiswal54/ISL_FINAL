import { createContext, useCallback, useContext, useRef, useState } from 'react'

const TranslationContext = createContext()

export const useTranslationStore = () => useContext(TranslationContext)

export const TranslationProvider = ({ children }) => {
  const [sentence, setSentence] = useState('')
  const [sentences, setSentences] = useState([])
  const [glosses, setGlosses] = useState([])
  const [currentWord, setCurrentWord] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [isListening, setIsListening] = useState(false)
  const controlRef = useRef(null)

  const registerControl = useCallback((send) => {
    controlRef.current = send
  }, [])

  const sendControl = useCallback((action, extra) => {
    controlRef.current?.(action, extra)
  }, [])

  return (
    <TranslationContext.Provider value={{
      sentence, setSentence,
      sentences, setSentences,
      glosses, setGlosses,
      currentWord, setCurrentWord,
      suggestions, setSuggestions,
      prediction, setPrediction,
      isListening, setIsListening,
      registerControl, sendControl,
    }}>
      {children}
    </TranslationContext.Provider>
  )
}
