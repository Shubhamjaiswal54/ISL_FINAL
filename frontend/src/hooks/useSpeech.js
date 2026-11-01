import { useCallback } from 'react'

export const useSpeech = () => {
  const speak = useCallback((text) => {
    if (!text || !('speechSynthesis' in window)) return
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'en-IN'
    window.speechSynthesis.speak(utterance)
  }, [])

  return { speak }
}
