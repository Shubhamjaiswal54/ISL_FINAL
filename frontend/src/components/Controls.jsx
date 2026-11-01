import { useTranslationStore } from '../context/TranslationContext.jsx'
import { useSpeech } from '../hooks/useSpeech'

export default function Controls() {
  const { isListening, setIsListening, sentence } = useTranslationStore()
  const { speak } = useSpeech()

  return (
    <div className="bg-card p-6 rounded-xl border border-muted/30">
      <div className="flex flex-col sm:flex-row gap-4">
        <button
          onClick={() => setIsListening(!isListening)}
          className={`px-6 py-3 rounded-lg font-medium transition-colors ${isListening ? 'bg-red-600 text-white' : 'bg-accent text-background'}`}
        >
          {isListening ? 'Stop' : 'Start'}
        </button>

        <button
          onClick={() => speak(sentence)}
          className="px-6 py-3 rounded-lg font-medium transition-colors bg-secondary text-white disabled:opacity-50"
          disabled={!sentence}
        >
          Speak
        </button>
      </div>
    </div>
  )
}
