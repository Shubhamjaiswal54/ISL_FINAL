import { useTranslationStore } from '../context/TranslationContext.jsx'

export default function PredictionDisplay() {
  const { sentence, glosses, prediction } = useTranslationStore()

  return (
    <div className="bg-card p-6 rounded-xl border border-muted/30">
      <h3 className="font-semibold mb-4">Real-time Predictions</h3>
      <p className="text-sm text-gray-500 mb-3">
        {prediction?.label
          ? `Seeing ${prediction.label} (${Math.round(prediction.confidence * 100)}%)${prediction.settled ? '' : ' · hold still'}`
          : 'No hands in view. Lower your hands for 2 seconds to end a word, or 4.5 seconds to end the sentence. Stop ends it immediately.'}
      </p>
      <div className="flex flex-wrap gap-2" id="predictions-list">
        {glosses.length === 0
          ? <p className="text-muted-foreground">Waiting for letters...</p>
          : glosses.map((word, index) => (
            <span key={`${word}-${index}`} className="font-mono font-medium px-2 py-1 rounded bg-gray-100">{word}</span>
          ))}
      </div>
      <div className="mt-4 pt-4 border-t border-muted/30">
        <p className="text-lg font-medium" id="sentence-display">Sentence: {sentence || '--'}</p>
      </div>
    </div>
  )
}
