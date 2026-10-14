import { useTranslationStore } from '../context/TranslationContext.jsx'

export default function PredictionDisplay() {
  const { sentence, glosses } = useTranslationStore()

  return (
    <div className="bg-card p-6 rounded-xl border border-muted/30">
      <h3 className="text-semibold mb-4">Real-time Predictions</h3>
      <div className="space-y-3" id="predictions-list">
        {glosses.length === 0 ? <p className="text-muted-foreground">Waiting for predictions...</p> : glosses.map((gloss, index) => (
          <p key={`${gloss}-${index}`} className="font-medium">{gloss}</p>
        ))}
      </div>
      <div className="mt-4 pt-4 border-t border-muted/30">
        <p className="text-lg font-medium" id="sentence-display">Sentence: {sentence || '--'}</p>
      </div>
    </div>
  )
}