import { useTranslationStore } from '../context/TranslationContext.jsx'

function ConfidenceBar({ value, settled }) {
  const pct = Math.min(100, Math.max(0, Math.round(value * 100)))
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-label="Prediction confidence"
      className="h-1.5 w-full rounded-full bg-muted/30 overflow-hidden"
    >
      <div
        className={`h-full rounded-full transition-all duration-200 ${
          settled ? 'bg-green-500' : 'bg-amber-500'
        }`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

export default function PredictionDisplay() {
  const { sentence, glosses, prediction } = useTranslationStore()
  const hasPrediction = Boolean(prediction?.label)
  const confidence = prediction?.confidence ?? 0
  const pct = Math.round(confidence * 100)

  return (
    <section
      aria-labelledby="predictions-heading"
      className="bg-card p-6 rounded-xl border border-muted/30 space-y-5"
    >
      <div className="flex items-center justify-between">
        <h3 id="predictions-heading" className="font-semibold">
          Real-time Predictions
        </h3>
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={`h-2 w-2 rounded-full ${
              hasPrediction
                ? 'bg-green-500 motion-safe:animate-pulse'
                : 'bg-muted-foreground/40'
            }`}
          />
          {hasPrediction ? 'Hands detected' : 'Idle'}
        </span>
      </div>

      {/* Live prediction */}
      <div aria-live="polite" className="rounded-lg bg-muted/10 p-4 space-y-3">
        {hasPrediction ? (
          <>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-3xl font-mono font-bold tracking-wide">
                {prediction.label}
              </span>
              <span className="text-sm tabular-nums text-muted-foreground">
                {pct}%
              </span>
            </div>
            <ConfidenceBar value={confidence} settled={prediction.settled} />
            <p className="text-xs text-muted-foreground">
              {prediction.settled ? '✓ Locked in' : 'Hold still to confirm…'}
            </p>
          </>
        ) : (
          <div className="space-y-2">
            <p className="text-sm font-medium">No hands in view</p>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>
                Lower your hands for <strong>2s</strong> to end a word
              </li>
              <li>
                Lower your hands for <strong>4.5s</strong> to end the sentence
              </li>
              <li>
                Press <strong>Stop</strong> to end it immediately
              </li>
            </ul>
          </div>
        )}
      </div>

      {/* Detected words */}
      <div>
        <h4 className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
          Detected words
        </h4>
        <div className="flex flex-wrap gap-2 min-h-[2rem]" id="predictions-list">
          {glosses.length === 0 ? (
            <p className="text-sm text-muted-foreground italic">
              Waiting for letters…
            </p>
          ) : (
            glosses.map((word, index) => {
              const isLatest = index === glosses.length - 1
              return (
                <span
                  key={`${word}-${index}`}
                  className={`font-mono font-medium px-2.5 py-1 rounded-md border text-sm transition-colors ${
                    isLatest
                      ? 'bg-primary/10 border-primary/40 text-primary'
                      : 'bg-muted/20 border-muted/30'
                  }`}
                >
                  {word}
                </span>
              )
            })
          )}
        </div>
      </div>

      {/* Sentence */}
      <div className="pt-4 border-t border-muted/30">
        <h4 className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
          Sentence
        </h4>
        <p
          id="sentence-display"
          aria-live="polite"
          className={`text-lg font-medium ${
            sentence ? '' : 'text-muted-foreground'
          }`}
        >
          {sentence || '—'}
        </p>
      </div>
    </section>
  )
}