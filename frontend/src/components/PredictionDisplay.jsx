import { useEffect } from 'react'
import { X } from 'lucide-react'
import { useTranslationStore } from '../context/TranslationContext.jsx'

const FALLBACK_THRESHOLD = 0.5

export default function PredictionDisplay() {
  const {
    sentence, sentences, glosses, setGlosses, currentWord, suggestions, prediction, sendControl,
  } = useTranslationStore()
  const threshold = prediction?.threshold ?? FALLBACK_THRESHOLD
  const percent = Math.round((prediction?.confidence || 0) * 100)
  const thresholdPercent = Math.round(threshold * 100)
  const above = Boolean(prediction?.label) && (prediction.confidence || 0) >= threshold
  const committed = currentWord && glosses[glosses.length - 1] === currentWord
    ? glosses.slice(0, -1)
    : glosses
  const working = committed.join(' ')
  const shownSentence = sentence || (working ? working.charAt(0).toUpperCase() + working.slice(1) : '')

  const removeGloss = (index) => {
    if (index < 0 || index >= glosses.length) return
    setGlosses((words) => words.filter((_, wordIndex) => wordIndex !== index))
    sendControl('backspace', { index })
  }

  useEffect(() => {
    const onKey = (event) => {
      const tag = event.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || event.target?.isContentEditable) return
      if (event.key === 'Backspace') {
        if (!glosses.length) return
        event.preventDefault()
        sendControl('backspace')
      } else if (event.key === ' ' && currentWord) {
        event.preventDefault()
        sendControl('space')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [glosses.length, currentWord, sendControl])

  return (
    <section className="rounded-3xl border border-white/10 bg-card p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Prediction</h2>
          <p className="text-sm text-muted">A letter joins the word after it holds at {thresholdPercent}% or more. Space ends the word.</p>
        </div>
        <p className={`font-mono text-sm ${above ? 'text-accent' : 'text-muted'}`}>{percent}%</p>
      </div>

      <div className="mt-5 flex items-end justify-between gap-4">
        <p className="font-mono text-6xl leading-none tracking-tight">
          {above ? prediction.label : '–'}
        </p>
        <p className="max-w-[14rem] text-right text-sm text-muted">
          {prediction?.label
            ? (above
              ? (prediction.settled ? 'Held. It will be added.' : 'Hold still so it can be added.')
              : `${percent}% is under ${thresholdPercent}%, so this letter stays out of the word.`)
            : 'Sign a letter, then press Space to finish the word.'}
        </p>
      </div>

      <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full transition-[width] duration-150 ${above ? 'bg-accent' : 'bg-muted'}`}
          style={{ width: `${Math.max(0, Math.min(percent, 100))}%` }}
        />
        <div
          className="absolute top-0 h-full w-px bg-foreground/80"
          style={{ left: `${thresholdPercent}%` }}
        />
      </div>

      <h3 className="mt-6 text-sm font-medium text-muted">Words</h3>
      <div className="mt-3 flex min-h-12 flex-wrap gap-2" id="predictions-list">
        {glosses.length === 0
          ? <p className="text-sm text-muted">Waiting for a letter above {thresholdPercent}%.</p>
          : glosses.map((word, index) => (
            <button
              key={`${word}-${index}`}
              type="button"
              onClick={() => removeGloss(index)}
              className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3 py-1.5 font-mono text-sm text-foreground transition-colors hover:border-red-400/60 hover:bg-red-500/10"
              aria-label={`Remove ${word}`}
            >
              {word}
              <X className="h-3.5 w-3.5 opacity-70" />
            </button>
          ))}
      </div>

      {suggestions.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-medium text-muted">Word predictions</h3>
          <div className="mt-2 flex flex-wrap gap-2">
            {suggestions.map((word) => (
              <button
                key={word}
                type="button"
                onClick={() => sendControl('space', { word })}
                className="rounded-full border border-secondary/40 bg-secondary/15 px-3 py-1.5 text-sm text-foreground transition-colors hover:bg-secondary/30"
              >
                {word}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 border-t border-white/10 pt-5">
        <p className="text-xs uppercase tracking-wide text-muted">Sentence</p>
        <p className="mt-2 text-2xl font-medium leading-snug" id="sentence-display">{shownSentence || '—'}</p>
        {sentences.length > 0 && (
          <ul className="mt-4 space-y-1">
            {sentences.slice(-5).map((line, index) => (
              <li key={`${line}-${index}`} className="text-sm text-muted">{line}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
