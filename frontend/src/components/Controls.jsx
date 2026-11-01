import { Delete, Space, Square, Video, Volume2 } from 'lucide-react'
import { useTranslationStore } from '../context/TranslationContext.jsx'

const buttonClass = 'inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40'

export default function Controls() {
  const { isListening, setIsListening, glosses, currentWord, sendControl } = useTranslationStore()

  return (
    <section className="rounded-3xl border border-white/10 bg-card p-5">
      <h2 className="text-lg font-semibold">Controls</h2>
      <p className="mt-1 text-sm text-muted">The same three actions can be signed. Hold each one still for a moment.</p>
      <div className="mt-4 grid gap-2 sm:grid-cols-3">
        <SignHint title="Space" detail="Both hands open, fingers spread. Ends the word." />
        <SignHint title="Backspace" detail="One hand, index finger pointing to the left edge of the picture." />
        <SignHint title="Accept" detail="Thumbs up. Keeps the word you spelled. Tap a suggestion to use that instead." />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <button
          type="button"
          onClick={() => setIsListening(true)}
          disabled={isListening}
          className={`${buttonClass} bg-accent text-background hover:bg-accent/90`}
        >
          <Video className="h-4 w-4" />
          Start
        </button>
        <button
          type="button"
          onClick={() => setIsListening(false)}
          disabled={!isListening}
          className={`${buttonClass} bg-red-600 text-white hover:bg-red-500`}
        >
          <Square className="h-4 w-4" />
          Stop
        </button>
        <button
          type="button"
          onClick={() => sendControl('space')}
          disabled={!currentWord}
          className={`${buttonClass} bg-secondary text-white hover:bg-secondary/90`}
        >
          <Space className="h-4 w-4" />
          Space
        </button>
        <button
          type="button"
          onClick={() => sendControl('backspace')}
          disabled={!glosses.length}
          className={`${buttonClass} border border-white/10 text-foreground hover:border-accent/50 hover:text-accent`}
        >
          <Delete className="h-4 w-4" />
          Backspace
        </button>
        <button
          type="button"
          onClick={() => sendControl('speak')}
          disabled={!glosses.length}
          className={`${buttonClass} bg-secondary text-white hover:bg-secondary/90 sm:col-span-2`}
        >
          <Volume2 className="h-4 w-4" />
          Speak
        </button>
      </div>
    </section>
  )
}

function SignHint({ title, detail }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/30 px-3 py-3">
      <p className="text-sm font-medium">{title}</p>
      <p className="mt-1 text-xs leading-relaxed text-muted">{detail}</p>
    </div>
  )
}
