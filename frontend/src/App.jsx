import Navbar from './components/Navbar.jsx'
import PredictionDisplay from './components/PredictionDisplay.jsx'
import Controls from './components/Controls.jsx'
import LiveTranslate from './pages/LiveTranslate.jsx'

export default function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <main className="mx-auto grid max-w-6xl items-start gap-5 px-4 py-6 md:px-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <LiveTranslate />
        <div className="flex flex-col gap-5">
          <PredictionDisplay />
          <Controls />
        </div>
      </main>
    </div>
  )
}
