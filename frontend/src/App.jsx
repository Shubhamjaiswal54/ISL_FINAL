import Navbar from './components/Navbar.jsx'
import CameraFeed from './components/CameraFeed.jsx'
import PredictionDisplay from './components/PredictionDisplay.jsx'
import HistoryLog from './components/HistoryLog.jsx'
import Controls from './components/Controls.jsx'
import LiveTranslate from './pages/LiveTranslate.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Login from './pages/Login.jsx'
import { useTranslationStore } from './context/TranslationContext.jsx'

export default function App() {
  const { sentence, glosses, isListening } = useTranslationStore()

  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <Navbar />
      <div className="container mx-auto py-8">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* Live Translation Panel */}
          <div>
            <h2 className="text-xl font-bold mb-4">Live Translation</h2>
            <LiveTranslate />
          </div>

          {/* Controls Panel */}
          <div>
            <h2 className="text-xl font-bold mb-4">Controls</h2>
            <Controls />
          </div>
        </div>

        {/* Prediction Display & History */}
        <div className="grid gap-6 mt-8">
          <PredictionDisplay />
          <HistoryLog />
        </div>
      </div>
    </div>
  )
}