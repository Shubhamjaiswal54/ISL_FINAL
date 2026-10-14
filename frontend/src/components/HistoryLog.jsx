export default function HistoryLog() {
  return (
    <div className="bg-card p-6 rounded-xl border border-muted/30">
      <h3 className="text-semibold mb-4">Translation History</h3>
      <div className="h-96 overflow-y-auto space-y-2">
        {/* History items rendered by context */}
      </div>
    </div>
  )
}