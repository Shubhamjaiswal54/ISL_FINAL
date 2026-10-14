export default function Dashboard() {
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>
      <p className="text-muted mb-4">Translation logs and statistics will appear here.</p>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card p-4 rounded-xl">
          <p className="text-muted">Total Translations</p>
          <p className="text-3xl font-bold">0</p>
        </div>
        <div className="bg-card p-4 rounded-xl">
          <p className="text-muted">Average Confidence</p>
          <p className="text-3xl font-bold">--</p>
        </div>
      </div>
    </div>
  )
}