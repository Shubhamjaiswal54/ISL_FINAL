export default function Login() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="bg-card p-8 rounded-xl w-full max-w-md border border-muted/30">
        <h2 className="text-2xl font-bold mb-6 text-center">Sign in to VoxSign</h2>
        <form className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Email</label>
            <input type="email" required className="w-full bg-subtext border border-border rounded-lg px-4 py-3" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Password</label>
            <input type="password" required className="w-full bg-subtext border border-rounded-lg px-4 py-3" />
          </div>
          <button type="submit" className="w-full bg-accent text-background py-3 rounded-lg font-medium hover:opacity-90 transition-opacity">
            Sign in
          </button>
        </form>
        <p className="text-center text-muted mt-4">Don't have an account? <a href="#" className="underline hover:text-accent">Sign up</a></p>
      </div>
    </div>
  )
}