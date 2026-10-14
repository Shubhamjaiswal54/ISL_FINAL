import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="bg-card border border-muted/30 backdrop-blur-sm px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link to="/" className="text-2xl font-bold text-accent">VoxSign</Link>
        <ul className="flex gap-6 text-muted">
          <li><NavLink to="/live" className={({ isActive }) => isActive ? 'text-accent' : 'hover:text-accent'}>Live</NavLink></li>
          <li><NavLink to="/dashboard" className={({ isActive }) => isActive ? 'text-accent' : 'hover:text-accent'}>Dashboard</NavLink></li>
        </ul>
      </div>
    </nav>
  )
}