/**
 * tailwind.config.js
 * VoxSign ISL Translation Frontend
 */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        background: '#0a0a0f',
        foreground: '#eaeaea',
        muted: {
          DEFAULT: '#6b6b80',
          foreground: '#6b6b80',
        },
        accent: '#00d4aa',
        card: '#16161f',
        secondary: '#2563eb',
      },
    },
  },
  plugins: [],
}