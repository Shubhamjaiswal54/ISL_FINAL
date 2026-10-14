import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import { TranslationProvider } from './context/TranslationContext.jsx'
import './index.css'

const root = createRoot(document.getElementById('root'))
root.render(
	<BrowserRouter>
		<TranslationProvider>
			<App />
		</TranslationProvider>
	</BrowserRouter>
)