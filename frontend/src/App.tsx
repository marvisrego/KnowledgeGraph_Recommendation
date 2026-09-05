import { StatusPanel } from "./components/StatusPanel"
import { ChatShell } from "./components/ChatShell"

export default function App() {
  return (
    <div className="advisor-app">
      <a className="skip-link" href="#advisor-main">Skip to conversation</a>
      <div className="advisor-layout">
        {/* Left: Hero / Status panel */}
        <aside className="advisor-sidebar" aria-label="Career Graph Studio">
          <StatusPanel />
        </aside>

        {/* Right: Chat */}
        <main id="advisor-main" tabIndex={-1} className="advisor-main">
          <ChatShell />
        </main>
      </div>
    </div>
  )
}
