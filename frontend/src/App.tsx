import { StatusPanel } from "./components/StatusPanel"
import { ChatShell } from "./components/ChatShell"

export default function App() {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-bg-deep">
      {/* Ambient background */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-96 w-96 rounded-full bg-accent/5 blur-3xl" />
        <div className="absolute top-1/2 right-0 h-64 w-64 rounded-full bg-indigo/5 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-48 w-48 rounded-full bg-accent/3 blur-2xl" />
      </div>

      {/* Two-column layout */}
      <div className="relative flex w-full">
        {/* Left: Hero / Status panel */}
        <aside className="hidden md:flex w-80 xl:w-96 flex-shrink-0 flex-col border-r border-line bg-bg-panel px-6 py-8 overflow-y-auto">
          <StatusPanel />
        </aside>

        {/* Right: Chat */}
        <main className="flex flex-1 flex-col min-w-0 bg-bg-panel">
          <ChatShell />
        </main>
      </div>
    </div>
  )
}
