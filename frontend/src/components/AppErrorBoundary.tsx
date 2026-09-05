import { Component, type ErrorInfo, type ReactNode } from "react"
import { AlertTriangle, RotateCcw } from "lucide-react"

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
}

export class AppErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Career Advisor rendering failed", error, info)
  }

  render() {
    if (!this.state.hasError) return this.props.children

    return (
      <main className="flex min-h-screen items-center justify-center bg-bg-deep px-6 text-ink">
        <section className="w-full max-w-md rounded-2xl border border-error/30 bg-bg-panel p-6 text-center shadow-2xl">
          <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-error" />
          <h1 className="text-lg font-semibold">The recommendation could not be displayed</h1>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            Your databases are still connected. Reload the advisor to start a fresh session.
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mx-auto mt-5 inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-4 py-2 text-sm text-accent transition-colors hover:bg-accent/20"
          >
            <RotateCcw className="h-4 w-4" />
            Reload advisor
          </button>
        </section>
      </main>
    )
  }
}
