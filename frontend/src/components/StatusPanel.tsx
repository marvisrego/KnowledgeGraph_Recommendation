import { useEffect, useState } from "react"
import { Network, Database, Cpu, CheckCircle, XCircle } from "lucide-react"
import { getStatus } from "@/api"
import type { StatusResponse } from "@/types"
import { cn } from "@/lib/utils"

export function StatusPanel() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setError(true))
  }, [])

  const cards = status
    ? [
        {
          icon: CheckCircle,
          label: "Status",
          value: status.ready ? "Ready" : "Loading…",
          ok: status.ready,
        },
        { icon: Cpu, label: "Model", value: status.chat_model ?? "—", ok: true },
        { icon: Database, label: "Nodes", value: status.graph_nodes?.toLocaleString() ?? "—", ok: true },
        { icon: Network, label: "Edges", value: status.graph_edges?.toLocaleString() ?? "—", ok: true },
      ]
    : []

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
            Retrieve → Reason → Rank → Advise
          </span>
        </div>
        <h1 className="text-2xl font-bold text-ink tracking-tight">Career Graph Studio</h1>
        <p className="text-sm text-ink-soft mt-1 leading-relaxed">
          GraphRAG-powered career advisor using O*NET, ESCO, and real career transition data.
        </p>
      </div>

      {/* Status cards */}
      {error ? (
        <div className="flex items-center gap-2 text-sm text-error">
          <XCircle className="w-4 h-4" /> Server unavailable
        </div>
      ) : cards.length > 0 ? (
        <div className="grid grid-cols-2 gap-2">
          {cards.map((c) => (
            <div
              key={c.label}
              className="rounded-xl border border-line bg-bg-raised px-3 py-2.5 flex flex-col gap-1"
            >
              <div className="flex items-center gap-1.5">
                <c.icon
                  className={cn("w-3 h-3", c.ok ? "text-success" : "text-warn")}
                />
                <span className="text-[10px] font-mono text-ink-muted uppercase tracking-wide">
                  {c.label}
                </span>
              </div>
              <span className="text-sm font-semibold text-ink font-mono truncate">{c.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          {["Status", "Model", "Nodes", "Edges"].map((l) => (
            <div key={l} className="h-16 rounded-xl border border-line bg-bg-raised animate-pulse" />
          ))}
        </div>
      )}

      {/* Graph link */}
      <a
        href="/graph"
        className="flex items-center gap-2 rounded-xl border border-line bg-bg-raised px-4 py-3 text-sm text-ink-soft hover:text-accent hover:border-accent/40 transition-colors"
      >
        <Network className="w-4 h-4 text-accent" />
        Explore knowledge graph →
      </a>

      {/* Tech badges */}
      <div className="flex flex-wrap gap-2">
        {["GraphRAG", "O*NET", "ESCO", "Karrierewege", "LangGraph"].map((t) => (
          <span
            key={t}
            className="rounded-full border border-line px-3 py-1 text-[10px] font-mono text-ink-muted uppercase tracking-wide"
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  )
}
