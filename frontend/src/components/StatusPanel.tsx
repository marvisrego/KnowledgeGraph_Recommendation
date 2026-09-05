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
    <div className="studio-panel">
      {/* Header */}
      <div className="studio-brand">
        <div className="studio-mark" aria-hidden="true"><Network /></div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono uppercase tracking-widest text-ink-muted">
            Retrieve → Reason → Rank → Advise
          </span>
        </div>
        <h1 className="studio-title">Career Graph Studio</h1>
        <p className="text-sm text-ink-soft mt-1 leading-relaxed">
          GraphRAG-powered career advisor using O*NET, ESCO, and real career transition data.
        </p>
      </div>

      {/* Status cards */}
      {error ? (
        <div className="studio-status-error flex items-center gap-2 text-sm text-error" role="status">
          <XCircle className="w-4 h-4" /> Server unavailable
        </div>
      ) : cards.length > 0 ? (
        <div className="studio-status" aria-label="System status" aria-live="polite">
          {cards.map((c) => (
            <div
              key={c.label}
              className="studio-stat"
            >
              <div className="flex items-center gap-1.5">
                <c.icon
                  aria-hidden="true"
                  className={cn("w-3.5 h-3.5", c.label === "Status" ? (c.ok ? "text-success" : "text-warn") : "text-ink-muted")}
                />
                <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">
                  {c.label}
                </span>
              </div>
              <span className="studio-stat-value" title={c.value}>{c.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="studio-status" aria-label="Loading system status" aria-busy="true">
          {["Status", "Model", "Nodes", "Edges"].map((l) => (
            <div key={l} className="studio-stat min-h-20 animate-pulse bg-bg-raised" />
          ))}
        </div>
      )}

      {/* Graph link */}
      <a
        href="/graph"
        className="studio-graph-link"
      >
        <Network className="w-4 h-4 text-accent" />
        Explore knowledge graph →
      </a>

      {/* Tech badges */}
      <div className="studio-sources">
        {["GraphRAG", "O*NET", "ESCO", "Karrierewege", "LangGraph"].map((t) => (
          <span
            key={t}
            className="text-xs text-ink-muted"
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  )
}
