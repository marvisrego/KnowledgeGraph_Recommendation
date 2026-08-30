import { useState } from "react"
import { ChevronDown, GitBranch, ArrowRight, CheckCircle2, AlertCircle, XCircle } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import type { Explanation, Faithfulness } from "@/types"

const RELATION_META: Record<string, { label: string; color: string; icon: string }> = {
  TRANSITIONS_TO: { label: "Transitions To", color: "text-accent border-accent/20 bg-accent/8", icon: "↗" },
  SIMILAR_TO:     { label: "Similar To",     color: "text-indigo border-indigo/20 bg-indigo/8", icon: "≈" },
  REQUIRES:       { label: "Requires",       color: "text-warn border-warn/20 bg-warn/8",   icon: "★" },
  PREDICTED_TRANSITION: { label: "Predicted",  color: "text-success border-success/20 bg-success/8", icon: "~" },
  SAME_ISCO_GROUP: { label: "Same Domain", color: "text-ink-muted border-line bg-bg-raised", icon: "⬡" },
}

interface Props {
  explanations?: Explanation[]
  faithfulness?: Faithfulness | null
}

export function EvidencePanel({ explanations, faithfulness }: Props) {
  const [open, setOpen] = useState(false)

  if (!explanations?.length && !faithfulness) return null

  const pct = faithfulness?.score !== undefined ? Math.round(faithfulness.score * 100) : null
  const matched = faithfulness?.matched_entities ?? []
  const unreachable = faithfulness?.unreachable_entities ?? []
  const unmatched = faithfulness?.unmatched_entities ?? []
  const hasEntityDetails = matched.length + unreachable.length + unmatched.length > 0

  return (
    <div className="rounded-xl border border-line bg-bg-raised overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-bg-interactive transition-colors"
      >
        <div className="flex items-center gap-2 text-sm font-medium text-ink">
          <GitBranch className="w-4 h-4 text-accent" />
          Why these roles?
          {pct !== null && (
            <span
              className={cn(
                "ml-1 rounded-full border px-2 py-0.5 text-[10px] font-mono",
                pct >= 70 ? "border-success/30 text-success" : pct >= 40 ? "border-warn/30 text-warn" : "border-error/30 text-error",
              )}
            >
              {pct}% graph-verified
            </span>
          )}
        </div>
        <ChevronDown className={cn("w-4 h-4 text-ink-muted transition-transform", open && "rotate-180")} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 flex flex-col gap-4">

              {/* Evidence strength bar */}
              {pct !== null && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono text-ink-muted">Evidence strength</span>
                    <span className="text-[10px] font-mono text-accent">
                      {faithfulness?.matched_count ?? faithfulness?.reachable_count ?? "?"}/{faithfulness?.total_entities ?? "?"} entities in graph
                    </span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-line overflow-hidden">
                    <div
                      className={cn("h-full rounded-full transition-all duration-700",
                        pct >= 70 ? "bg-success" : pct >= 40 ? "bg-warn" : "bg-error"
                      )}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Entity verification breakdown */}
              {hasEntityDetails && (
                <div className="flex flex-col gap-2">
                  {matched.length > 0 && (
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <CheckCircle2 className="w-3 h-3 text-success" />
                        <span className="text-[10px] font-mono text-success uppercase tracking-wide">Verified in graph</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {matched.map((e) => (
                          <span key={e} className="rounded-full border border-success/20 bg-success/8 px-2 py-0.5 text-[10px] text-success">
                            {e}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {unreachable.length > 0 && (
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <AlertCircle className="w-3 h-3 text-warn" />
                        <span className="text-[10px] font-mono text-warn uppercase tracking-wide">In graph, not on path</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {unreachable.map((e) => (
                          <span key={e} className="rounded-full border border-warn/20 bg-warn/8 px-2 py-0.5 text-[10px] text-warn">
                            {e}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {unmatched.length > 0 && (
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <XCircle className="w-3 h-3 text-error" />
                        <span className="text-[10px] font-mono text-error uppercase tracking-wide">Not found in graph</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {unmatched.map((e) => (
                          <span key={e} className="rounded-full border border-error/20 bg-error/8 px-2 py-0.5 text-[10px] text-error">
                            {e}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Provenance chains */}
              {explanations && explanations.length > 0 && (
                <div className="flex flex-col gap-3">
                  <p className="text-[10px] font-mono text-ink-muted uppercase tracking-wide">Provenance paths</p>
                  {explanations.slice(0, 4).map((exp, i) => (
                    <div key={i} className="rounded-lg border border-line bg-bg-interactive p-3 flex flex-col gap-2">
                      <div className="flex items-center gap-1 text-[11px] text-ink-soft font-mono">
                        <span className="truncate max-w-[100px]">{exp.source_role}</span>
                        <ArrowRight className="w-3 h-3 text-ink-muted flex-shrink-0" />
                        <span className="truncate max-w-[100px] text-accent">{exp.target_role}</span>
                      </div>
                      <div className="flex flex-wrap items-center gap-1">
                        {exp.steps.map((step, j) => {
                          const meta = RELATION_META[step.relation]
                          return (
                            <div key={j} className="flex items-center gap-1">
                              <span
                                className={cn(
                                  "inline-flex items-center gap-0.5 rounded border px-1.5 py-0.5 text-[9px] font-mono uppercase whitespace-nowrap",
                                  meta?.color ?? "text-ink-muted border-line bg-bg-raised",
                                )}
                              >
                                <span>{meta?.icon ?? "→"}</span>
                                {meta?.label ?? step.relation.replace(/_/g, " ")}
                              </span>
                              {j < exp.steps.length - 1 && (
                                <span className="text-[9px] text-ink-muted">{step.to}</span>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
