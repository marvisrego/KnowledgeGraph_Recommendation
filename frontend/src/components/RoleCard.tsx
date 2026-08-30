import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown, TrendingUp, Users, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Role } from "@/types"

interface Props {
  role: Role
  index?: number
}

const sourceColors: Record<string, string> = {
  onet: "bg-accent/10 text-accent border-accent/20",
  esco: "bg-indigo/10 text-indigo border-indigo/20",
}

const effortBorderColors: Record<string, string> = {
  low: "border-l-success",
  moderate: "border-l-warn",
  high: "border-l-error",
}

const effortTextColors: Record<string, string> = {
  low: "text-success",
  moderate: "text-warn",
  high: "text-error",
}

const SOURCE_LABELS: Record<string, string> = {
  direct_transition: "↗ direct transition",
  semantic_transition_backoff: "~ inferred transition",
  predicted_transition: "~ predicted",
  vector: "semantic match",
  skill_gap: "skill match",
}

export function RoleCard({ role, index = 0 }: Props) {
  const [showEvidence, setShowEvidence] = useState(false)

  const qualification = role.qualification_score
  const effortPct = role.effort_score !== undefined ? Math.round(role.effort_score * 100) : null
  const band = role.effort_band

  const transitionCount = role.transition?.count
  const transitionProb = role.transition?.probability
  const hasTransition = transitionCount !== undefined && transitionCount > 0

  const retrievalSources = role.retrieval_sources ?? []
  const primarySource = retrievalSources[0]

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className={cn(
        "group relative rounded-xl border border-l-4 bg-bg-raised p-4 flex flex-col gap-3",
        "transition-all duration-200 hover:shadow-[0_0_0_1px_rgba(76,194,234,0.15)] hover:-translate-y-0.5",
        band ? effortBorderColors[band] : "border-l-line",
        "border-line",
      )}
    >
      {/* Top row: source + effort score */}
      <div className="flex items-start justify-between gap-2">
        <span
          className={cn(
            "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide",
            sourceColors[role.source] ?? "bg-line/50 text-ink-muted border-line",
          )}
        >
          {role.source}
        </span>
        <div className="flex items-center gap-2 flex-shrink-0">
          {role.demand_trend && (
            <span className={cn("text-[10px] font-mono",
              role.demand_trend === "growing" ? "text-success" : role.demand_trend === "declining" ? "text-error" : "text-ink-muted"
            )}>
              {role.demand_trend === "growing" ? "↑ Growing" : role.demand_trend === "declining" ? "↓ Declining" : ""}
            </span>
          )}
          {effortPct !== null && band && (
            <div className="flex flex-col items-end">
              <span className={cn("text-lg font-bold leading-none font-mono", effortTextColors[band])}>
                {effortPct}%
              </span>
              <span className={cn("text-[9px] font-mono uppercase tracking-wide", effortTextColors[band])}>
                {band} effort
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Title */}
      <h3 className="text-sm font-semibold text-ink leading-snug line-clamp-2">{role.title}</h3>

      {/* Description */}
      {role.description && (
        <p className="text-[11px] text-ink-muted leading-relaxed line-clamp-2">{role.description}</p>
      )}

      {/* Qualification bar */}
      {qualification !== undefined && (
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] text-ink-muted font-mono">Skill match</span>
            <span className="text-[10px] text-accent font-mono font-semibold">
              {Math.round(qualification * 100)}%
              {role.have_count !== undefined && role.required_skill_count !== undefined && (
                <span className="text-ink-muted ml-1">({role.have_count}/{role.required_skill_count})</span>
              )}
            </span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-line overflow-hidden">
            <div
              className="h-full rounded-full bg-accent transition-all duration-700"
              style={{ width: `${qualification * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Time to upskill */}
      {role.estimated_weeks_min !== undefined && role.estimated_weeks_max !== undefined && (
        <span className="text-[11px] text-ink-muted font-mono">
          ⏱ ~{formatWeeks(role.estimated_weeks_min)}–{formatWeeks(role.estimated_weeks_max)} to upskill
        </span>
      )}

      {/* Skills */}
      {((role.have?.length ?? 0) > 0 || (role.need?.length ?? 0) > 0) && (
        <div className="flex flex-wrap gap-1">
          {role.have?.slice(0, 3).map((skill) => (
            <span key={skill} className="inline-flex rounded-full border border-success/25 bg-success/8 px-2 py-0.5 text-[10px] text-success">
              ✓ {truncate(skill, 18)}
            </span>
          ))}
          {role.need?.slice(0, 2).map((skill) => (
            <span key={skill} className="inline-flex rounded-full border border-warn/25 bg-warn/8 px-2 py-0.5 text-[10px] text-warn">
              + {truncate(skill, 18)}
            </span>
          ))}
        </div>
      )}

      {/* Why recommended — collapsible */}
      <button
        onClick={() => setShowEvidence((v) => !v)}
        className="flex items-center gap-1 text-[10px] text-ink-muted hover:text-ink-soft transition-colors"
      >
        <Zap className="w-3 h-3" />
        Why recommended?
        <ChevronDown className={cn("w-3 h-3 transition-transform", showEvidence && "rotate-180")} />
      </button>

      <AnimatePresence>
        {showEvidence && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-1.5 pt-1 border-t border-line">
              {/* Transition evidence */}
              {hasTransition && (
                <div className="flex items-center gap-1.5 text-[11px]">
                  <Users className="w-3 h-3 text-accent flex-shrink-0" />
                  <span className="text-ink-soft">
                    <span className="text-accent font-semibold">{transitionCount?.toLocaleString()}</span> real transitions observed
                    {transitionProb !== undefined && (
                      <span className="text-ink-muted ml-1">(p={transitionProb.toFixed(3)})</span>
                    )}
                  </span>
                </div>
              )}
              {/* Retrieval sources */}
              {primarySource && (
                <div className="flex items-center gap-1.5 text-[11px]">
                  <TrendingUp className="w-3 h-3 text-indigo flex-shrink-0" />
                  <span className="text-ink-soft">
                    Matched via <span className="text-indigo">{SOURCE_LABELS[primarySource] ?? primarySource}</span>
                  </span>
                </div>
              )}
              {/* Qualification note */}
              {qualification !== undefined && (
                <div className="flex items-center gap-1.5 text-[11px]">
                  <span className="w-3 h-3 text-center text-success">✓</span>
                  <span className="text-ink-soft">
                    You already have <span className="text-success font-semibold">{Math.round(qualification * 100)}%</span> of required skills
                  </span>
                </div>
              )}
              {/* Retrieval source chips */}
              {retrievalSources.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-0.5">
                  {retrievalSources.slice(0, 3).map((src) => (
                    <span key={src} className="rounded-full border border-line bg-bg-interactive px-2 py-0.5 text-[9px] text-ink-muted font-mono">
                      {src.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function formatWeeks(weeks: number): string {
  if (weeks < 4) return `${weeks}w`
  return `${Math.round(weeks / 4.3)}mo`
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s
}

declare module "@/types" {
  interface Role {
    demand_trend?: "growing" | "stable" | "declining"
  }
}
