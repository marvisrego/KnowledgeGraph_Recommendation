import { cn } from "@/lib/utils"
import type { Role } from "@/types"

interface Props {
  role: Role
}

const bandColors: Record<string, string> = {
  low: "bg-success/15 text-success border-success/30",
  moderate: "bg-warn/15 text-warn border-warn/30",
  high: "bg-error/15 text-error border-error/30",
}

export function EffortBadge({ role }: Props) {
  const { effort_band, effort_score, effort_source, estimated_weeks_min, estimated_weeks_max } = role
  if (!effort_band) return null

  const pct = effort_score !== undefined ? Math.round(effort_score * 100) : null
  const isRelative = effort_source === "relative"

  return (
    <div className="flex flex-col gap-1">
      <span
        className={cn(
          "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium",
          bandColors[effort_band] ?? "bg-ink-muted/15 text-ink-muted border-line",
        )}
        title={pct !== null ? `Effort score: ${pct}%` : undefined}
      >
        {effort_band === "low" && "↑ "}
        {effort_band === "high" && "↓ "}
        {effort_band.charAt(0).toUpperCase() + effort_band.slice(1)} effort
        {pct !== null && (
          <span className="opacity-70 font-mono">{pct}%</span>
        )}
        {isRelative && (
          <span className="opacity-50 text-[10px]">relative</span>
        )}
      </span>
      {estimated_weeks_min !== undefined && estimated_weeks_max !== undefined && (
        <span className="text-[11px] text-ink-muted font-mono">
          ⏱ ~{formatWeeks(estimated_weeks_min)}–{formatWeeks(estimated_weeks_max)} to upskill
        </span>
      )}
    </div>
  )
}

function formatWeeks(weeks: number): string {
  if (weeks < 4) return `${weeks}w`
  const months = Math.round(weeks / 4.3)
  return `${months}mo`
}
