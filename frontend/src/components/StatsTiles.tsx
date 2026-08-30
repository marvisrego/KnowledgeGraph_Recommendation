import { motion } from "framer-motion"
import { TrendingUp, Clock, Users, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Role } from "@/types"

interface Props {
  role: Role
}

interface Tile {
  icon: React.ElementType
  label: string
  value: string
  sub?: string
  color: string
}

export function StatsTiles({ role }: Props) {
  const tiles: Tile[] = []

  if (role.qualification_score !== undefined) {
    tiles.push({
      icon: TrendingUp,
      label: "Skill match",
      value: `${Math.round(role.qualification_score * 100)}%`,
      sub: role.have_count !== undefined && role.required_skill_count !== undefined
        ? `${role.have_count}/${role.required_skill_count} skills`
        : "qualified",
      color: "text-accent",
    })
  }

  if (role.estimated_weeks_min !== undefined && role.estimated_weeks_max !== undefined) {
    const minMo = Math.round(role.estimated_weeks_min / 4.3)
    const maxMo = Math.round(role.estimated_weeks_max / 4.3)
    tiles.push({
      icon: Clock,
      label: "Time to ready",
      value: minMo === maxMo ? `~${minMo} mo` : `${minMo}–${maxMo} mo`,
      sub: "estimated",
      color: "text-indigo",
    })
  }

  const count = role.transition?.count
  if (count !== undefined && count > 0) {
    tiles.push({
      icon: Users,
      label: "Real transitions",
      value: count >= 1000 ? `${(count / 1000).toFixed(1)}k` : count.toString(),
      sub: "people made this move",
      color: "text-success",
    })
  }

  if (role.effort_score !== undefined && role.effort_band) {
    const bandColors: Record<string, string> = {
      low: "text-success",
      moderate: "text-warn",
      high: "text-error",
    }
    tiles.push({
      icon: Zap,
      label: "Effort",
      value: `${Math.round(role.effort_score * 100)}%`,
      sub: `${role.effort_band} difficulty`,
      color: bandColors[role.effort_band] ?? "text-ink-muted",
    })
  }

  if (tiles.length === 0) return null

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {tiles.map((tile, i) => (
        <motion.div
          key={tile.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className="rounded-xl border border-line bg-bg-raised px-3 py-3 flex flex-col gap-1"
        >
          <div className="flex items-center gap-1.5">
            <tile.icon className={cn("w-3 h-3", tile.color)} />
            <span className="text-[10px] font-mono text-ink-muted uppercase tracking-wide truncate">
              {tile.label}
            </span>
          </div>
          <span className={cn("text-xl font-bold font-mono leading-none", tile.color)}>
            {tile.value}
          </span>
          {tile.sub && (
            <span className="text-[10px] text-ink-muted truncate">{tile.sub}</span>
          )}
        </motion.div>
      ))}
    </div>
  )
}
