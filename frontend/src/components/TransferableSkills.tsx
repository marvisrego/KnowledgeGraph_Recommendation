import { motion } from "framer-motion"
import { Sparkles } from "lucide-react"
import type { Role } from "@/types"

interface Props {
  roles: Role[]
}

export function TransferableSkills({ roles }: Props) {
  // Count how many roles each skill appears in
  const skillCounts = new Map<string, number>()
  for (const role of roles) {
    for (const skill of role.have ?? []) {
      skillCounts.set(skill, (skillCounts.get(skill) ?? 0) + 1)
    }
  }

  // Keep skills appearing in ≥ 2 roles, sort by count desc
  const shared = Array.from(skillCounts.entries())
    .filter(([, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)

  if (shared.length === 0) return null

  return (
    <div className="rounded-xl border border-line bg-bg-raised px-4 py-3">
      <div className="flex items-center gap-2 mb-2">
        <Sparkles className="w-4 h-4 text-success" />
        <span className="text-sm font-medium text-ink">Your transferable strengths</span>
        <span className="text-[10px] text-ink-muted font-mono ml-auto">
          {shared.length} skills apply across multiple roles
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {shared.map(([skill, count], i) => (
          <motion.span
            key={skill}
            initial={{ scale: 0.85, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: i * 0.03 }}
            className="inline-flex items-center gap-1 rounded-full border border-success/25 bg-success/8 px-3 py-1 text-[11px] text-success"
          >
            {skill}
            <span className="rounded-full bg-success/20 px-1.5 py-0.5 text-[9px] font-mono">
              ×{count}
            </span>
          </motion.span>
        ))}
      </div>
    </div>
  )
}
