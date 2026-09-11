import { useState } from "react"
import { ChevronDown, Zap, AlertTriangle } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import type { SkillGapEntry } from "@/types"

interface Props {
  entries: SkillGapEntry[]
}

export function SkillGapCard({ entries }: Props) {
  const [openIdx, setOpenIdx] = useState<number | null>(0)

  if (!entries?.length) return null

  return (
    <div className="skill-gap-panel rounded-xl border border-line bg-bg-raised overflow-hidden">
      <div className="px-4 py-3 border-b border-line flex items-center gap-2">
        <Zap className="w-4 h-4 text-warn" />
        <span className="text-sm font-medium text-ink">Skill Gap Analysis</span>
      </div>
      <div className="flex flex-col divide-y divide-line">
        {entries.slice(0, 3).map((entry, i) => (
          <div key={entry.role_id}>
            <button
              aria-expanded={openIdx === i}
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
              className="flex w-full items-center justify-between px-4 py-2.5 hover:bg-bg-interactive transition-colors"
            >
              <span className="text-sm text-ink-soft truncate">{entry.role_title}</span>
              <ChevronDown
                className={cn("w-4 h-4 text-ink-muted flex-shrink-0 transition-transform", openIdx === i && "rotate-180")}
              />
            </button>
            <AnimatePresence>
              {openIdx === i && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4 flex flex-col gap-3">
                    {/* Priority skills */}
                    {entry.priority_skills.length > 0 && (
                      <div>
                        <p className="text-xs font-mono text-ink-muted uppercase tracking-wide mb-1.5">
                          Learn first (highest impact)
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {entry.priority_skills.map((ps, j) => (
                            <span
                              key={j}
                              className="inline-flex items-center gap-1 rounded-full border border-accent/20 bg-accent/8 px-2.5 py-1 text-[13px] text-accent"
                              title={`TES reduction: ${(ps.tes_reduction * 100).toFixed(1)}%`}
                            >
                              {ps.skill}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-3">
                      {/* Quick wins */}
                      {entry.quick_wins.length > 0 && (
                        <div>
                          <p className="text-xs font-mono text-success uppercase tracking-wide mb-1">
                            ✓ Your strengths
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {entry.quick_wins.map((qw, j) => (
                              <span
                                key={j}
                                className="rounded-full border border-success/20 bg-success/8 px-2 py-0.5 text-xs text-success"
                              >
                                {qw.skill}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {/* Blockers */}
                      {entry.blockers.length > 0 && (
                        <div>
                          <p className="text-xs font-mono text-error uppercase tracking-wide mb-1">
                            <AlertTriangle className="w-3 h-3 inline mr-0.5" />
                            Key gaps
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {entry.blockers.map((bl, j) => (
                              <span
                                key={j}
                                className="rounded-full border border-error/20 bg-error/8 px-2 py-0.5 text-xs text-error"
                              >
                                {bl.skill}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  )
}
