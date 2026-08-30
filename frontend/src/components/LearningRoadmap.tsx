import { useState } from "react"
import { ChevronDown, BookOpen, Clock } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import type { LearningPlanEntry, LearningPhase } from "@/types"

interface Props {
  plan: LearningPlanEntry[]
}

export function LearningRoadmap({ plan }: Props) {
  const [selectedRole, setSelectedRole] = useState(0)

  if (!plan?.length) return null

  const entry = plan[selectedRole]

  return (
    <div className="rounded-xl border border-line bg-bg-raised overflow-hidden">
      <div className="px-4 py-3 border-b border-line flex items-center gap-2">
        <BookOpen className="w-4 h-4 text-indigo" />
        <span className="text-sm font-medium text-ink">Learning Roadmap</span>
        <span className="ml-auto flex items-center gap-1 text-[11px] text-ink-muted font-mono">
          <Clock className="w-3 h-3" />
          ~{entry?.weeks_to_ready ?? 0} weeks total
        </span>
      </div>

      {/* Role selector */}
      {plan.length > 1 && (
        <div className="px-4 py-2 flex gap-2 overflow-x-auto border-b border-line">
          {plan.map((p, i) => (
            <button
              key={p.role_id}
              onClick={() => setSelectedRole(i)}
              className={cn(
                "flex-shrink-0 rounded-lg border px-3 py-1 text-xs transition-colors",
                i === selectedRole
                  ? "border-indigo/50 bg-indigo/10 text-indigo"
                  : "border-line text-ink-muted hover:text-ink",
              )}
            >
              {p.role_title.length > 20 ? p.role_title.slice(0, 20) + "…" : p.role_title}
            </button>
          ))}
        </div>
      )}

      {/* Timeline */}
      <div className="px-4 py-3 flex flex-col gap-0">
        {entry?.phases.map((phase, i) => (
          <PhaseRow key={i} phase={phase} isLast={i === entry.phases.length - 1} />
        ))}
      </div>
    </div>
  )
}

function PhaseRow({ phase, isLast }: { phase: LearningPhase; isLast: boolean }) {
  const [isOpen, setIsOpen] = useState(!isLast)

  return (
    <div className="flex gap-3">
      {/* Timeline line */}
      <div className="flex flex-col items-center pt-1">
        <div className="w-2.5 h-2.5 rounded-full border-2 border-indigo bg-bg-raised flex-shrink-0" />
        {!isLast && <div className="w-px flex-1 bg-line mt-1 mb-1" />}
      </div>

      {/* Content */}
      <div className={cn("flex-1 pb-3", isLast && "pb-0")}>
        <button
          onClick={() => setIsOpen((v) => !v)}
          className="flex w-full items-start justify-between gap-2 text-left"
        >
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-indigo">Phase {phase.phase}</span>
              <span className="text-[10px] text-ink-muted font-mono">Weeks {phase.weeks}</span>
            </div>
            <p className="text-sm font-medium text-ink mt-0.5">{phase.focus}</p>
            <p className="text-[11px] text-ink-muted mt-0.5">
              {phase.skills_unlocked} skills · {phase.courses.length} course{phase.courses.length !== 1 ? "s" : ""}
            </p>
          </div>
          <ChevronDown
            className={cn("w-3.5 h-3.5 text-ink-muted flex-shrink-0 mt-1 transition-transform", isOpen && "rotate-180")}
          />
        </button>

        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.18 }}
              className="overflow-hidden"
            >
              <div className="mt-2 flex flex-col gap-1.5">
                {/* Skills */}
                <div className="flex flex-wrap gap-1">
                  {phase.skills?.map((s, j) => (
                    <span
                      key={j}
                      className="rounded-full border border-indigo/20 bg-indigo/8 px-2 py-0.5 text-[10px] text-indigo"
                    >
                      {s}
                    </span>
                  ))}
                </div>
                {/* Courses */}
                {phase.courses.map((course, j) => (
                  <div
                    key={j}
                    className="rounded-lg border border-line bg-bg-interactive px-3 py-2 flex items-start gap-2"
                  >
                    <BookOpen className="w-3 h-3 text-indigo flex-shrink-0 mt-0.5" />
                    <div className="min-w-0">
                      <p className="text-[11px] font-medium text-ink line-clamp-1">{course.title}</p>
                      {course.provider && (
                        <p className="text-[10px] text-ink-muted font-mono">{course.provider}</p>
                      )}
                    </div>
                    {course.url && (
                      <a
                        href={course.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-auto flex-shrink-0 text-[10px] text-accent hover:underline"
                      >
                        Open →
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
