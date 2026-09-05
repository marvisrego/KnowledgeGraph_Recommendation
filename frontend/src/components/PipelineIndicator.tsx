import { useEffect, useState } from "react"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import { cn } from "@/lib/utils"

const STAGES = [
  { key: "intent", label: "Intent" },
  { key: "retrieve", label: "Retrieve" },
  { key: "rank", label: "Rank" },
  { key: "effort", label: "Effort" },
  { key: "generate", label: "Generate" },
]

const STAGE_DURATION_MS = 900

interface Props {
  visible: boolean
}

export function PipelineIndicator({ visible }: Props) {
  const reduceMotion = useReducedMotion()
  const [activeIdx, setActiveIdx] = useState(0)

  useEffect(() => {
    if (!visible) {
      setActiveIdx(0)
      return
    }
    const timers = STAGES.map((_, i) =>
      setTimeout(() => setActiveIdx(i), i * STAGE_DURATION_MS),
    )
    return () => timers.forEach(clearTimeout)
  }, [visible])

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          role="status"
          className="pipeline-indicator flex flex-col gap-4 rounded-xl border border-line bg-bg-raised px-4 py-5 w-fit max-w-full"
        >
          <div className="pipeline-stages">
            {STAGES.map((stage, i) => (
              <div key={stage.key} className="flex items-center gap-2">
                <div className="flex flex-col items-center gap-1">
                  <motion.div
                    animate={{
                      scale: i === activeIdx && !reduceMotion ? [1, 1.25, 1] : 1,
                      opacity: i <= activeIdx ? 1 : 0.25,
                    }}
                    transition={{ repeat: i === activeIdx && !reduceMotion ? Infinity : 0, duration: reduceMotion ? 0 : 0.8 }}
                    className={cn(
                      "w-2 h-2 rounded-full transition-colors duration-300",
                      i < activeIdx
                        ? "bg-success"
                        : i === activeIdx
                          ? "bg-accent"
                          : "bg-line",
                    )}
                  />
                  <span
                    className={cn(
                      "text-[13px] font-mono uppercase tracking-wider transition-colors duration-300",
                      i === activeIdx ? "text-accent" : i < activeIdx ? "text-success" : "text-ink-muted",
                    )}
                  >
                    {stage.label}
                  </span>
                </div>
                {i < STAGES.length - 1 && (
                  <div
                    className={cn(
                      "h-px w-6 -mt-3 transition-colors duration-300",
                      i < activeIdx ? "bg-success/50" : "bg-line",
                    )}
                  />
                )}
              </div>
            ))}
          </div>
          <p className="text-[13px] text-ink-muted font-mono">
            {STAGES[activeIdx]?.label === "Intent"
              ? "Understanding your goals…"
              : STAGES[activeIdx]?.label === "Retrieve"
                ? "Searching knowledge graph…"
                : STAGES[activeIdx]?.label === "Rank"
                  ? "Ranking by skill fit…"
                  : STAGES[activeIdx]?.label === "Effort"
                    ? "Computing transition effort…"
                    : "Generating your advice…"}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
