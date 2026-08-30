import { ShieldCheck, ShieldAlert, Shield } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Faithfulness } from "@/types"

interface Props {
  faithfulness?: Faithfulness | null
}

export function FaithfulnessBadge({ faithfulness }: Props) {
  if (!faithfulness) return null

  const pct = Math.round(faithfulness.score * 100)
  const isHigh = faithfulness.score >= 0.7
  const isMed = faithfulness.score >= 0.4

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-mono",
        "backdrop-blur-sm transition-colors",
        isHigh
          ? "border-success/25 bg-success/8 text-success"
          : isMed
            ? "border-warn/25 bg-warn/8 text-warn"
            : "border-error/25 bg-error/8 text-error",
      )}
      title={`${faithfulness.matched_count ?? "?"} of ${faithfulness.total_entities ?? "?"} entities verified in the knowledge graph`}
    >
      {isHigh ? (
        <ShieldCheck className="w-3 h-3" />
      ) : isMed ? (
        <Shield className="w-3 h-3" />
      ) : (
        <ShieldAlert className="w-3 h-3" />
      )}
      <span>Graph-verified · {pct}%</span>
    </div>
  )
}
