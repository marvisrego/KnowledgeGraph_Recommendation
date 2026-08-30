import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Send, RotateCcw, Network } from "lucide-react"
import { cn } from "@/lib/utils"
import { postChat } from "@/api"
import { PipelineIndicator } from "./PipelineIndicator"
import { RoleCard } from "./RoleCard"
import { CourseCard } from "./CourseCard"
import { FaithfulnessBadge } from "./FaithfulnessBadge"
import { EvidencePanel } from "./EvidencePanel"
import { SkillGapCard } from "./SkillGapCard"
import { LearningRoadmap } from "./LearningRoadmap"
import { StatsTiles } from "./StatsTiles"
import { TransferableSkills } from "./TransferableSkills"
import type { Message, ChatResponse, Role } from "@/types"

function EffortLegend({ roles }: { roles: Role[] }) {
  const low = roles.filter(r => r.effort_band === "low").length
  const mod = roles.filter(r => r.effort_band === "moderate").length
  const high = roles.filter(r => r.effort_band === "high").length
  const none = roles.filter(r => !r.effort_band).length
  return (
    <div className="flex items-center gap-3 text-[11px] font-mono">
      <span className="text-ink-muted">{roles.length} paths · sorted by effort</span>
      <span className="flex items-center gap-1">
        {low > 0 && <span className="text-success">● {low} low</span>}
        {mod > 0 && <span className="text-warn">● {mod} moderate</span>}
        {high > 0 && <span className="text-error">● {high} high</span>}
        {none > 0 && <span className="text-ink-muted/50">● {none} unscored</span>}
      </span>
    </div>
  )
}

function MarkdownText({ text }: { text: string }) {
  // Bold only — safe, no XSS
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return (
    <span>
      {parts.map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </span>
  )
}

function AssistantPayload({ payload }: { payload: ChatResponse }) {
  const roles = payload.path?.roles ?? []
  const courses = payload.courses ?? []
  const explanations = payload.explanations ?? []
  const faithfulness = payload.faithfulness
  const skillGapAnalysis = payload.skill_gap_analysis ?? []
  const learningPlan = payload.learning_plan ?? []
  const topRole = roles[0]

  return (
    <div className="flex flex-col gap-4 mt-2">
      {/* Faithfulness badge */}
      {faithfulness && <FaithfulnessBadge faithfulness={faithfulness} />}

      {/* Stats tiles for top role */}
      {topRole && <StatsTiles role={topRole} />}

      {/* Career path roles — vertical grid, sorted by effort */}
      {roles.length > 0 && (
        <div>
          <div className="mb-2">
            <EffortLegend roles={roles} />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {roles.map((role, i) => (
              <RoleCard key={role.id} role={role} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* Transferable skills */}
      {roles.length > 0 && <TransferableSkills roles={roles} />}

      {/* Skill gap analysis + Learning roadmap as unified "Your Path" */}
      {(skillGapAnalysis.length > 0 || learningPlan.length > 0) && (
        <YourPathPanel skillGapAnalysis={skillGapAnalysis} learningPlan={learningPlan} />
      )}

      {/* Courses — always shown */}
      {courses.length > 0 && (
        <div>
          <p className="text-[11px] text-ink-muted font-mono uppercase tracking-wider mb-2">
            Recommended courses
          </p>
          <div className="flex gap-3 overflow-x-auto pb-2" style={{ scrollbarWidth: "thin" }}>
            {courses.map((course, i) => (
              <CourseCard key={course.title} course={course} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* Evidence panel */}
      {(explanations.length > 0 || faithfulness) && (
        <EvidencePanel explanations={explanations} faithfulness={faithfulness} />
      )}
    </div>
  )
}

function YourPathPanel({
  skillGapAnalysis,
  learningPlan,
}: {
  skillGapAnalysis: ChatResponse["skill_gap_analysis"]
  learningPlan: ChatResponse["learning_plan"]
}) {
  const [tab, setTab] = useState<"gap" | "plan">("gap")
  return (
    <div className="rounded-xl border border-line bg-bg-raised overflow-hidden">
      <div className="flex border-b border-line">
        {["gap", "plan"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t as "gap" | "plan")}
            className={cn(
              "flex-1 py-2.5 text-xs font-mono uppercase tracking-wide transition-colors",
              tab === t ? "text-accent bg-accent/8 border-b-2 border-accent" : "text-ink-muted hover:text-ink",
            )}
          >
            {t === "gap" ? "Skill Gap Analysis" : "Learning Plan"}
          </button>
        ))}
      </div>
      <div className="p-0">
        {tab === "gap" && (skillGapAnalysis?.length ?? 0) > 0 && (
          <SkillGapCard entries={skillGapAnalysis!} />
        )}
        {tab === "plan" && (learningPlan?.length ?? 0) > 0 && (
          <LearningRoadmap plan={learningPlan!} />
        )}
      </div>
    </div>
  )
}

export function ChatShell() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm your Career Graph Advisor. Tell me about your current role and skills, and where you'd like to go next — I'll map the best paths using real career transition data.",
    },
  ])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const chatLogRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Scroll to new message — respects user scroll position
  const scrollToLatest = useCallback(() => {
    const log = chatLogRef.current
    if (!log) return
    const userScrolledUp = log.scrollTop < log.scrollHeight - log.clientHeight - 200
    if (userScrolledUp) return
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
  }, [])

  useEffect(() => {
    scrollToLatest()
  }, [messages, loading, scrollToLatest])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    const userMsg: Message = { role: "user", content: query }
    const nextMessages = [...messages, userMsg]
    setMessages(nextMessages)
    setInput("")
    setLoading(true)
    setError(null)

    try {
      const response = await postChat(nextMessages)
      const assistantMsg: Message = {
        role: "assistant",
        content: response.message,
        payload: response,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.")
    } finally {
      setLoading(false)
    }
  }

  const handleNewChat = () => {
    abortRef.current?.abort()
    setMessages([
      {
        role: "assistant",
        content:
          "Hello! I'm your Career Graph Advisor. Tell me about your current role and skills, and where you'd like to go next.",
      },
    ])
    setInput("")
    setError(null)
    setLoading(false)
    inputRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as unknown as React.FormEvent)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-line px-4 py-3 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-accent" />
          <span className="text-sm font-semibold text-ink">Career Advisor</span>
        </div>
        <button
          onClick={handleNewChat}
          className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs text-ink-muted hover:text-ink hover:border-accent/40 transition-colors"
        >
          <RotateCcw className="w-3 h-3" /> New chat
        </button>
      </div>

      {/* Messages */}
      <div
        ref={chatLogRef}
        className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4"
        style={{ scrollbarWidth: "thin" }}
      >
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
              className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}
            >
              <div
                className={cn(
                  "rounded-xl px-4 py-3 max-w-[85%]",
                  msg.role === "user"
                    ? "bg-accent/10 border border-accent/20 text-ink text-sm"
                    : "bg-bg-raised border border-line border-l-accent border-l-2 text-ink text-sm",
                )}
              >
                <p className="leading-relaxed">
                  <MarkdownText text={msg.content} />
                </p>
                {msg.payload && <AssistantPayload payload={msg.payload} />}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading state */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex justify-start"
            >
              <PipelineIndicator visible={loading} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="rounded-xl border border-error/30 bg-error/8 px-4 py-3 text-sm text-error"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={chatEndRef} />
      </div>

      {/* Composer */}
      <form
        onSubmit={handleSubmit}
        className="flex-shrink-0 border-t border-line px-4 py-3 flex gap-2 items-end"
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Tell me your role, skills, and where you want to go…"
          rows={2}
          maxLength={4000}
          disabled={loading}
          className={cn(
            "flex-1 resize-none rounded-xl border border-line bg-bg-raised px-4 py-3",
            "text-sm text-ink placeholder:text-ink-muted/50 outline-none",
            "focus:border-accent/50 focus:ring-0 transition-colors",
            "disabled:opacity-50",
          )}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className={cn(
            "flex-shrink-0 flex items-center justify-center rounded-full w-10 h-10",
            "border border-accent/30 bg-accent/10 text-accent",
            "hover:bg-accent/20 hover:border-accent/60 transition-all duration-200",
            "disabled:opacity-30 disabled:cursor-not-allowed",
          )}
          aria-label="Send"
        >
          {loading ? (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
            >
              <RotateCcw className="w-4 h-4" />
            </motion.div>
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </form>
    </div>
  )
}
