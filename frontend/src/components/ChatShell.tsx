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
    <div className="effort-legend flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
      <span className="text-ink-muted">{roles.length} paths · sorted by effort</span>
      <span className="flex flex-wrap items-center gap-3">
        {low > 0 && <span className="text-success">● {low} low</span>}
        {mod > 0 && <span className="text-warn">● {mod} moderate</span>}
        {high > 0 && <span className="text-error">● {high} high</span>}
        {none > 0 && <span className="text-ink-muted">● {none} unscored</span>}
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
    <div className="assistant-payload">
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
          <div className="role-grid">
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
          <h3 className="result-section-title">
            Recommended courses
          </h3>
          <div className="course-track" tabIndex={0} aria-label="Recommended courses">
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
    <div className="path-panel rounded-xl border border-line bg-bg-raised overflow-hidden">
      <div className="flex border-b border-line">
        {["gap", "plan"].map((t) => (
          <button
            key={t}
            aria-pressed={tab === t}
            onClick={() => setTab(t as "gap" | "plan")}
            className={cn(
              "flex-1 px-3 py-3 text-sm font-medium transition-colors",
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
  const welcomeRef = useRef<HTMLDivElement>(null)
  const isWelcome = messages.length === 1

  // Decorative only: no chat state updates or interception of input events.
  useEffect(() => {
    const area = welcomeRef.current
    if (!isWelcome || !area) return
    const enabled = window.matchMedia("(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)")
    let frame = 0
    let x = 0
    let y = 0
    const hide = () => {
      cancelAnimationFrame(frame)
      frame = 0
      area.style.setProperty("--glow-visible", "0")
    }
    const move = (event: PointerEvent) => {
      if (!enabled.matches || event.pointerType !== "mouse") return
      const bounds = area.getBoundingClientRect()
      x = event.clientX - bounds.left
      y = event.clientY - bounds.top
      if (frame) return
      frame = requestAnimationFrame(() => {
        area.style.setProperty("--glow-x", `${x}px`)
        area.style.setProperty("--glow-y", `${y}px`)
        area.style.setProperty("--glow-visible", "1")
        frame = 0
      })
    }
    area.addEventListener("pointermove", move, { passive: true })
    area.addEventListener("pointerleave", hide)
    area.addEventListener("pointercancel", hide)
    enabled.addEventListener("change", hide)
    return () => {
      hide()
      area.removeEventListener("pointermove", move)
      area.removeEventListener("pointerleave", hide)
      area.removeEventListener("pointercancel", hide)
      enabled.removeEventListener("change", hide)
    }
  }, [isWelcome])

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
    <div className="chat-shell">
      {/* Toolbar */}
      <header className="chat-toolbar">
        <div className="flex items-center gap-2">
          <Network aria-hidden="true" className="w-5 h-5 text-accent" />
          <h2 className="text-base font-medium text-ink">Career Advisor</h2>
        </div>
        <button
          onClick={handleNewChat}
          className="new-chat-button"
        >
          <RotateCcw aria-hidden="true" className="w-4 h-4" /> New chat
        </button>
      </header>

      {/* Messages */}
      <div
        ref={chatLogRef}
        className={cn("chat-log", messages.length === 1 && "chat-log-welcome")}
        role="log"
        aria-label="Conversation"
        aria-live="polite"
        style={{ scrollbarWidth: "thin" }}
      >
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
              className={cn("message-row", msg.role === "user" ? "message-row-user" : "message-row-assistant", messages.length === 1 && "message-row-welcome")}
            >
              <div
                className={cn(
                  "message-body",
                  msg.role === "user"
                    ? "user-message"
                    : "assistant-message",
                )}
              >
                {messages.length === 1 && (
                  <div className="welcome-heading" ref={welcomeRef}>
                    <div className="welcome-atmosphere" aria-hidden="true">
                      <svg className="welcome-network" viewBox="0 0 400 240" fill="none">
                        <path d="M40 150 130 60 230 110 330 35M130 60l35 145 65-95 125 85M40 150l125 55 190-10M230 110l100-75" />
                        <circle cx="40" cy="150" r="5" /><circle cx="130" cy="60" r="7" />
                        <circle cx="165" cy="205" r="5" /><circle cx="230" cy="110" r="9" />
                        <circle cx="330" cy="35" r="5" /><circle cx="355" cy="195" r="6" />
                      </svg>
                    </div>
                    <div className="welcome-mark" aria-hidden="true"><Network /></div>
                    <h2>Your next move,<br /><span>grounded in evidence.</span></h2>
                  </div>
                )}
                {messages.length > 1 && <p className="message-author">{msg.role === "user" ? "You" : "Career Advisor"}</p>}
                <p className="message-copy">
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
              className="message-row flex justify-start"
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
              role="alert"
              className="message-row rounded-xl border border-error/30 bg-error/8 px-4 py-3 text-sm text-error"
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
        className="chat-composer"
      >
        <label htmlFor="career-message" className="composer-label">Your role, skills, and next step</label>
        <div className="composer-field">
        <textarea
          id="career-message"
          name="message"
          autoComplete="off"
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Tell me your role, skills, and where you want to go…"
          rows={2}
          maxLength={4000}
          disabled={loading}
          className="composer-input"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="send-button"
          aria-label="Send"
        >
          {loading ? (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
            >
              <RotateCcw aria-hidden="true" className="w-5 h-5" />
            </motion.div>
          ) : (
            <Send aria-hidden="true" className="w-5 h-5" />
          )}
        </button>
        </div>
        <div className="composer-hint" aria-hidden="true"><span>Start with where you are. Explore where you could go.</span><span>Enter to send <span className="hint-separator">/</span> Shift + Enter for a new line</span></div>
      </form>
    </div>
  )
}
