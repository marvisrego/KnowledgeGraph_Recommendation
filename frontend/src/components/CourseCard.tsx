import { motion } from "framer-motion"
import { ExternalLink, BookOpen, Clock } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Course } from "@/types"

interface Props {
  course: Course
  index?: number
}

const TYPE_LABELS: Record<string, string> = {
  course: "Course",
  specialization: "Specialization",
  professional_certificate: "Certificate",
}

export function CourseCard({ course, index = 0 }: Props) {
  const label = TYPE_LABELS[course.content_type ?? "course"] ?? "Course"

  return (
    <motion.a
      href={course.url || undefined}
      target={course.url ? "_blank" : undefined}
      rel="noopener noreferrer"
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.07 }}
      className={cn(
        "flex-shrink-0 w-64 rounded-xl border border-line bg-bg-raised p-4 flex flex-col gap-2",
        "transition-all duration-200",
        course.url
          ? "cursor-pointer hover:border-indigo/50 hover:bg-bg-interactive hover:shadow-[0_0_0_1px_rgba(135,149,237,0.15)] hover:-translate-y-0.5"
          : "cursor-default",
      )}
      onClick={(e) => { if (!course.url) e.preventDefault() }}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <BookOpen className="w-4 h-4 text-indigo mt-0.5 flex-shrink-0" />
        <div className="flex gap-1 flex-wrap">
          <span className="rounded-full border border-indigo/20 bg-indigo/8 px-2 py-0.5 text-[9px] font-mono text-indigo uppercase tracking-wide">
            {label}
          </span>
          {course.estimated_workload && (
            <span className="flex items-center gap-0.5 rounded-full border border-line px-2 py-0.5 text-[9px] font-mono text-ink-muted">
              <Clock className="w-2.5 h-2.5" />
              {course.estimated_workload}
            </span>
          )}
        </div>
      </div>

      {/* Title */}
      <h4 className="text-sm font-medium text-ink line-clamp-2 leading-snug">{course.title}</h4>

      {/* Provider */}
      {course.provider && (
        <p className="text-[11px] text-ink-muted font-mono">{course.provider}</p>
      )}

      {/* Description */}
      {course.description && (
        <p className="text-[11px] text-ink-soft leading-relaxed line-clamp-2 flex-1">
          {course.description}
        </p>
      )}

      {/* Skills */}
      {course.skills && course.skills.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {course.skills.slice(0, 3).map((s) => (
            <span
              key={s}
              className="rounded-full border border-indigo/20 bg-indigo/8 px-2 py-0.5 text-[10px] text-indigo"
            >
              {s}
            </span>
          ))}
        </div>
      )}

      {/* Link indicator */}
      {course.url && (
        <div className="flex items-center gap-1 text-[11px] text-indigo mt-auto pt-1 border-t border-line/50">
          <ExternalLink className="w-3 h-3" />
          Open on Coursera
        </div>
      )}
    </motion.a>
  )
}
