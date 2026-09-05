import type {
  ApiExplanation,
  ApiExplanationStep,
  ApiRole,
  ApiSkill,
  ChatResponse,
  Explanation,
  ExplanationStep,
  Role,
} from "@/types"

function skillTitle(skill: ApiSkill | null | undefined): string | null {
  if (typeof skill === "string") {
    const value = skill.trim()
    return value || null
  }
  if (!skill || typeof skill !== "object") return null
  const value = skill.title ?? skill.name
  return typeof value === "string" && value.trim() ? value.trim() : null
}

function stringList(values: ApiSkill[] | null | undefined): string[] {
  if (!Array.isArray(values)) return []
  return values.map(skillTitle).filter((value): value is string => Boolean(value))
}

function normalizeRole(role: ApiRole): Role {
  return {
    ...role,
    qualification_score:
      typeof role.qualification_score === "number" ? role.qualification_score : undefined,
    have: stringList(role.have),
    need: stringList(role.need),
    retrieval_sources: Array.isArray(role.retrieval_sources)
      ? role.retrieval_sources.filter((value): value is string => typeof value === "string")
      : [],
  }
}

function normalizeStep(step: ApiExplanationStep): ExplanationStep {
  return {
    relation: typeof step.relation === "string" ? step.relation : "RELATED_TO",
    from: step.from ?? step.source ?? "",
    to: step.to ?? step.target ?? "",
    attrs: step.attrs ?? step.attributes,
  }
}

function normalizeExplanation(explanation: ApiExplanation): Explanation {
  const steps = Array.isArray(explanation.steps) ? explanation.steps.map(normalizeStep) : []
  return {
    source_role: explanation.source_role ?? steps[0]?.from ?? "",
    target_role: explanation.target_role ?? explanation.role_title ?? steps.at(-1)?.to ?? "",
    steps,
  }
}

export function normalizeChatResponse(raw: ChatResponse): ChatResponse {
  const path = raw.path
  return {
    ...raw,
    message: typeof raw.message === "string" ? raw.message : "Recommendation generated.",
    path: path
      ? {
          ...path,
          roles: Array.isArray(path.roles)
            ? (path.roles as unknown as ApiRole[]).map(normalizeRole)
            : [],
        }
      : { roles: [] },
    explanations: Array.isArray(raw.explanations)
      ? (raw.explanations as unknown as ApiExplanation[]).map(normalizeExplanation)
      : [],
  }
}
