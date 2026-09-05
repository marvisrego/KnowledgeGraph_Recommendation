export interface Role {
  id: string
  title: string
  source: "onet" | "esco" | string
  description?: string
  prep?: string
  accessibility?: string
  effort_score?: number
  effort_band?: "low" | "moderate" | "high"
  effort_source?: "from_current_role" | "relative"
  estimated_weeks_min?: number
  estimated_weeks_max?: number
  qualification_score?: number
  have?: string[]
  need?: string[]
  transition?: {
    count?: number
    probability?: number
    type?: string
  }
  retrieval_sources?: string[]
  have_count?: number
  need_count?: number
  required_skill_count?: number
}

export interface SkillReference {
  id?: string
  title?: string
  name?: string
}

export type ApiSkill = string | SkillReference

export interface ApiRole extends Omit<Role, "have" | "need"> {
  have?: ApiSkill[]
  need?: ApiSkill[]
}

export interface SkillGapEntry {
  role_id: string
  role_title: string
  priority_skills: Array<{ skill: string; idf_weight: number; tes_reduction: number }>
  quick_wins: Array<{ skill: string; idf_weight: number }>
  blockers: Array<{ skill: string; idf_weight: number }>
}

export interface LearningPhase {
  phase: number
  weeks: string
  focus: string
  courses: Course[]
  skills_unlocked: number
  skills?: string[]
}

export interface LearningPlanEntry {
  role_id: string
  role_title: string
  weeks_to_ready: number
  phases: LearningPhase[]
}

export interface Course {
  title: string
  provider?: string
  url?: string
  skills?: string[]
  description?: string
  estimated_workload?: string
  content_type?: string
}

export interface ExplanationStep {
  relation: string
  from: string
  to: string
  attrs?: Record<string, unknown>
}

export interface ApiExplanationStep extends Partial<ExplanationStep> {
  source?: string
  target?: string
  attributes?: Record<string, unknown>
}

export interface Explanation {
  source_role: string
  target_role: string
  steps: ExplanationStep[]
}

export interface ApiExplanation extends Partial<Omit<Explanation, "steps">> {
  role_title?: string
  steps?: ApiExplanationStep[]
}

export interface Faithfulness {
  score: number
  total_entities?: number
  matched_count?: number
  reachable_count?: number
  matched_entities?: string[]
  unmatched_entities?: string[]
  unreachable_entities?: string[]
}

export interface PathData {
  roles: Role[]
  skills?: unknown[]
  similar_pairs?: unknown[]
}

export interface ExploreData {
  roles?: Role[]
  skills?: string[]
}

export interface ChatResponse {
  status: "ok" | "error"
  message: string
  courses?: Course[]
  path?: PathData
  explore?: ExploreData
  evidence?: Record<string, unknown>
  faithfulness?: Faithfulness
  explanations?: Explanation[]
  skill_gap_analysis?: SkillGapEntry[]
  learning_plan?: LearningPlanEntry[]
  metadata?: Record<string, number>
}

export interface Message {
  role: "user" | "assistant"
  content: string
  payload?: ChatResponse
}

export interface StatusResponse {
  ready: boolean
  graph_loaded: boolean
  qdrant_loaded: boolean
  graph_nodes: number
  graph_edges: number
  chat_model: string
  embed_model: string
  use_langgraph?: boolean
  error?: string
}
