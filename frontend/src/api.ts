import type { ChatResponse, Message, StatusResponse } from "./types"
import { normalizeChatResponse } from "./lib/normalizeChatResponse"

const BASE = ""

export async function getStatus(): Promise<StatusResponse> {
  const res = await fetch(`${BASE}/api/status`)
  if (!res.ok) throw new Error(`Status ${res.status}`)
  return res.json()
}

export async function postChat(messages: Message[]): Promise<ChatResponse> {
  const payload = messages.map((m) => ({ role: m.role, content: m.content }))
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: payload }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: "Network error" }))
    throw new Error(err.message ?? `HTTP ${res.status}`)
  }
  const response = await res.json() as ChatResponse
  return normalizeChatResponse(response)
}

export async function getGraphData() {
  const res = await fetch(`${BASE}/api/graph-data`)
  if (!res.ok) throw new Error(`Graph data ${res.status}`)
  return res.json()
}
