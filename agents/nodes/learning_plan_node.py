"""Learning Plan node — builds phased upskill roadmap per recommended role."""

from __future__ import annotations

import math
import time
from typing import Any

from agents.state import CareerAgentState


def learning_plan_node(state: CareerAgentState, *, G, **kwargs) -> dict[str, Any]:
    """Combine skill gap priorities and fetched courses into a phased learning plan.

    Each phase groups ~3 priority skills with matching course(s) and an
    estimated week range derived from effort data.
    """
    t0 = time.time()
    skill_gap_analysis: list[dict] = state.get("skill_gap_analysis", [])
    courses: list[dict] = state.get("courses", [])
    path_data = state.get("path_data", {})
    roles = path_data.get("roles", [])

    if not skill_gap_analysis:
        return {
            "learning_plan": [],
            "metadata": {**state.get("metadata", {}), "learning_plan_ms": 0},
        }

    # Build a quick lookup: course title keywords → course dict
    def _course_keywords(course: dict) -> set[str]:
        text = (course.get("title", "") + " " + " ".join(course.get("skills", []))).lower()
        return set(text.split())

    course_index = [(c, _course_keywords(c)) for c in courses]

    def _find_course_for_skill(skill: str) -> dict | None:
        skill_words = set(skill.lower().split())
        best_score = 0
        best_course = None
        for course, keywords in course_index:
            score = len(skill_words & keywords)
            if score > best_score:
                best_score = score
                best_course = course
        return best_course if best_score > 0 else None

    # Build effort data lookup from path roles
    effort_lookup: dict[str, tuple[int, int]] = {}
    for r in roles:
        rid = r.get("id", "")
        if rid:
            wmin = r.get("estimated_weeks_min", 0)
            wmax = r.get("estimated_weeks_max", 0)
            effort_lookup[rid] = (wmin, wmax)

    learning_plan: list[dict] = []

    for gap in skill_gap_analysis:
        role_id = gap["role_id"]
        priority_skills = gap.get("priority_skills", [])

        if not priority_skills:
            continue

        weeks_min, weeks_max = effort_lookup.get(role_id, (0, 0))
        total_weeks = (weeks_min + weeks_max) // 2 if weeks_max else 0

        # Split skills into phases of ~3 skills each
        phase_size = 3
        num_phases = max(1, math.ceil(len(priority_skills) / phase_size))
        weeks_per_phase = max(2, total_weeks // num_phases) if total_weeks else 4

        phases = []
        cumulative_week = 1
        for i in range(num_phases):
            phase_skills = priority_skills[i * phase_size : (i + 1) * phase_size]
            if not phase_skills:
                break

            # Find matching courses for this phase's skills
            phase_courses: list[dict] = []
            seen_titles: set[str] = set()
            for ps in phase_skills:
                course = _find_course_for_skill(ps["skill"])
                if course and course.get("title") not in seen_titles:
                    phase_courses.append(course)
                    seen_titles.add(course.get("title", ""))

            week_start = cumulative_week
            week_end = cumulative_week + weeks_per_phase - 1
            cumulative_week = week_end + 1

            # Pick focus theme from first skill's domain
            focus = phase_skills[0]["skill"] if phase_skills else f"Phase {i+1}"

            phases.append({
                "phase": i + 1,
                "weeks": f"{week_start}–{week_end}",
                "focus": focus,
                "courses": phase_courses[:2],
                "skills_unlocked": len(phase_skills),
                "skills": [ps["skill"] for ps in phase_skills],
            })

        learning_plan.append({
            "role_id": role_id,
            "role_title": gap["role_title"],
            "weeks_to_ready": total_weeks or (num_phases * 4),
            "phases": phases,
        })

    return {
        "learning_plan": learning_plan,
        "metadata": {**state.get("metadata", {}), "learning_plan_ms": int((time.time() - t0) * 1000)},
    }
