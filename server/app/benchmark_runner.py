from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = APP_DIR / "evidence_log.jsonl"

DIAGNOSTIC_MAX_MASTERY = 0.55
DEFAULT_MASTERY_THRESHOLD = 0.7
DEFAULT_MODULE_THRESHOLD = 0.7


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                events.append({
                    "event": "invalid_json_line",
                    "line_no": line_no,
                    "error": str(exc),
                })
                continue
            if isinstance(item, dict):
                item["_line_no"] = line_no
                events.append(item)
    return events


def pct(ok: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(ok / total, 4)


def score_status(score: float, minimum: float) -> str:
    return "PASS" if score >= minimum else "FAIL"


def get_mastery_values(event: dict[str, Any], field: str) -> Iterable[float]:
    mastery = event.get(field)
    if not isinstance(mastery, dict):
        return []
    out: list[float] = []
    for value in mastery.values():
        try:
            out.append(float(value))
        except Exception:
            pass
    return out


def evaluate(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(e.get("event", "unknown")) for e in events)
    issues: list[dict[str, Any]] = []

    diagnostic_submitted = [e for e in events if e.get("event") == "diagnostic_submitted"]
    diagnostic_with_role = [
        e for e in diagnostic_submitted
        if e.get("evidence_role") == "screening_only"
    ]
    diagnostic_cap_ok = 0
    for event in diagnostic_submitted:
        profile_values = list(get_mastery_values(event, "diagnostic_profile"))
        before = event.get("mastery_before") if isinstance(event.get("mastery_before"), dict) else {}
        after = event.get("mastery_after") if isinstance(event.get("mastery_after"), dict) else {}
        mastery_unchanged = before == after
        profile_capped = bool(profile_values) and max(profile_values) <= DIAGNOSTIC_MAX_MASTERY + 1e-9
        if event.get('mode') == 'hierarchical':
            profile_capped = not profile_values and bool(event.get('statuses'))
        if mastery_unchanged and profile_capped:
            diagnostic_cap_ok += 1
        else:
            issues.append({
                "metric": "diagnostic_screening_cap",
                "line_no": event.get("_line_no"),
                "message": "Diagnostic should not update mastery and profile values must stay capped.",
                "mastery_before": before,
                "mastery_after": after,
                "max_diagnostic_profile": max(profile_values) if profile_values else None,
            })

    practice_started = [e for e in events if e.get("event") == "practice_started"]
    practice_length_ok = 0
    practice_coverage_ok = 0
    cumulative_ok = 0
    cumulative_candidates = 0
    grounded_practice = 0
    for event in practice_started:
        n = int(event.get("n_questions") or 0)
        min_q = int(event.get("practice_min_questions") or 0)
        max_q = int(event.get("practice_max_questions") or 0)
        essential_count = int(event.get("lesson_essential_count") or 0)
        if min_q <= n <= max_q and n >= min(essential_count, max_q):
            practice_length_ok += 1
        else:
            issues.append({
                "metric": "practice_length_adaptivity",
                "line_no": event.get("_line_no"),
                "message": "Practice length does not match lesson coverage bounds.",
                "n_questions": n,
                "min": min_q,
                "max": max_q,
                "lesson_essential_count": essential_count,
            })

        targets = event.get("lesson_essential_targets") or []
        coverage_plan = event.get("coverage_plan") or []
        if targets and isinstance(coverage_plan, list):
            expected_ids = {f"E{i}" for i in range(1, len(targets) + 1)}
            covered_ids = set()
            for item in coverage_plan:
                if not isinstance(item, dict):
                    continue
                qnums = item.get("question_numbers") or []
                if item.get("target_id") in expected_ids and isinstance(qnums, list) and qnums:
                    covered_ids.add(str(item.get("target_id")))
            if expected_ids.issubset(covered_ids):
                practice_coverage_ok += 1
            else:
                issues.append({
                    "metric": "practice_essential_coverage",
                    "line_no": event.get("_line_no"),
                    "message": "Practice QCM does not cover every essential lesson target.",
                    "missing_target_ids": sorted(expected_ids - covered_ids),
                    "expected_targets": targets,
                    "coverage_plan": coverage_plan,
                })
        elif not targets:
            issues.append({
                "metric": "practice_essential_coverage",
                "line_no": event.get("_line_no"),
                "message": "Practice event has no extracted lesson essential targets to verify.",
            })

        review_ids = event.get("review_kc_ids") or []
        if isinstance(review_ids, list) and review_ids:
            cumulative_candidates += 1
            cumulative_ok += 1
        if event.get("source_pages"):
            grounded_practice += 1

    practice_submitted = [e for e in events if e.get("event") == "practice_submitted"]
    practice_policy_ok = 0
    for event in practice_submitted:
        score = float(event.get("score") or 0.0)
        threshold = float(event.get("threshold") or DEFAULT_MASTERY_THRESHOLD)
        action = str(event.get("tutor_action") or "")
        passed = bool(event.get("passed"))
        misconceptions = event.get("misconceptions") or []

        valid = False
        if passed and score >= threshold and action in {"validate_kc", "validate_kc_then_module_checkpoint"}:
            valid = True
        if not passed and score < threshold:
            if misconceptions and action == "hint_ladder_then_remediate":
                valid = True
            if not misconceptions and action == "retry_with_micro_lesson":
                valid = True
        if valid:
            practice_policy_ok += 1
        else:
            issues.append({
                "metric": "practice_policy_consistency",
                "line_no": event.get("_line_no"),
                "message": "Practice tutor action does not match score/pass/misconception state.",
                "score": score,
                "threshold": threshold,
                "passed": passed,
                "tutor_action": action,
                "misconceptions": misconceptions,
            })

    module_submitted = [e for e in events if e.get("event") == "module_checkpoint_submitted"]
    module_policy_ok = 0
    for event in module_submitted:
        score = float(event.get("score") or 0.0)
        threshold = float(event.get("threshold") or DEFAULT_MODULE_THRESHOLD)
        action = str(event.get("tutor_action") or "")
        valid = (
            (score >= threshold and action == "validate_module")
            or (score < threshold and action == "retry_module_checkpoint")
        )
        if valid:
            module_policy_ok += 1
        else:
            issues.append({
                "metric": "module_policy_consistency",
                "line_no": event.get("_line_no"),
                "message": "Module tutor action does not match checkpoint score.",
                "score": score,
                "threshold": threshold,
                "tutor_action": action,
            })

    lessons = [e for e in events if e.get("event") == "micro_lesson_generated"]
    lesson_size_ok = 0
    grounded_lessons = 0
    lesson_pdf_basis_ok = 0
    for event in lessons:
        words = int(event.get("word_estimate") or 0)
        plan = event.get("lesson_plan") if isinstance(event.get("lesson_plan"), dict) else {}
        budget = event.get("lesson_budget_words") or plan.get("max_words") or 220
        try:
            budget = int(budget)
        except Exception:
            budget = 220
        if words <= budget:
            lesson_size_ok += 1
        else:
            issues.append({
                "metric": "micro_lesson_size",
                "line_no": event.get("_line_no"),
                "message": "Micro-lesson exceeds its adaptive lesson-plan budget.",
                "word_estimate": words,
                "lesson_budget_words": budget,
                "lesson_plan": plan,
                "kc_id": event.get("kc_id"),
                "kc_title": event.get("kc_title"),
            })
        if event.get("source_pages"):
            grounded_lessons += 1
        text = str(event.get("lesson_text") or event.get("lesson_preview") or "")
        if "PDF basis:" in text or "Source :" in text or "À retenir" in text:
            lesson_pdf_basis_ok += 1
        elif text:
            issues.append({
                "metric": "lesson_pdf_basis",
                "line_no": event.get("_line_no"),
                "message": "Micro-lesson does not expose the PDF basis used for the lesson.",
                "kc_id": event.get("kc_id"),
                "kc_title": event.get("kc_title"),
            })

    metrics = {
        "diagnostic_screening_role_rate": pct(len(diagnostic_with_role), len(diagnostic_submitted)),
        "diagnostic_mastery_cap_rate": pct(diagnostic_cap_ok, len(diagnostic_submitted)),
        "practice_length_adaptivity_rate": pct(practice_length_ok, len(practice_started)),
        "practice_essential_coverage_rate": pct(practice_coverage_ok, len(practice_started)),
        "cumulative_practice_rate": pct(cumulative_ok, cumulative_candidates),
        "practice_policy_consistency_rate": pct(practice_policy_ok, len(practice_submitted)),
        "module_policy_consistency_rate": pct(module_policy_ok, len(module_submitted)),
        "micro_lesson_size_rate": pct(lesson_size_ok, len(lessons)),
        "lesson_pdf_basis_rate": pct(lesson_pdf_basis_ok, len(lessons)),
        "practice_grounding_rate": pct(grounded_practice, len(practice_started)),
        "lesson_grounding_rate": pct(grounded_lessons, len(lessons)),
    }
    summary_score = round(sum(metrics.values()) / max(1, len(metrics)), 4)

    by_thread: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        by_thread[str(event.get("thread_id") or "unknown")][str(event.get("event") or "unknown")] += 1

    return {
        "status": score_status(summary_score, 0.8),
        "summary_score": summary_score,
        "metrics": metrics,
        "event_counts": dict(counts),
        "threads": {thread_id: dict(counter) for thread_id, counter in by_thread.items()},
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ITS behavior from evidence_log.jsonl.")
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="Path to evidence_log.jsonl")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report")
    args = parser.parse_args()

    report = evaluate(load_events(Path(args.log)))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "PASS" else 1

    print(f"ITS benchmark: {report['status']}  score={report['summary_score']:.2%}")
    print("\nMetrics:")
    for name, value in report["metrics"].items():
        print(f"- {name}: {value:.2%}")

    print("\nEvent counts:")
    for name, count in sorted(report["event_counts"].items()):
        print(f"- {name}: {count}")

    if report["issues"]:
        print("\nIssues:")
        for issue in report["issues"][:20]:
            print(f"- line {issue.get('line_no')}: {issue['metric']} - {issue['message']}")
    else:
        print("\nIssues: none")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
