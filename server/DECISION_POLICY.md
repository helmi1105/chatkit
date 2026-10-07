# Auditable baseline decision policy

## Resuming a simulation

From `server` with the API key exported in your shell:

```bash
python simulation/run.py --resume simulation/runs/20260928T143644Z-323c8b4f --max-steps 200 --max-model-calls 150
```

Resume restores the original provider, profile, seed, learner ID, assessment
thresholds and attempt limit. Step/model-call limits are **additional allowances**;
attempt limits remain cumulative per KC/chapter. Settings and old logs are retained;
new records append to the same run. Do not run two processes on the same run folder.

New runs save `checkpoint.json` at each completed step and graceful shutdown,
including the session, answer/bank random states, submission counters and hints.
Resume uses that checkpoint. Hard termination during a step may leave later log
records beyond the checkpoint; resumption restarts from the checkpoint boundary,
so generation may be repeated. Live model output is not deterministic.

Legacy finished runs reconstruct answer RNG/counters from logged requests and
verify the recorded choices. Bank RNG is reseeded and this limitation is logged.
The existing September 28 P1 run was checked offline: reconstruction preserves
13 validated KCs and resumes at g_k19. No simulation was run against an API.

Validation quizzes retain valid questions and allow two additional generation
attempts for missing objectives/insufficient question counts. All repair calls
are budgeted. Persistent generation errors still stop safely and can be resumed.

After all KCs are validated, outstanding real chapter checkpoints are presented
through the orchestrator. `course_finished` requires all current curriculum KCs
and all non-ROOT chapter checkpoints to pass. Other stop statuses are incomplete
and return a nonzero exit code; `run_finished.completion` lists remaining IDs.

`app/decision_policy.py` contains pure learner-state snapshot construction and
`decide_next_action`. The hierarchical diagnostic and post-practice routing call
this policy. Existing thresholds, ordering, follow-up limits and challenge routes
are preserved. Flat diagnostic and chapter checkpoint routing remain unchanged.

Actions are `probe`, `teach`, `validate` (administer a challenge), `remediate`,
`review`, `advance` (practice passed), and `finish` (diagnostic stopped).
Finish does not imply course mastery. Decisions do not execute lessons or mutate
sessions; the orchestrator remains responsible for execution and persistence.

New `pedagogical_decision` evidence events contain the versioned decision,
pre-action learner snapshot, curriculum groups and applicable practice thresholds.
An event records a selected action, not proof that generation or delivery succeeded.
Existing generation/submission events record subsequent execution.

Snapshots separate diagnostic observations/status from practice scores and
validation. They include practice attempts, recorded confusion labels (not proven
misconceptions), known objectives and newly recorded practice quiz evidence,
including assessed target IDs. Missing historical evidence stays missing. Objective
IDs are quiz-local labels; do not treat E1 across different quizzes as a stable
global objective. Lesson availability is not proof the learner read the lesson.
Legacy `after_instruction` diagnostic flags are retained as recorded, not corrected
or used as causal evidence. Confidence is null, not an invented probability.

The curriculum supplies chapter membership and order only. Prerequisite reasoning
requires separately reviewed prerequisite edges and a future policy version.
This baseline is not a claim of improved learning or expert decision accuracy.

From `server`, replay newly collected evidence without API calls:

```bash
python -m app.eval_decisions path/to/evidence.jsonl
```

For expert evaluation, copy events to an annotation dataset and add an
`acceptable_decisions` list, e.g. `[{"action":"review","target_kc":"g_k2"}]`.
Multiple acceptable actions are supported. Without annotations, expert agreement
is null. Replay consistency is not pedagogical correctness. Historical logs from
before this change cannot be reconstructed reliably and yield no replayed events.

Offline tests:

```bash
python -m unittest test_decision_policy test_diagnostic_policy -q
python -m unittest discover -s simulation -p 'test_*.py' -q
```
