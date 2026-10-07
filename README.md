# PDF2ITS: PDF-Grounded LLM Intelligent Tutoring System

PDF2ITS is a self-hosted OpenAI ChatKit application that transforms PDF instructional material into an adaptive Intelligent Tutoring System (ITS).

The system is not only a PDF chatbot. It uses a knowledge-component graph, diagnostic screening, adaptive micro-lessons, adaptive QCM practice, feedback, remediation, mastery tracking, progress visualization, and visual symbol question answering.

## What It Does

```text
PDF course material
-> KC graph
-> diagnostic QCM
-> learner profile
-> adaptive micro-lesson
-> adaptive practice QCM
-> feedback and remediation
-> mastery tracking
-> next KC / module checkpoint
```

## Features

- Course-grounded generation from local course passages, with optional OpenAI file search on supported paths.
- Public username/password registration, login and logout.
- Conversations and learning progress saved per authenticated learner ID.
- KC-based course navigation using `server/app/kc_graph1.json`.
- Diagnostic QCM for learner-level estimation.
- Adaptive micro-lessons based on learner state.
- Adaptive QCM length based on KC targets, lesson content, mistakes, attempts, and difficulty.
- Cumulative practice with current KC plus previous KCs.
- Mistake explanation and misconception tracking.
- Hint ladder for remediation.
- KC and module mastery tracking.
- Radar progress visualization.
- Image upload for symbol interpretation.
- Follow-up visual questions using the same uploaded image.
- Evidence logging in JSONL.
- Benchmark runner for automatic ITS behavior checks.
- Learner guide available in chat and as a markdown document.

## Architecture

```text
web/
  Next.js frontend with @openai/chatkit-react

server/
  FastAPI backend

server/app/chatkit_server.py
  ChatKit server adapter, widget rendering, QCM submit handling, image conversion

server/app/orchestrator.py
  ITS workflow, agents, learner model, pedagogical controller

server/app/data_store.py
  Per-learner threads/messages persisted through the state store; local image attachments

server/app/auth.py
  Accounts, password hashing, browser sessions and optional account creation CLI

server/app/storage.py
  Local or S3-compatible persistence for accounts, learner state and events

server/app/widgets/
  ChatKit widgets for QCM, study cards, maps, Plotly, and radar

server/app/benchmark_runner.py
  Automatic benchmark using evidence_log.jsonl

docs/learner_guide.md
  Learner-facing usage guide
```

## Main Agents

- `DiagnosticQcmAgent`: creates the global diagnostic QCM.
- `KcEssentialTargetAgent`: extracts assessable KC targets from the PDF.
- `MicroLessonAgent`: creates adaptive PDF-grounded micro-lessons.
- `PracticeQcmAgent`: creates and repairs adaptive practice QCMs.
- `ExplainMistakeAgent`: explains wrong answers.
- `LearnerQuestionAgent`: answers text questions during lessons.
- `VisualQuestionAgent`: answers questions about uploaded symbol images.
- `TutorDecisionPolicy`: decides validate, retry, remediate, hint, or next.
- `MisconceptionTracker`: records simple misconception evidence from wrong answers.

## Requirements

- Python 3.11+ (the backend Docker image uses Python 3.11).
- Node.js 20+ (the frontend Docker image uses Node.js 20).
- A Mistral key for the default tutoring provider, or an OpenAI key when using OpenAI.
- An OpenAI key for image analysis and the dedicated free-question answering path, even when Mistral is selected for tutoring.
- An OpenAI vector store only for paths using hosted file search; image retrieval uses local course passages.

Backend dependencies are listed in:

```text
server/requirements.txt
```

Frontend dependencies are listed in:

```text
web/package.json
```

## Configuration

The examples below use **Git Bash on Windows**, from the project root. Set keys
in the same terminal that starts the backend; do not commit real keys.

```bash
export DEFAULT_PROVIDER="mistral"
export MISTRAL_API_KEY="YOUR_MISTRAL_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

The default Mistral model is `mistral/mistral-large-latest`. Set `MISTRAL_MODEL`
to a model available to your account if necessary. For OpenAI tutoring, set
`DEFAULT_PROVIDER=openai`; the default `OPENAI_MODEL` is `gpt-4.1`.
The chat settings also allow provider selection and an optional personal key.

Optional backend settings:

```bash
export VECTOR_STORE_ID="YOUR_VECTOR_STORE_ID"
export PUBLIC_BASE_URL="http://127.0.0.1:8000"
```

`VECTOR_STORE_ID` applies to hosted file-search paths; configure your own store
when using them. `PUBLIC_BASE_URL` is used for public course asset links.

## Installation

Backend:

```bash
python -m venv server/.venv
source server/.venv/Scripts/activate
python -m pip install -r server/requirements.txt
```

Frontend:

```bash
cd web
npm install
cd ..
```

## Run Locally

Backend:

```bash
source server/.venv/Scripts/activate
python -m uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal from the project root:

```bash
cd web
npm run dev
```

Open:

```text
http://localhost:3000
```

If Next.js uses another port such as `3001`, add that exact frontend origin to
the backend's `ALLOWED_ORIGINS` and restart the backend. The defaults allow
`http://localhost:3000` and `http://127.0.0.1:3000`.

## Learner Commands

```text
help
aide
start diagnostic
practice
hint
next
radar
clear image
```

Typical learner workflow:

```text
start diagnostic
-> answer diagnostic QCM
-> read micro-lesson
-> ask questions if needed
-> practice
-> answer adaptive QCM
-> read feedback / use hint if needed
-> retry practice or type next
-> radar
```

## Learner accounts

Select **Créer un compte**, enter a username and password, and registration signs
you in automatically. Usernames are case-insensitive and contain 3–64 letters,
digits, dots, underscores or hyphens. Passwords require at least **4 characters**;
letters, digits or a mix are accepted. Existing users select **Se connecter**.

Conversations and progress are linked to the authenticated permanent user ID,
so signing into the same account restores them across browsers. Passwords are
salted and hashed; the browser uses an HttpOnly session cookie. **Se déconnecter**
revokes that session. If `ACCESS_CODE` is configured, it remains an additional
shared gate for learning access.

Administrators can optionally run `python -m app.auth learner_001` from `server/`
with the virtual environment active. See [account setup](server/ACCOUNTS.md)
for linking existing anonymous progress and storage configuration.

## Response timeouts

Model calls have a 90-second deadline (including queueing and retries), configurable
with `LLM_TIMEOUT_SECONDS`. A complete tutor response has a 180-second deadline,
configurable with `CHAT_RESPONSE_TIMEOUT_SECONDS`. The browser also aborts requests
that have not completed after 210 seconds, including stalled response streams.
Keep the server deadlines below this browser limit. Timeouts show a retry message;
closing a response stream cancels its pending asynchronous generation task.

## Deployment settings

The frontend calls `/backend/*` through a Next.js proxy so login cookies remain
on the frontend origin. Set `CHATKIT_BACKEND_URL` (or the existing
`NEXT_PUBLIC_CHATKIT_API_URL` Docker build argument) to the backend address before
building the frontend. Its local default is `http://127.0.0.1:8000`.

On the production backend, configure:

```bash
export ALLOWED_ORIGINS="https://YOUR-FRONTEND-HOST"
export AUTH_COOKIE_SECURE="true"
export PUBLIC_BASE_URL="https://YOUR-BACKEND-HOST"
```

Use an explicit frontend origin rather than `*`, and serve production over HTTPS.
Accounts and learning state use the configured S3 backend or local `app/data/`;
local container data requires persistent storage to survive replacement.
See [account deployment details](server/ACCOUNTS.md) and [Scaleway deployment](DEPLOY_SCALEWAY.md).

## Image Upload

Image questions follow three steps:

1. Observe visible shapes, colours, lettering and relationships without assigning a domain meaning yet.
2. Search the entire course separately for each observed element, retaining source page numbers.
3. Explain using the original image, observations, your question and relevant course passages, with instructions to cite pages and acknowledge uncertainty.

Both model stages use **OpenAI**, even when Mistral is selected for tutoring.
`OPENAI_INVENTORY_MODEL` and `OPENAI_VISUAL_EXPLANATION_MODEL` both default to
`gpt-5.4`. They use the server's `OPENAI_API_KEY`, or a personal OpenAI key when
OpenAI is selected. A personal Mistral key cannot be used for these stages.

The default retrieval mode is **`VISUAL_RETRIEVAL_MODE=bm25`**, using local keyword
search without embedding requests. Reference passages come from
`doctrine_pages.json` and optional OCR content. Set `VISUAL_RETRIEVAL_MODE=hybrid`
to combine BM25 with OpenAI embeddings (default `text-embedding-3-small`). In
hybrid mode, course text and image descriptions are sent to the embeddings API;
embeddings do not process image pixels.

Prompts and implementation:

- Observation: `OBSERVE_INSTRUCTIONS` in [visual_retrieval.py](server/app/visual_retrieval.py).
- Retrieval: `retrieve_visual_evidence()` in the same file.
- Explanation: `INSTR_VISUAL` and `_answer_visual_question()` in [orchestrator.py](server/app/orchestrator.py).
- Model selection: [providers.py](server/app/providers.py).

To enable optional hybrid retrieval and prepare its index (Git Bash, project root):

```bash
source server/.venv/Scripts/activate
export OPENAI_EMBEDDING_API_KEY="YOUR_OPENAI_API_KEY"
export VISUAL_RETRIEVAL_MODE="hybrid"
export PYTHONPATH="server"
python -m app.prepare_embeddings
python -m uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8000
```

`OPENAI_API_KEY` is also accepted when `OPENAI_EMBEDDING_API_KEY` is unset.
The chat's optional personal API key is not used for embeddings. In particular,
a Mistral key cannot pay for OpenAI embeddings. Embedding errors identify retrieval
separately from the selected answering provider.

Course vectors are cached under `server/app/data/openai_embeddings/`, separately
from any old E5 cache. Set `OPENAI_EMBEDDING_CACHE_DIR` to change the directory.
The cache is rebuilt when the course content or `OPENAI_EMBEDDING_MODEL` changes.
Restart the backend after editing the course. Query descriptions are batched
(up to 32 per request). In hybrid mode, after indexing, an image question normally makes two
vision/generation calls plus one embedding request; larger inventories and retries
can add requests. Initial indexing also makes embedding requests, billed by OpenAI.
No hosted vector store or local E5/PyTorch model is used.

Hybrid retrieval failures do not silently switch to BM25.
Retrieved candidates are not proof: the answering model must check them against
the image and cite supporting pages or acknowledge uncertainty.

Offline checks, from `server/`:

```bash
./.venv/Scripts/python.exe -m unittest test_embeddings test_visual_retrieval -v
```

API reference: https://developers.openai.com/api/docs/guides/embeddings

The frontend enables ChatKit attachments with a two-phase upload strategy.

Uploaded images are stored locally in:

```text
server/app/uploads/
```

Example:

```text
Upload a symbol image
Ask: What does this symbol mean?
Ask: What about the color?
Ask: What about the form?
Type: clear image
```

The same image is reused for follow-up visual questions until `clear image` is used.

## Learner Guide

The learner guide is available at:

```text
docs/learner_guide.md
```

When the backend is running:

```text
http://127.0.0.1:8000/docs/learner_guide.md
```

The chat also supports:

```text
help
aide
guide
```

## Benchmark

Run the automatic ITS benchmark:

```powershell
server\.venv\Scripts\python.exe server/app/benchmark_runner.py
```

JSON output:

```powershell
server\.venv\Scripts\python.exe server/app/benchmark_runner.py --json
```

The benchmark reads:

```text
server/app/evidence_log.jsonl
```

It checks:

- diagnostic screening role
- diagnostic mastery cap
- adaptive practice length
- essential target coverage
- cumulative practice
- tutor decision consistency
- module checkpoint policy
- lesson and practice grounding

## Evidence Logs

New learning and feedback events include `user_id` and `username`. Existing older
events are not rewritten. Conversations are saved under `threads/<user_id>.json`
and progress under `sessions/<user_id>.json` in the configured state backend.
Events are also written to that backend under `events/`.

The system records tutoring events in:

```text
server/app/evidence_log.jsonl
```

Examples of logged events:

- `diagnostic_started`
- `diagnostic_submitted`
- `micro_lesson_generated`
- `practice_started`
- `practice_submitted`
- `learner_question_answered`
- `visual_question_answered`
- `module_checkpoint_started`
- `module_checkpoint_submitted`

## Files excluded from Git

The repository's `.gitignore` excludes `.env` files, local accounts and learner
state in `server/app/data/`, `server/app/evidence_log.jsonl`, uploaded images,
generated flashcard images, virtual environments and build output. Keep API keys
in environment variables. Custom storage directories need their own ignore rules.

For the application-only push, also exclude `server/simulation/`, `server/ecg/`
and `server/benchmark_pilot/` when staging. Those three exclusions are a staging
choice, not a claim that all three folders are covered by `.gitignore`.

## Research Framing

PDF2ITS can be described as:

```text
A PDF-grounded LLM Intelligent Tutoring System that converts instructional material into KC-based adaptive tutoring workflows.
```

The central contribution is the orchestration layer that connects:

```text
PDF grounding
KC graph navigation
learner-state tracking
tutor decision policy
adaptive lessons
adaptive practice
feedback and remediation
mastery tracking
```

