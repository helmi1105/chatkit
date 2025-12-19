# OpenAI ChatKit Demo: Advance Self-hosting Integration

A demo of integrating ChatKit by hosting a ChatKit server in our own infrastructure.




## What's Inside

- **Custom FastAPI ChatKit host** – `server/app/main.py` exposes `/chatkit`, injects the per-user context (`userId` header), and streams responses back to the browser.
- **Multi-agent orchestrator** – `server/app/orchestrator.py` routes every prompt to specialized agents (doctrine QA, course builder, schematic designer, quiz generator, evaluator, Plotly MCP charts, OSM map helper) and builds normalized payloads for the UI.
- **Widget toolkit** – files under `server/app/widgets/` (e.g. `qcmwidget.py`, `studywidget.py`, `mapwidget.py`, `plotlywidget.py`) turn orchestrator payloads into ChatKit widgets so quizzes, study cards, maps, and charts render natively in the chat.
- **Persistent thread store stub** – `server/app/data_store.py` keeps an in-memory thread+message history keyed by the UI’s stable `userId`. Swap this out for your database when deploying.
- **Next.js chat surface** – `web/src/app/ChatKitComponent.tsx` embeds ChatKit, injects the `userId`, and renders a dual-pane experience where the right pane shows inline maps or Plotly charts triggered by widget actions.


## How It Works

1. The frontend stores a random `userId` in `localStorage`, mounts `ChatKitComponent`, and sends every request to `http://127.0.0.1:8000/chatkit` with that header.
2. FastAPI (`server/app/main.py`) forwards the raw payload to `MyChatKitServer`, which loads the recent thread items from `MyDataStore`.
3. `MyChatKitServer.respond` converts the history into agent input, calls the orchestrator, and streams back either plain assistant messages or widget payloads (QCM, study, map, Plotly). Widget actions (quiz submission, map preview, chart open) are also handled.
4. On the browser, `ChatKitComponent` listens for widget actions. Map actions open an inline OpenStreetMap iframe beside the chat; Plotly actions inject responsive HTML into a sandboxed iframe so charts render safely. Quiz submissions call back to the server, which grades answers via `EvaluatorAgent`.


## Using the Demo

- **Doctrine Q&A** – ask operational questions ("Quels sont les niveaux d'alerte ?") to get grounded answers sourced from the vector store configured in `VECTOR_STORE_ID`.
- **Study cards** – ask for a "fiche" or "course" to receive a study widget with sections, bullets, and self-check questions.
- **Quizzes** – request a quiz; the assistant sends a QCM widget. Select answers and hit “Submit” to receive a scored evaluation.
- **Maps** – mention coordinates/localisation; the assistant returns a map widget. Click “Afficher la carte” to preview the tile inline.
- **Charts** – ask for a chart/graph; the Plotly MCP agent generates HTML. Click the chart widget in the chat to display the interactive Plotly output in the right pane.


## Run the App

### Backend API Server

1. Install dependencies
```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-proj-...
```

2. start the server
```bash
uvicorn app.main:app --reload --port 8000
```


### Frontend App
1. Install dependencies

```bash
cd web
npm install
```

2. start the app
```bash
npm run dev
```

3. Go to `http://localhost:3000` and start chatting


## Deployment Note
In the frontend when configuring `useChatKit`,  we have a `domainKey` used by the ChatKit for verifying the registered domain for the integration.

For local testing, this can be any placeholder. However, for production, please make sure to register your domain in https://platform.openai.com/settings/organization/security/domain-allowlist and replace it with the real key.
