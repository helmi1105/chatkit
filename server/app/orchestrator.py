# orchestrator.py

from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Dict, List

# OpenAI Agents SDK
from agents import Agent, Runner, FileSearchTool, ModelSettings, StopAtTools
from openai.types.shared.reasoning import Reasoning
# ChatKit context type
from chatkit.agents import AgentContext
from contextlib import AsyncExitStack
from agents.mcp import MCPServerStreamableHttp


# =====================================================
# CONFIG
# =====================================================

VECTOR_STORE_ID = "vs_692857ec28d8819194c4952fcb7f4687"
# 🔐 Agents SDK uses OPENAI_API_KEY from env, so no need for OpenAI() client here.


# =====================================================
# ROUTER AGENT (Agent[AgentContext] – intent classifier)
# =====================================================

class RouterAgent:
    def __init__(self) -> None:
        self.agent: Agent[AgentContext] = Agent[AgentContext](
            name="RouterAgent",
            model="gpt-4.1",
           instructions=(
            "You are an intent classifier for a firefighter assistant.\n\n"
            "Return EXACTLY one of the following tokens (no extra text):\n"
            "- DOCTRINE_QUERY\n"
            "- QUIZ_REQUEST\n"
            "- STUDY_CARD_REQUEST\n"
            "- SCHEMA_REQUEST\n"
            "- MAP_REQUEST\n"
            "- PLOTLY_REQUEST\n"
            "- EVALUATION_REQUEST\n\n"
            "EVALUATION_REQUEST is for answers like: 1B 2C 3C ...\n"
            "Use STUDY_CARD_REQUEST whenever the user asks for a course, lesson plan, "
            "or study card to be generated.\n"
            "Use SCHEMA_REQUEST for requests about incident schematics, operational layouts, "
            "or drawing tactical plans.\n"
            "Use MAP_REQUEST whenever the user asks for a map, localisation, GPS point, "
            "OpenStreetMap link, coordinates, or wants to see a map preview.\n"
            "Use PLOTLY_REQUEST when the user asks to build, draw, or update a graph/chart/plot "
            "using data or wants a visual representation (bar chart, line chart, etc.)."
        ),
            model_settings=ModelSettings(
                store=True,
                # reasoning=Reasoning(effort="medium")
            )
        )


    async def classify(self, text: str, ctx: AgentContext) -> str:
        """Classify intent: doctrine, quiz, evaluation."""
        print("[RouterAgent] text:", text)
        result = await Runner.run(self.agent, text, context=ctx)
        intent = (result.final_output or "").strip()
        return intent


# =====================================================
# DOCTRINE AGENT (Agent[AgentContext] + FileSearchTool)
# =====================================================



class PlotlyMCPAgent:
    """
    Agent that uses the PredictOps Plotly MCP server to build visualizations.
    """

    PLOTLY_INSTRUCTIONS = """
Tu es un assistant de visualisation connecté à un serveur MCP « plotly ».
Tu peux appeler les outils exposés par ce serveur pour générer des graphiques Plotly
(2D, 3D, cartes, finance, hiérarchies, etc.).

## Familles d’outils
- **Traces 2D de base** : `create_scatter_plot`, `create_bar_chart`, `create_line_chart`,
  `create_pie_chart`, `create_histogram`.
- **Traces statistiques** : `create_box_plot`, `create_violin_plot`, `create_heatmap_chart`,
  `create_contour_chart`, `create_splom_chart`, `create_parallel_coordinates`,
  `create_parallel_categories`, `create_histogram2d_chart`.
- **3D & volumes** : `create_scatter3d_plot`, `create_surface_plot`, `create_mesh3d_plot`,
  `create_volume_plot`, `create_isosurface_plot`, `create_cone_plot`, `create_streamtube_plot`.
- **Cartographie** : `create_choropleth_map`, `create_choroplethmap`,
  `create_choropleth_mapbox`, `create_scatter_geo_plot`, `create_scatter_map`,
  `create_scatter_mapbox_plot`, `create_density_map`, `create_density_mapbox`.
- **Finance et hiérarchies** : `create_candlestick_chart`, `create_ohlc_chart`,
  `create_waterfall_chart`, `create_treemap_chart`, `create_sunburst_chart`,
  `create_icicle_chart`, `create_sankey_diagram`.
- **Utilitaires** : `generate_sample_data`, `create_multi_trace_figure`, `server_status`, `stop_server`.

TON COMPORTEMENT :

1. À partir de la question de l’utilisateur, choisis toi-même l’outil le plus adapté
   parmi ceux ci-dessus et appelle-le via le serveur MCP.

2. Tu laisses le serveur MCP décider du contenu de la réponse :
   logs, chemins de fichiers, « HTML Content: <html>...</html> », etc.

3. **Tu ne modifies RIEN** dans la sortie de l’outil :
   
4. Ta réponse finale à l’utilisateur doit être **strictement** la sortie brute renvoyée
   par l’outil MCP (comme dans l’intercepteur MCP), caractère pour caractère.
"""


    def __init__(self) -> None:
        # Async context stack to manage the MCP server lifetime
        self._stack: AsyncExitStack | None = None
        self._agent: Agent[AgentContext] | None = None
        self._initialized: bool = False

    async def _ensure_agent(self) -> None:
        if self._initialized:
            return

        if self._stack is None:
            self._stack = AsyncExitStack()

        # Create ONE MCP server instance pointing to PredictOps Plotly MCP
        mcp_server = await self._stack.enter_async_context(
            MCPServerStreamableHttp(
                name="PredictOps Plotly MCP",
                params={
                    "url": "https://predictops-chat.duckdns.org/predictops-mcp/plotly/mcp",
                    "timeout": 30,
                },
                cache_tools_list=True,
                max_retry_attempts=4,
                client_session_timeout_seconds=60,
            )
        )

        # Create the LLM agent that uses this MCP server
        self._agent = Agent[AgentContext](
            name="PlotlyAgent",
            model="gpt-4.1",
            instructions=self.PLOTLY_INSTRUCTIONS,
            mcp_servers=[mcp_server],
            model_settings=ModelSettings(tool_choice="required"),
        )

        self._initialized = True

    async def build_chart(self, question: str, ctx: AgentContext) -> str:
        """
        Ask the Plotly MCP agent to build or explain a chart.
        """
        await self._ensure_agent()
        assert self._agent is not None

        result = await Runner.run(self._agent, question, context=ctx)
        print("result::",result)
        return (result.final_output or "").strip()

    async def aclose(self) -> None:
        """
        Optional: to be called when shutting down the app/server.
        """
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._initialized = False






class DoctrineAgent:
    async def answer(self, query: str, ctx: AgentContext) -> str:
        """
        Answer doctrine questions using the vector store via FileSearchTool.

        ctx: AgentContext from chatkit.agents
          - ctx.thread
          - ctx.store
          - ctx.request_context
        """

        # ------------------------------
        # 1) Logging / Context info
        # ------------------------------
        thread_id = getattr(ctx.thread, "id", "unknown-thread")
        print(f"[DoctrineAgent] thread={thread_id} | query={query!r}")

        # ------------------------------
        # 2) Dynamic vector store via context
        # ------------------------------
        vector_store_id = VECTOR_STORE_ID
        rc = getattr(ctx, "request_context", None)

        if isinstance(rc, dict):
            vector_store_id = rc.get("vector_store_id", VECTOR_STORE_ID)

        # Optional: adapt system prompt with profile / role
        system_prompt = (
            "You are a doctrine assistant for firefighters.\n"
            "Use ONLY the content provided by the doctrine files in the vector store.\n"
            "If doctrine data is missing, say so explicitly."
        )
        if isinstance(rc, dict) and rc.get("profile"):
            system_prompt += f"\nUser profile: {rc['profile']}."

        # ------------------------------
        # 3) Build a per-call Agent with FileSearchTool
        #    (vector_store_id can change per request)
        # ------------------------------
        file_search_tool = FileSearchTool(
            max_num_results=8,
            vector_store_ids=[vector_store_id],
        )

        doctrine_agent: Agent[AgentContext] = Agent[AgentContext](
            name="DoctrineAgent",
            model="gpt-4.1",
            instructions=system_prompt,
            tools=[file_search_tool],
        )

        # The Agents SDK will automatically:
        #  - decide if/when to call file_search
        #  - run the tool
        #  - feed results back to the model
        result = await Runner.run(doctrine_agent, query, context=ctx)

        return (result.final_output or "").strip()
    
class SchemaAgent:
    async def build_schema(self, query: str, ctx: AgentContext) -> str:
        """
        Build a fictitious operational schematic description
        using doctrinal conventions: zones, actions, moyens,
        commandement, flux, symboles, positions, sectorisation…

        ctx gives access to thread, store, request_context.
        """

        thread_id = getattr(ctx.thread, "id", "unknown-thread")
        print(f"[SchemaAgent] thread={thread_id} | query={query!r}")

        # Dynamic vector store if provided
        vector_store_id = VECTOR_STORE_ID
        rc = getattr(ctx, "request_context", None)

        if isinstance(rc, dict):
            vector_store_id = rc.get("vector_store_id", VECTOR_STORE_ID)

        # You control the style and conventions here
        system_prompt = (
            "You are an operational schematic assistant for firefighters.\n"
            "Your output is a structured schematic plan based on doctrine.\n"
            "You ALWAYS explain how to draw the schematic (paper or digital).\n"
            "Your answers MUST follow this structure:\n"
            "\n"
            "1) Contexte général du sinistre (lieu, risques, zones sensibles)\n"
            "2) Représentation graphique (formes, positions, symboles doctrinaux)\n"
            "3) Actions codifiées (R-x, A-x, D-x)\n"
            "4) Moyens engagés (dessinable, positions)\n"
            "5) Organisation opérationnelle (PC, circulation, secteurs)\n"
            "\n"
            "The answer must be procedural: step-by-step instructions to reproduce the scheme.\n"
            "If doctrine content is missing from the vector store, state it clearly."
        )

        # Optional personalization
        if isinstance(rc, dict) and rc.get("profile"):
            system_prompt += f"\nUser profile: {rc['profile']}."

        # File search tool (same logic as doctrine agent)
        file_search_tool = FileSearchTool(
            max_num_results=8,
            vector_store_ids=[vector_store_id],
        )

        schema_agent: Agent[AgentContext] = Agent[AgentContext](
            name="SchemaAgent",
            model="gpt-4.1",
            instructions=system_prompt,
            tools=[file_search_tool],
        )

        result = await Runner.run(schema_agent, query, context=ctx)

        return (result.final_output or "").strip()


# =====================================================
# STUDY AGENT (Agent[AgentContext] + FileSearchTool)
# =====================================================

class StudyAgent:
    """
    Generates structured study card payloads that match the study.widget schema.
    """

    STUDY_SCHEMA_EXAMPLE = (
        '{"payload":{"title":"string","sections":[{"label":"string","content":"string",'
        '"bullets":["string"],"questions":[{"id":"string","question":"string","choices":'
        '["string","string","string","string"],"correct":"string"}]}],"next_steps":["string"]}}'
    )

    def __init__(self) -> None:
        self.instructions = (
            "You are a firefighter doctrine course designer. Build concise study cards "
            "strictly from the doctrine materials available via the file_search tool.\n"
            "Output MUST be valid JSON (no markdown, no prose) that matches this schema:\n"
            f"{self.STUDY_SCHEMA_EXAMPLE}\n"
            "- Always populate payload.title with an informative course name.\n"
            "- Provide 3-4 sections when possible; each section must have a label and at "
            "least one of: content, bullets, or questions.\n"
            "- When adding questions, include 2-4 multiple-choice options and specify the "
            "correct answer text in the 'correct' field.\n"
            "- next_steps should contain actionable follow-ups or drills.\n"
            "If doctrine content is missing, explain that limitation in the content fields "
            "but still return valid JSON."
        )

    async def create_course(self, request_text: Any, ctx: AgentContext) -> Dict[str, Any]:
        # Normalize request_text safely, supporting list, dict, and chatkit-style messages
        topic_text = ""

        # Case 1: If request_text is already a simple string
        if isinstance(request_text, str):
            topic_text = request_text.strip()

        # Case 2: If request_text is list (ChatKit message list)
        elif isinstance(request_text, list):
            for msg in reversed(request_text):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                blocks = msg.get("content", [])
                if not isinstance(blocks, list):
                    continue
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "input_text":
                        extracted = (block.get("text") or "").strip()
                        if extracted:
                            topic_text = extracted
                            break
                if topic_text:
                    break

        # Case 3: Fallback for weird types (numbers, dict, None, etc.)
        else:
            topic_text = str(request_text).strip()

        # Ensure topic is non-empty
        topic = topic_text or "Firefighter doctrine overview"

        # -------------------------
        # Continue original logic
        # -------------------------
        rc = getattr(ctx, "request_context", None)
        vector_store_id = VECTOR_STORE_ID
        if isinstance(rc, dict):
            vector_store_id = rc.get("vector_store_id", VECTOR_STORE_ID)

        file_search_tool = FileSearchTool(
            max_num_results=8,
            vector_store_ids=[vector_store_id],
        )

        agent = Agent[AgentContext](
            name="StudyAgent",
            model="gpt-4.1",
            instructions=self.instructions,
            tools=[file_search_tool],
        )

        user_prompt = f"""
    Build a structured firefighter study card for the following learner request.
    Focus strictly on verified doctrine from the files and include actionable teaching points.

    Learner request:
    \"\"\"{topic}\"\"\"\n
    Return ONLY the JSON payload that matches the schema described in your instructions.
    """

        result = await Runner.run(agent, user_prompt, context=ctx)
        raw = (result.final_output or "").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print("[StudyAgent] JSON decode error:", exc, raw)
            data = {}

        normalized = normalize_study_widget_payload(data if isinstance(data, dict) else {})
        return normalized



# =====================================================
# QCM AGENT (Agent[AgentContext] + FileSearchTool)
# =====================================================

@dataclass
class QcmQuestion:
    number: int
    text: str
    choices: List[str]


@dataclass
class QcmPayload:
    visible: List[QcmQuestion]
    hidden: Dict[int, str]


class QcmAgent:
    def __init__(self) -> None:
        file_search_tool = FileSearchTool(
            max_num_results=5,
            vector_store_ids=[VECTOR_STORE_ID],
        )

        self.agent: Agent[AgentContext] = Agent[AgentContext](
            name="QcmAgent",
            model="gpt-4.1",
            instructions=(
                "You build multiple-choice questions (QCM) strictly from doctrine files.\n"
                "Always return ONLY the JSON list as specified in the prompt (no extra text)."
            ),
            tools=[file_search_tool],
        )

    async def generate(self, ctx: AgentContext) -> QcmPayload:
        prompt = """
Generate 3 QCM questions about firefighter doctrine.
Each includes:
- number (1..3)
- text
- choices [A,B,C,D]
- answer (A/B/C/D)

Return ONLY JSON, no explanation, exactly in this format:
[
  {"number":1,"text":"...","choices":["A..","B..","C..","D.."],"answer":"B"},
  {"number":2,"text":"...","choices":["A..","B..","C..","D.."],"answer":"A"},
  {"number":3,"text":"...","choices":["A..","B..","C..","D.."],"answer":"D"}
]
"""
        # 👇 IMPORTANT: use result.final_output, not result directly
        result = await Runner.run(self.agent, prompt, context=ctx)
        raw = (result.final_output or "").strip()
        data = json.loads(raw)

        visible: List[QcmQuestion] = []
        hidden: Dict[int, str] = {}

        for q in data:
            visible.append(
                QcmQuestion(
                    number=q["number"],
                    text=q["text"],
                    choices=q["choices"],
                )
            )
            hidden[q["number"]] = q["answer"]

        return QcmPayload(visible=visible, hidden=hidden)


# =====================================================
# EVALUATOR AGENT (pure Python, no LLM)
# =====================================================

class EvaluatorAgent:
    def evaluate(self, user: Dict[int, str], correct: Dict[int, str]):
        score = 0
        details: List[str] = []

        for num, ans in correct.items():
            user_ans = user.get(num, "").upper()
            if user_ans == ans:
                score += 1
                details.append(f"Q{num}: ✓ Correct ({user_ans})")
            else:
                details.append(f"Q{num}: ✗ Wrong (Your: {user_ans}, Correct: {ans})")

        score20 = round((score / len(correct)) * 20, 2)
        return score, score20, details


# =====================================================
# ORCHESTRATOR (async – all Agents via Runner)
# =====================================================


def qcm_payload_to_widget_data(qcm: QcmPayload) -> Dict[str, Any]:
    """
    Convert QcmPayload → JSON-serializable dict that matches your schema:

    {
      "title": "string",
      "questions": [
        {
          "id": "1",
          "prompt": "...",
          "choices": [
            {"label": "A) ...", "value": "A"},
            {"label": "B) ...", "value": "B"},
            {"label": "C) ...", "value": "C"},
            {"label": "D) ...", "value": "D"}
          ]
        },
        ...
      ]
    }
    """
    questions: list[dict[str, Any]] = []

    for q in qcm.visible:
        questions.append(
            {
                "id": str(q.number),
                "prompt": q.text,
                "choices": [
                    {"label": f"A) {q.choices[0]}", "value": "A"},
                    {"label": f"B) {q.choices[1]}", "value": "B"},
                    {"label": f"C) {q.choices[2]}", "value": "C"},
                    {"label": f"D) {q.choices[3]}", "value": "D"},
                ],
            }
        )

    return {
        "title": "QCM doctrine (3 questions)",  # or dynamic
        "questions": questions,
    }


def _ensure_text(value: Any) -> str:
    """
    Convert arbitrary values into trimmed strings.
    """
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return ""


def _ensure_string_list(value: Any) -> list[str]:
    """
    Normalize a list-like structure into a list of non-empty strings.
    Supports plain strings, dicts with label/text/value, or nested lists.
    """
    result: list[str] = []

    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = [value]
    elif isinstance(value, str):
        items = [part.strip() for part in value.split("\n") if part.strip()]
    else:
        items = []

    for item in items:
        if isinstance(item, dict):
            text = (
                _ensure_text(item.get("label"))
                or _ensure_text(item.get("value"))
                or _ensure_text(item.get("text"))
            )
        else:
            text = _ensure_text(item)
        if text:
            result.append(text)

    return result


def normalize_study_widget_payload(data: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort normalization to ensure the Study widget payload always matches
    the schema expected by study.widget. Fills in sensible defaults when data
    is missing or malformed.
    """
    payload_source: dict[str, Any] = {}
    if isinstance(data, dict):
        if isinstance(data.get("payload"), dict):
            payload_source = data["payload"]
        else:
            payload_source = data

    title = _ensure_text(payload_source.get("title")) or "Doctrine Study Card"

    sections_input = payload_source.get("sections")
    normalized_sections: list[dict[str, Any]] = []

    if isinstance(sections_input, list):
        for idx, raw_sec in enumerate(sections_input):
            if not isinstance(raw_sec, dict):
                continue
            label = _ensure_text(raw_sec.get("label")) or f"Section {idx + 1}"
            section: dict[str, Any] = {"label": label}

            content = _ensure_text(raw_sec.get("content"))
            section["content"] = content

            bullets = _ensure_string_list(raw_sec.get("bullets"))
            section["bullets"] = bullets

            questions_out: list[dict[str, Any]] = []
            raw_questions = raw_sec.get("questions")
            if isinstance(raw_questions, list):
                for q_idx, raw_q in enumerate(raw_questions):
                    if not isinstance(raw_q, dict):
                        continue
                    question_text = _ensure_text(raw_q.get("question"))
                    choices = _ensure_string_list(raw_q.get("choices"))
                    if not question_text or not choices:
                        continue
                    question = {
                        "id": _ensure_text(raw_q.get("id")) or f"{idx + 1}-{q_idx + 1}",
                        "question": question_text,
                        "choices": choices[:4],
                        "correct": _ensure_text(raw_q.get("correct")) or choices[0],
                    }
                    questions_out.append(question)

            section["questions"] = questions_out

            if not section["content"]:
                section["content"] = "Content unavailable in doctrine files."

            normalized_sections.append(section)

    if not normalized_sections:
        normalized_sections.append(
            {
                "label": "Explanation",
                "content": "Doctrine content not available. Provide official references to build this lesson.",
                "bullets": [],
                "questions": [],
            }
        )

    next_steps = _ensure_string_list(payload_source.get("next_steps"))
    if not next_steps:
        next_steps = [
            "Share official doctrine passages for deeper study.",
            "Outline scenarios or drills to apply this lesson.",
        ]

    normalized_payload = {
        "title": title,
        "sections": normalized_sections,
        "next_steps": next_steps,
    }
    return {"payload": normalized_payload}

import math
from typing import Any, Dict

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def latlng_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return x_tile, y_tile


def build_osm_tile_map_payload(
    lat: float,
    lng: float,
    zoom: int = 1,
) -> Dict[str, Any]:
    x_tile, y_tile = latlng_to_tile(lat, lng, zoom)

    src = OSM_TILE_URL.format(z=zoom, x=x_tile, y=y_tile)
    alt = f"OpenStreetMap tile at lat={lat:.4f}, lon={lng:.4f}, z={zoom}"

    return {
        "src": src,
        "alt": alt,
        "lat": lat,
        "lng": lng,
        "zoom": zoom,
    }




import re

def extract_html(raw: str) -> str:
    # Try to isolate the <html>...</html> block
    m = re.search(r"(<html[\\s\\S]*</html>)", raw)
    return m.group(1).strip() if m else raw.strip()




class Orchestrator:
    def __init__(self):
        self.router = RouterAgent()
        self.doctrine = DoctrineAgent()
        self.schema_agent = SchemaAgent()
        self.study_agent = StudyAgent()
        self.qcm_agent = QcmAgent()
        self.evaluator = EvaluatorAgent()
        self.plotly_agent = PlotlyMCPAgent() 
        # NOTE: single global store; for multi-thread safety, key this by thread_id
        self.hidden_answers: Dict[int, str] | None = None
        self.last_qcm_payload: QcmPayload | None = None
        self.last_qcm_widget_data: dict | None = None

    async def handle(self, user_input: Any, ctx: AgentContext) -> Any:
        # RouterAgent.classify is now async (uses Runner)
        latest_text = user_input

        intent = await self.router.classify(latest_text, ctx)

        if intent == "DOCTRINE_QUERY":
            return await self.doctrine.answer(latest_text, ctx)

        elif intent == "QUIZ_REQUEST":
            qcm = await self.qcm_agent.generate(ctx)   # ✅ returns QcmPayload
            self.hidden_answers = qcm.hidden

            # Build widget-compatible data
            qcm_data = qcm_payload_to_widget_data(qcm)

            # Server will see this and render a widget
            return {"type": "qcm", "data": qcm_data}

        elif intent == "STUDY_CARD_REQUEST":
            study_data = await self.study_agent.create_course(latest_text, ctx)
            return {"type": "study", "data": study_data}

        elif intent == "SCHEMA_REQUEST":
            return await self.schema_agent.build_schema(latest_text, ctx)
        
        elif intent == "MAP_REQUEST":
            # For now: return fixed coordinates (London-like example).
            # Later you can parse coordinates from the user message.
            payload = build_osm_tile_map_payload(
                lat=51.5,
                lng=-0.09,
                zoom=13,
            )
            return {"type": "map", "data": payload}
        
        elif intent == "PLOTLY_REQUEST":
            text = self.extract_latest_user_text(user_input)
            raw = await self.plotly_agent.build_chart(text, ctx)  # this returns the MCP string you pasted
            html = extract_html(raw)

            # Send a widget payload to ChatKit
            return {
                "type": "plotly",          # your custom widget type
                "data": {
                    "html": html,
                    "title": "Trafic boutique",         # full <html>...</html> content
                },
            }

        
        elif intent == "EVALUATION_REQUEST":
            if not self.hidden_answers:
                return "⚠️ Please generate a QCM first."

            answers = self.parse_answers(user_input)
            score, score20, details = self.evaluator.evaluate(
                answers, self.hidden_answers
            )
            return self.format_evaluation(score, score20, details)

        else:
            return "Unknown request."

    # -----------------------------------------------------

    def extract_latest_user_text(self, text: Any) -> str:
        """
        Extract the most recent user-authored text block from ChatKit-style items
        or fall back to a plain string.
        """
        if isinstance(text, list):
            for msg in reversed(text):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                for block in msg.get("content", []):
                    if block.get("type") == "input_text":
                        candidate = (block.get("text") or "").strip()
                        if candidate:
                            return candidate
            return ""
        return str(text or "").strip()

    # -----------------------------------------------------

    def parse_answers(self, text: Any) -> Dict[int, str]:
        """
        Accepts either:
        - a plain string: "1A 2C 3B"
        - or a list of ChatKit/Agent-style message dicts, like the debug print

        Returns: {1: "A", 2: "C", 3: "B"}
        """
        # ---------------------------------------
        # 1) Normalize to text
        # ---------------------------------------
        text_str = self.extract_latest_user_text(text)

        print("answers (clean) :::", text_str)

        # ---------------------------------------
        # 2) Your original parsing logic
        # ---------------------------------------
        out: Dict[int, str] = {}
        text_str = text_str.replace("answers:", "").replace("réponses:", "")
        for token in text_str.split():
            if len(token) >= 2 and token[0].isdigit():
                out[int(token[0])] = token[1].upper()
        return out


    # -----------------------------------------------------
    def format_qcm(self, qcm: QcmPayload) -> str:
        lines = ["🔥 QCM (3 questions)\n"]
        for q in qcm.visible:
            lines.append(f"Q{q.number}. {q.text}")
            lines.append(f"A) {q.choices[0]}")
            lines.append(f"B) {q.choices[1]}")
            lines.append(f"C) {q.choices[2]}")
            lines.append(f"D) {q.choices[3]}\n")
        lines.append("➡️ Reply like: 1A 2C 3B")
        return "\n".join(lines)

    # -----------------------------------------------------
    def format_evaluation(self, score, score20, details):
        return f"""📊 Evaluation

Score: {score}/3
Score /20: {score20}

Details:
{chr(10).join(details)}
"""
