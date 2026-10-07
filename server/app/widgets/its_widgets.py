# app/widgets/its_widgets.py
"""ChatKit widgets built in Python (no Jinja template): action buttons,
answer sheet (corrigé), lesson source card, progress summary, confirmation.

Buttons carry a SERVER action `its.command` whose payload is the same French
command a learner could type ("quiz", "indice", "notion suivante"...). The
server handler dispatches it to the orchestrator exactly like a typed
message, so buttons and text stay two faces of one behaviour."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from chatkit.actions import ActionConfig
from chatkit.widgets import (
    Badge,
    Button,
    Caption,
    Card,
    Col,
    Divider,
    Image,
    Row,
    Spacer,
    Text,
    Title,
)

ButtonSpec = Tuple[str, str]  # (label, command)


def command_action(command: str, **extra: Any) -> ActionConfig:
    payload: Dict[str, Any] = {"command": command}
    payload.update(extra)
    return ActionConfig(type="its.command", payload=payload, handler="server")


def _button(label: str, command: str, *, primary: bool = False) -> Button:
    return Button(
        label=label.replace("é", "e").replace("É", "E").replace("ç", "c").replace("Ç", "C"),
        onClickAction=command_action(command),
        style="primary" if primary else "secondary",
        size="sm",
        pill=True,
    )


def actions_card(buttons: Sequence[ButtonSpec], *, title: Optional[str] = None, caption: Optional[str] = None) -> Card:
    """A row of pill buttons. The first one is rendered as primary."""
    children: List[Any] = []
    if title:
        children.append(Title(value=title, size="sm"))
    if caption:
        children.append(Caption(value=caption, color="secondary"))
    row = Row(
        gap=2,
        wrap="wrap",
        children=[_button(label, cmd, primary=(i == 0)) for i, (label, cmd) in enumerate(buttons)],
    )
    children.append(row)
    return Card(size="md", padding=3, children=[Col(gap=2, children=children)])


def confirm_card(question: str, yes: ButtonSpec, no: ButtonSpec) -> Card:
    return Card(
        size="md",
        padding=3,
        children=[
            Col(
                gap=2,
                children=[
                    Text(value=question, size="sm", weight="semibold"),
                    Row(gap=2, children=[_button(yes[0], yes[1], primary=True), _button(no[0], no[1])]),
                ],
            )
        ],
    )


def correction_card(
    title: str,
    items: List[Dict[str, Any]],
    *,
    score_label: str,
    passed: Optional[bool],
    quiz_id: str,
    kc_id: str,
) -> Card:
    """One block per question: statement, learner answer vs expected answer,
    a one-line justification with the page, and a « Signaler » ghost button
    that lets a trainer flag a bad question (server action qcm.report)."""
    children: List[Any] = [
        Row(
            align="center",
            gap=2,
            children=[
                Title(value=title, size="sm"),
                Spacer(),
                Badge(
                    label=score_label,
                    color="success" if passed else ("danger" if passed is False else "info"),
                    variant="soft",
                ),
            ],
        )
    ]
    for it in items:
        ok = bool(it.get("correct"))
        learner = it.get("learner_letter") or "-"
        correct_letter = it.get("correct_letter") or "?"
        learner_txt = it.get("learner_choice") or ""
        correct_txt = it.get("correct_choice") or ""
        block: List[Any] = [
            Row(
                align="start",
                gap=2,
                children=[
                    Badge(label="Juste" if ok else "Faux", color="success" if ok else "danger", variant="solid", size="sm"),
                    Text(value=f"Q{it.get('number')}. {it.get('question', '')}", size="sm", weight="semibold"),
                ],
            ),
        ]
        if ok:
            block.append(Text(value=f"Votre réponse : {learner} · {learner_txt}", size="sm", color="secondary"))
        else:
            block.append(Text(value=f"Votre réponse : {learner} · {learner_txt}", size="sm", color="danger"))
            block.append(Text(value=f"Bonne réponse : {correct_letter} · {correct_txt}", size="sm", color="success"))
        expl = str(it.get("explanation") or "").strip()
        page = it.get("page")
        if expl:
            suffix = f" (page {page})" if page and f"page {page}" not in expl.lower() and f"p.{page}" not in expl.lower() else ""
            block.append(Text(value=f"Pourquoi : {expl}{suffix}", size="sm", color="secondary"))
        elif page:
            block.append(Caption(value=f"Source : page {page}", color="secondary"))
        block.append(
            Row(
                justify="end",
                children=[
                    Button(
                        label="Signaler cette question",
                        variant="ghost",
                        size="xs",
                        onClickAction=ActionConfig(
                            type="qcm.report",
                            payload={"quiz_id": quiz_id, "question_id": it.get("id"), "number": it.get("number"), "kc_id": kc_id},
                            handler="server",
                        ),
                    )
                ],
            )
        )
        block.append(Divider())
        children.append(Col(gap=1, children=block))
    return Card(size="md", padding=3, children=[Col(gap=2, children=children)])


def source_card(
    *,
    kc_title: str,
    pages: List[int],
    image_url: Optional[str],
    pdf_url: Optional[str],
    buttons: Sequence[ButtonSpec],
) -> Card:
    """Page image of the memento next to the lesson, a link to the PDF page
    and the next-step buttons."""
    pages_label = ", ".join(f"p. {p}" for p in pages) if pages else ""
    children: List[Any] = [
        Row(
            align="center",
            gap=2,
            children=[
                Col(
                    gap=0,
                    flex="auto",
                    children=[
                        Title(value=f"Source : {kc_title}", size="sm"),
                        Caption(value=f"Mémento GOC, {pages_label}" if pages_label else "Mémento GOC", color="secondary"),
                    ],
                ),
            ],
        )
    ]
    if image_url:
        children.append(Image(src=image_url, alt=f"Page {pages[0] if pages else ''} du mémento", fit="contain", radius="md", frame=True))
    row_buttons: List[Any] = [_button(label, cmd, primary=(i == 0)) for i, (label, cmd) in enumerate(buttons)]
    if pdf_url:
        row_buttons.append(
            Button(
                label="Ouvrir la page du PDF",
                variant="outline",
                size="sm",
                pill=True,
                onClickAction=ActionConfig(
                    type="report.open",
                    payload={"url": pdf_url, "html": "", "title": kc_title},
                    handler="client",
                ),
            )
        )
    children.append(Row(gap=2, wrap="wrap", children=row_buttons))
    return Card(size="md", padding=3, children=[Col(gap=2, children=children)])


def progress_card(
    *,
    title: str,
    modules: List[Dict[str, Any]],
    validated: int,
    total: int,
    radar_html: Optional[str],
    buttons: Sequence[ButtonSpec],
) -> Card:
    """Per-chapter validated / estimated counts, with the radar opened in the
    side pane on demand (client action report.open)."""
    children: List[Any] = [
        Row(
            align="center",
            gap=2,
            children=[
                Title(value=title, size="sm"),
                Spacer(),
                Badge(label=f"{validated}/{total} notions validées", color="info", variant="soft"),
            ],
        )
    ]
    for m in modules:
        n = int(m.get("total") or 0)
        v = int(m.get("validated") or 0)
        e = int(m.get("estimated") or 0)
        cur = " · en cours" if m.get("current") else ""
        label = f"{v}/{n} validées" + (f", {e} estimées au diagnostic" if e else "") + cur
        color = "success" if n and v == n else ("info" if v or e else "secondary")
        children.append(
            Row(
                align="center",
                gap=2,
                children=[
                    Col(flex="auto", children=[Text(value=str(m.get("title") or ""), size="sm", maxLines=2)]),
                    Badge(label=label, color=color, variant="soft", size="sm"),
                ],
            )
        )
    row_buttons: List[Any] = [_button(label, cmd, primary=(i == 0)) for i, (label, cmd) in enumerate(buttons)]
    if radar_html:
        row_buttons.append(
            Button(
                label="Voir le radar",
                variant="outline",
                size="sm",
                pill=True,
                onClickAction=ActionConfig(type="report.open", payload={"html": radar_html, "title": "Progression"}, handler="client"),
            )
        )
    children.append(Row(gap=2, wrap="wrap", children=row_buttons))
    return Card(size="md", padding=3, children=[Col(gap=2, children=children)])
