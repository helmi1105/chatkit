# app/widgets/plotlywidget.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from chatkit.widgets import WidgetTemplate, DynamicWidgetRoot

_TEMPLATE_PATH = Path(__file__).with_name("plotly.widget")
_plotly_template = WidgetTemplate.from_file(str(_TEMPLATE_PATH))


def _clean_plot_html(raw_html: str) -> str:
    """Strip MCP log headers and return only the HTML fragment."""
    if not raw_html:
        return ""

    marker = "HTML Content:"
    if marker in raw_html:
        raw_html = raw_html.split(marker, 1)[1]

    raw_html = raw_html.strip()

    for tag in ("<html", "<!DOCTYPE", "<div", "<iframe"):
        idx = raw_html.find(tag)
        if idx != -1:
            return raw_html[idx:]

    return raw_html


def build_plotly_widget_from_data(data: Dict[str, Any]) -> DynamicWidgetRoot:
    """
    Expects at least:
    {
      "html": "<full plotly HTML ...>",
      "title": "Trafic boutique"
    }

    We normalize into the schema required by the widget:
      - title
      - subtitle
      - buttonLabel
      - icon
      - action.type
      - action.payload.html
      - action.payload.title
    """

    html = _clean_plot_html(str(data.get("html") or ""))
    print("htmlhtml::",html)
    base_title = data.get("title") or "Visualisation Plotly"

    normalized: Dict[str, Any] = {
        "title": base_title,
        "subtitle": data.get(
            "subtitle",
            "Clique sur le bouton pour afficher le graphique interactif.",
        ),
        "buttonLabel": data.get("buttonLabel", "Afficher le graphique"),
        "icon": data.get("icon", "chart"),  # must be "analytics" or "chart"
        "action": {
            "type": "report.open",
            "handler": data.get("handler", "client"),
            "payload": {
                "html": html,
                "title": base_title,
            },
        },
    }

    return _plotly_template.build(normalized)
