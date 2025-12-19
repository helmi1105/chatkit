# app/widgets/mapwidget.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from chatkit.widgets import WidgetTemplate, DynamicWidgetRoot

_TEMPLATE_PATH = Path(__file__).with_name("map.widget")
_map_template = WidgetTemplate.from_file(str(_TEMPLATE_PATH))


def build_map_widget_from_data(data: Dict[str, Any]) -> DynamicWidgetRoot:
    """
    Expects data like:
    {
      "title": "Localisation opérationnelle",
      "subtitle": "Cliquer pour afficher la carte à côté du chat",
      "lat": 45.7,
      "lng": 4.8,
      "zoom": 14,
      "src": "https://...",
      "alt": "...",
      "url": "https://..."
    }
    """
    return _map_template.build(data)
