# app/widgets/qcmwidget.py

from __future__ import annotations
from pathlib import Path

from chatkit.widgets import WidgetRoot, WidgetTemplate

BASE_DIR = Path(__file__).resolve().parent

qcm_widget_template = WidgetTemplate.from_file(BASE_DIR / "qcm.widget")


def build_qcm_widget_from_data(data: dict) -> WidgetRoot:
    """
    data example:
    {
      "title": "QCM doctrine",
      "questions": [
        {"id": "1", "prompt": "...", "choices": [{"label": "...", "value": "A"}, ...]}
      ]
    }
    """
    return qcm_widget_template.build(data=data)
