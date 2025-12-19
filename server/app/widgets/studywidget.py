"""Helpers for building the Fire Doctrine Study Card widget."""

from __future__ import annotations

from pathlib import Path

from chatkit.widgets import WidgetRoot, WidgetTemplate

BASE_DIR = Path(__file__).resolve().parent

study_widget_template = WidgetTemplate.from_file(BASE_DIR / "study.widget")


def build_study_widget_from_data(data: dict) -> WidgetRoot:
    """
    Build the study widget from structured data.

    Expected data payload (matches study.widget schema):
    {
      "payload": {
        "title": "string",
        "sections": [
          {
            "label": "string",
            "content": "string | optional",
            "bullets": ["string", ...] | optional,
            "questions": [
              {
                "id": "string",
                "question": "string",
                "choices": ["string", ...],
                "correct": "string"
              },
              ...
            ] | optional
          },
          ...
        ],
        "next_steps": ["string", ...]
      }
    }
    """
    return study_widget_template.build(data=data)

