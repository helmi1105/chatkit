"""KC-scoped targets for a curriculum node that shares a source page."""
from __future__ import annotations

import unicodedata


def _fold(value: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value).lower())
                   if not unicodedata.combining(c))


def scoped_targets(kc_id: str, rules: list[str]) -> list[str] | None:
    if kc_id != 'g_k4':
        return None
    state_rule = next((r for r in rules if 'pointill' in _fold(r)
                       and 'continu' in _fold(r) and 'prevu' in _fold(r)), None)
    if not state_rule:
        raise ValueError("La regle d'etat de la page 4 est absente : quiz suspendu.")
    return [
        f"Distinguer une action prevue (trait pointille) d'une action en cours (trait continu) : {state_rule}",
        "Choisir le trace quand une action passe de prevue a en cours, sans confondre le trait avec la couleur.",
    ]
