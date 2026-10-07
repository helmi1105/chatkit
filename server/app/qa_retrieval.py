"""Budgeted, multi-query course retrieval for free questions (no API calls)."""
import re
import unicodedata


def tokens(text):
    folded = ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))
    stop = {'le', 'la', 'les', 'des', 'du', 'de', 'un', 'une', 'et', 'en', 'dans', 'avec', 'pour', 'sur', 'est', 'que', 'qui'}
    return {t for t in re.findall(r'[a-z0-9]+', folded) if len(t) > 2 and t not in stop}


def retrieve_question_context(doc, question, current_pages=(), *, max_pages=10, max_chars=24000):
    available = set(doc.pages)
    explicit = []
    # Only numbers introduced as page references, never operational times/counts.
    for match in re.finditer(r'\b(?:pages?|p\.)\s*(\d+)(?:\s*(?:à|a|au|[-–])\s*(\d+))?', question, re.I):
        start = int(match[1])
        end = int(match[2] or start)
        explicit.extend(p for p in sorted(available) if start <= p <= end)
    queries = [question] + [s.strip() for s in re.split(r'[;\n?!]|\bet\b', question) if len(tokens(s)) >= 2]
    scores = {}
    for query in dict.fromkeys(queries):
        for rank, page in enumerate(doc.search_pages(query, k=max_pages)):
            if page in available:
                scores[page] = scores.get(page, 0) + 1 / (10 + rank)
    ranked = sorted(scores, key=lambda p: (-scores[p], p))
    # Broader questions benefit from coverage of distinct requested concepts.
    uncovered = tokens(question)
    diversified = []
    while ranked:
        page = max(ranked, key=lambda p: scores[p] * (1 + len(uncovered & tokens(doc.page_text(p)))))
        diversified.append(page)
        uncovered -= tokens(doc.page_text(page))
        ranked.remove(page)
    selected = list(dict.fromkeys(explicit))[:max_pages]
    for page in diversified[:6]:
        if page not in selected and len(selected) < max_pages:
            selected.append(page)
    if tokens(question) & {'svg', 'symbole', 'symboles', 'couleur', 'couleurs', 'forme', 'formes', 'schema', 'dessine'}:
        for page in (3, 4, 5):
            if page in available and page not in selected and len(selected) < max_pages:
                selected.append(page)
    for page in [*current_pages, *diversified]:
        if page in available and page not in selected and len(selected) < max_pages:
            selected.append(page)
    if not selected:
        return '', [], {'selected_pages': [], 'explicit_pages': explicit, 'truncated_pages': []}
    blocks, included, truncated = [], [], []
    allowance = max_chars // len(selected)
    wanted = tokens(question)
    for page in selected:
        header = f'=== Page {page} ===\n'
        body = doc.page_text(page).strip()
        if not body:
            continue
        limit = max(0, allowance - len(header) - 2)
        if len(body) > limit:
            # Preserve relevant passages even when they occur near the page end.
            passages = [body[i:i + 700] for i in range(0, len(body), 700)]
            order = sorted(range(len(passages)), key=lambda i: (-len(wanted & tokens(passages[i])), i))
            chosen, used = [], 0
            for i in order:
                if used + len(passages[i]) + 5 <= limit:
                    chosen.append(i)
                    used += len(passages[i]) + 5
            body = '\n[…]\n'.join(passages[i] for i in sorted(chosen)) if chosen else passages[order[0]][:limit]
            truncated.append(page)
        if body:
            blocks.append(header + body)
            included.append(page)
    return '\n\n'.join(blocks), included, {'selected_pages': selected, 'explicit_pages': list(dict.fromkeys(explicit)), 'truncated_pages': truncated}
