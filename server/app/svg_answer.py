"""Restricted SVG illustrations for free answers, never quiz content."""
import base64
import html
import re
import os
import hashlib
import tempfile
from pathlib import Path
import pymupdf
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from chatkit.actions import ActionConfig
from chatkit.widgets import Button, Card, Image, Caption


class IllustratedAnswer(BaseModel):
    answer: str
    svg: str = ''
    illustration_pages: list[int] = Field(default_factory=list)


SVG_INSTRUCTIONS = (
    'Retourne answer (explication française avec pages), svg et illustration_pages. '
    'svg reste vide sauf si un dessin aide à répondre à la question. '
    'Dessine uniquement les attributs explicitement documentés dans les extraits. '
    'Respecte les positions des surcharges indiquées par la source : des points de niveau '
    'décrits à l’extérieur doivent rester entièrement hors de la forme, avec un espace visible. '
    'Ne confonds pas ces points de niveau avec les pointillés du contour. '
    'Si les formes, couleurs ou contours ne sont pas assez décrits, explique cette limite et laisse svg vide. '
    'Une illustration est schématique, pas une reproduction officielle validée. '
    'SVG autonome avec xmlns="http://www.w3.org/2000/svg" et viewBox="0 0 400 240". '
    'Balises autorisées : svg, g, rect, circle, ellipse, line, polyline, polygon, path, text, tspan, title, desc. '
    'Utilise des attributs simples de géométrie, fill, stroke, stroke-width, stroke-dasharray, font-size. '
    'Pour les flèches, defs et marker sont autorisés avec un id unique ; marker-start, marker-mid '
    'et marker-end peuvent utiliser uniquement url(#id) vers un marker du même SVG. '
    'Aucun script, style CSS, lien, image externe, animation ou autre référence url(). '
    'illustration_pages doit citer les pages fournies qui justifient le dessin.'
)


def validate_svg(markup):
    if len(markup) > 30000:
        raise ValueError('SVG exceeds 30000 characters')
    markup = markup.strip()
    fenced = re.fullmatch(r'```(?:svg|xml)?\s*\n(.*?)\n```', markup, re.S)
    if fenced:
        markup = fenced.group(1).strip()
    markup = re.sub(r'^<\?xml\s+[^?]*\?>\s*', '', markup)
    markup = re.sub(r'<!--.*?-->', '', markup, flags=re.S)
    if '<!' in markup or '<?' in markup:
        raise ValueError('Unsupported SVG declaration or size')
    root = ET.fromstring(markup)
    tags = {'svg','g','rect','circle','ellipse','line','polyline','polygon','path','text','tspan','title','desc','defs','marker'}
    attrs = {'viewBox','width','height','x','y','x1','x2','y1','y2','cx','cy','r','rx','ry','d','points',
             'fill','stroke','stroke-width','stroke-dasharray','stroke-linecap','stroke-linejoin',
             'opacity','fill-opacity','stroke-opacity','transform','font-size','font-family',
             'font-weight','text-anchor','dominant-baseline', 'dx', 'dy', 'version',
             'preserveAspectRatio', 'fill-rule', 'clip-rule', 'id', 'markerWidth', 'markerHeight',
             'refX', 'refY', 'orient', 'markerUnits'}
    if root.tag not in ('svg', '{http://www.w3.org/2000/svg}svg') or 'viewBox' not in root.attrib:
        raise ValueError('Missing SVG root/viewBox')
    values = root.attrib['viewBox'].replace(',', ' ').split()
    if len(values) != 4 or not all(re.fullmatch(r'-?\d+(?:\.\d+)?', v) for v in values) or any(float(v) <= 0 or float(v) > 4096 for v in values[2:]):
        raise ValueError('Invalid viewBox')
    nodes = list(root.iter())
    if len(nodes) > 200:
        raise ValueError('Too many SVG elements')
    ids = {}
    for node in nodes:
        identifier = node.get('id')
        if identifier is not None:
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', identifier) or identifier in ids:
                raise ValueError('Invalid or duplicate SVG id')
            ids[identifier] = node
    for node in nodes:
        tag = node.tag.removeprefix('{http://www.w3.org/2000/svg}')
        if tag not in tags:
            raise ValueError(f'Unsupported SVG element: {tag[:80]}')
        for key, value in node.attrib.items():
            if key in {'marker-start', 'marker-mid', 'marker-end'}:
                if value == 'none':
                    continue
                match = re.fullmatch(r'''url\(\s*(['"]?)#([A-Za-z_][A-Za-z0-9_.-]*)\1\s*\)''', value)
                target = ids.get(match.group(2)) if match else None
                if target is None or target.tag.removeprefix('{http://www.w3.org/2000/svg}') != 'marker':
                    raise ValueError('Marker must reference a local SVG marker')
                # Avoid recursive markers, including references on their children.
                if any(node in list(marker.iter()) for marker in nodes
                       if marker.tag.removeprefix('{http://www.w3.org/2000/svg}') == 'marker'):
                    raise ValueError('Nested marker references are unsupported')
                continue
            if key not in attrs or re.search(r'url\s*\(|[<>\\]|(?:https?|data|javascript):', value, re.I):
                raise ValueError(f'Unsupported SVG attribute: {key[:80]}')
        node.tag = '{http://www.w3.org/2000/svg}' + tag
    return ET.tostring(root, encoding='unicode')


INLINE_IMAGE_DIR = Path(__file__).parent / 'generated_flashcards'


def inline_image_url(svg):
    cleaned = validate_svg(svg)
    root = ET.fromstring(cleaned)
    _, _, width, height = map(float, root.attrib['viewBox'].replace(',', ' ').split())
    scale = min(1200 / width, 800 / height)
    root.set('width', str(max(1, round(width * scale))))
    root.set('height', str(max(1, round(height * scale))))
    data = ET.tostring(root, encoding='utf-8')
    name = hashlib.sha256(b'flashcard-png-v1' + data).hexdigest() + '.png'
    INLINE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    target = INLINE_IMAGE_DIR / name
    if not target.exists():
        with pymupdf.open(stream=data, filetype='svg') as drawing:
            png = drawing[0].get_pixmap(alpha=False).tobytes('png')
        # Unique temporary files avoid concurrent writes exposing incomplete images.
        with tempfile.NamedTemporaryFile(dir=INLINE_IMAGE_DIR, suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            output.write(png)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    # The hosted HTTPS ChatKit iframe cannot fetch loopback HTTP assets.
    # Embed raster PNG bytes (not SVG markup), so no cross-origin request occurs.
    return 'data:image/png;base64,' + base64.b64encode(target.read_bytes()).decode('ascii')


def inline_illustration_widget(svg):
    """Validated SVG as a native chat image; no viewer action or answer metadata."""
    return {'type': 'widget', 'title': 'Illustration', 'widget': Card(children=[
        Image(src=inline_image_url(svg), alt='Symbole à étudier',
              width='100%', height=240, fit='contain', frame=True, radius='md'),
        Caption(value='Illustration schématique — à comparer au mémento.')])}


def illustration_widget(svg, pages, pdf_url):
    encoded = base64.b64encode(svg.encode()).decode('ascii')
    links = ' · '.join(f'<a target="_blank" rel="noopener" href="{html.escape(pdf_url(p), quote=True)}">Page {p}</a>' for p in pages)
    document = ('<!doctype html><html lang="fr"><meta charset="utf-8"><body style="font-family:system-ui;padding:20px">'
                '<p>Illustration schématique générée — à comparer au mémento.</p>'
                f'<img alt="Schéma explicatif" style="max-width:100%;width:600px" src="data:image/svg+xml;base64,{encoded}">'
                f'<p>{links}</p></body></html>')
    return {'type': 'widget', 'title': 'Illustration', 'widget': Card(children=[Button(
        label='Voir le schema', onClickAction=ActionConfig(type='report.open', handler='client',
        payload={'html': document, 'url': '', 'title': 'Illustration'}))])}
