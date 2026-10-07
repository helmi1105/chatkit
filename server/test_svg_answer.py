import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from app.svg_answer import IllustratedAnswer, validate_svg
from app.orchestrator import Orchestrator, Session


SVG = '<svg viewBox="0 0 400 240"><circle cx="100" cy="100" r="30" fill="red"/></svg>'
ARROW = '<svg viewBox="0 0 400 240"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker></defs><line x1="10" y1="10" x2="100" y2="100" marker-end="url(#arrow)"/></svg>'


class SvgTests(unittest.TestCase):
    def test_inline_png_is_nonblank_and_cached(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from app.svg_answer import inline_image_url
        with tempfile.TemporaryDirectory() as folder, patch('app.svg_answer.INLINE_IMAGE_DIR', Path(folder)):
            url = inline_image_url(SVG)
            import base64
            self.assertTrue(url.startswith('data:image/png;base64,'))
            self.assertNotIn('http://', url)
            path = next(Path(folder).glob('*.png'))
            self.assertEqual(base64.b64decode(url.split(',', 1)[1]), path.read_bytes())
            self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            with Image.open(path) as picture:
                self.assertGreater(len(picture.getcolors(1000000)), 1)
                self.assertLessEqual(max(picture.size), 1200)
            self.assertEqual(inline_image_url(SVG), url)
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_local_arrow_markers(self):
        for attr in ['marker-start', 'marker-mid', 'marker-end']:
            self.assertIn(attr, validate_svg(ARROW.replace('marker-end', attr)))

    def test_rejects_invalid_marker_references(self):
        for value in ['url(https://example.com/a.svg#arrow)', 'url(#missing)', 'url(data:x)',
                      'url(#arrow) url(#arrow)']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_svg(ARROW.replace('url(#arrow)', value))
        for markup in [ARROW.replace('<path ', '<path marker-end="url(#arrow)" '),
                       ARROW.replace('<line ', '<line id="arrow" '),
                       SVG.replace('<circle ', '<circle id="x" marker-end="url(#x)" ')]:
            with self.assertRaises(ValueError):
                validate_svg(markup)

    def test_geometry_gets_svg_namespace(self):
        self.assertIn('http://www.w3.org/2000/svg', validate_svg(SVG))

    def test_harmless_model_formatting(self):
        wrapped = '```svg\n<?xml version="1.0"?>\n<!-- illustration -->' + SVG + '\n```'
        self.assertIn('circle', validate_svg(wrapped))
        self.assertNotIn('<!--', validate_svg(wrapped))
        self.assertIn('version=', validate_svg(SVG.replace('<svg ', '<svg version="1.1" ')))

    def test_rejects_active_and_external_content(self):
        for child in ['<script/>', '<foreignObject/>', '<image href="https://example.com"/>',
                      '<circle onload="alert(1)"/>', '<circle fill="url(#x)"/>',
                      '<circle style="fill:red"/>', '<x:circle xmlns:x="urn:other"/>']:
            with self.subTest(child=child), self.assertRaises(ValueError):
                validate_svg('<svg viewBox="0 0 400 240">' + child + '</svg>')
        with self.assertRaises(ValueError):
            validate_svg('<!DOCTYPE svg>' + SVG)


class FreeAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_diagram_and_rejection_preserve_pending_quiz(self):
        for svg, pages, expected in [('', [], 2), (SVG, [6], 3), (ARROW, [6], 3),
                                     (SVG, [999], 2), ('<script/>', [6], 2)]:
            with self.subTest(svg=svg, pages=pages):
                orch = Orchestrator.__new__(Orchestrator)
                orch.graph = SimpleNamespace(nodes={})
                orch.doc = SimpleNamespace(pages={6: 'source'}, page_text=lambda p: 'source', search_pages=lambda *a, **kw: [6],
                    pages_text=lambda *a, **kw: 'source', pdf_url=lambda p: f'/course.pdf#page={p}')
                orch._record_event = Mock()
                sess = Session(phase='waiting_answers', quiz_questions={'1': {'answer': 'A'}})
                out = IllustratedAnswer(answer='Explanation p. 6', svg=svg, illustration_pages=pages)
                with patch('app.orchestrator.run_structured', new=AsyncMock(return_value=out)):
                    blocks = await orch._answer_free_question(sess, 'Draw a symbol', None)
                self.assertEqual(len(blocks), expected)
                self.assertIn('Explanation', blocks[0]['text'])
                self.assertEqual(sess.phase, 'waiting_answers')
                self.assertEqual(sess.quiz_questions, {'1': {'answer': 'A'}})
                self.assertEqual(sess.mastery, {})
