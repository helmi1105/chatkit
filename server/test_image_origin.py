import unittest
from datetime import datetime
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from chatkit.types import WidgetItem
from chatkit.widgets import Card, Image
from chatkit.types import ThreadMetadata
from app.data_store import MyDataStore


class ImageOriginTests(unittest.TestCase):
    def test_origin_is_allowed_and_existing_domains_preserved(self):
        thread = ThreadMetadata(id='t', created_at=datetime.now(), allowed_image_domains=['https://example.org'])
        with patch.dict('os.environ', {'PUBLIC_BASE_URL': 'https://backend.example/'}):
            context = {'frontend_origin': 'https://frontend.example'}
            MyDataStore._with_image_origin(thread, context)
            MyDataStore._with_image_origin(thread, context)
        self.assertEqual(thread.allowed_image_domains, [
            'example.org', 'backend.example', 'frontend.example'])

    def test_local_inline_pngs_are_not_rejected_by_hostname_filter(self):
        thread = ThreadMetadata(id='t', created_at=datetime.now(),
                                allowed_image_domains=['http://localhost:3000'])
        MyDataStore._with_image_origin(thread, {'frontend_origin': 'http://localhost:3000'})
        self.assertEqual(thread.allowed_image_domains, [])

    def test_old_local_widget_is_repaired_on_copy_only(self):
        name = 'a' * 64 + '.png'
        src = 'http://localhost:3000/backend/static/generated_flashcards/' + name
        item = WidgetItem(id='w', thread_id='t', created_at=datetime.now(),
                          widget=Card(children=[Image(src=src)]))
        with TemporaryDirectory() as folder, patch('app.data_store.APP_DIR', Path(folder)):
            generated = Path(folder) / 'generated_flashcards'
            generated.mkdir()
            (generated / name).write_bytes(b'png')
            repaired = MyDataStore._local_widget_images(item.model_copy(deep=True),
                {'frontend_origin': 'http://localhost:3000'})
            self.assertEqual(repaired.widget.children[0].src, 'data:image/png;base64,cG5n')
            self.assertEqual(item.widget.children[0].src, src)
            remote = MyDataStore._local_widget_images(item.model_copy(deep=True),
                {'frontend_origin': 'https://frontend.example'})
            self.assertEqual(remote.widget.children[0].src, src)
