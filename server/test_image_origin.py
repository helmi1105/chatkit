import unittest
from datetime import datetime
from unittest.mock import patch
from chatkit.types import ThreadMetadata
from app.data_store import MyDataStore


class ImageOriginTests(unittest.TestCase):
    def test_origin_is_allowed_and_existing_domains_preserved(self):
        thread = ThreadMetadata(id='t', created_at=datetime.now(), allowed_image_domains=['https://example.org'])
        with patch.dict('os.environ', {'PUBLIC_BASE_URL': 'http://127.0.0.1:8000/'}):
            MyDataStore._with_image_origin(thread)
            MyDataStore._with_image_origin(thread)
        self.assertEqual(thread.allowed_image_domains, ['https://example.org', 'http://127.0.0.1:8000'])
