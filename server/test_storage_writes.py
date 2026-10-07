import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.storage import _LocalBackend


class LocalWrites(unittest.TestCase):
    def test_concurrent_writes(self):
        with TemporaryDirectory() as directory:
            backend = _LocalBackend(Path(directory))
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: backend.put('threads/test.json', json.dumps({'i': i}).encode()), range(80)))
            self.assertIn(json.loads(backend.get('threads/test.json'))['i'], range(80))
            self.assertFalse(list(Path(directory).rglob('*.tmp')))

    def test_windows_lock_retry(self):
        import os
        replace = os.replace
        error = OSError('locked')
        error.winerror = 32
        with TemporaryDirectory() as directory:
            backend = _LocalBackend(Path(directory))
            attempts = []
            def locked_once(src, dst):
                attempts.append(src)
                if len(attempts) == 1:
                    raise error
                replace(src, dst)
            with patch('app.storage.os.replace', side_effect=locked_once), patch('app.storage.time.sleep'):
                backend.put('threads/test.json', b'{}')
            self.assertEqual(backend.get('threads/test.json'), b'{}')
            self.assertEqual(len(attempts), 2)

    def test_permanent_failure_cleans_temp(self):
        with TemporaryDirectory() as directory:
            backend = _LocalBackend(Path(directory))
            backend.put('threads/test.json', b'old')
            with patch('app.storage.os.replace', side_effect=OSError('failure')):
                with self.assertRaises(OSError):
                    backend.put('threads/test.json', b'new')
            self.assertEqual(backend.get('threads/test.json'), b'old')
            self.assertFalse(list(Path(directory).rglob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
