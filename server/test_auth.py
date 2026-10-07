from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi.testclient import TestClient
from app.auth import Accounts, COOKIE_NAME
from app.storage import _LocalBackend
from app import main
from app.data_store import MyDataStore
from chatkit.types import ThreadMetadata, AttachmentCreateParams
from chatkit.store import NotFoundError


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.accounts = Accounts(_LocalBackend(Path(self.temp.name)))
        self.alice = self.accounts.create('Alice', 'alice password 123')

    def test_hashing_persistence_and_case_insensitive_login(self):
        raw = self.accounts.backend.get('auth/accounts/alice.json')
        self.assertNotIn(b'alice password 123', raw)
        other_process = Accounts(self.accounts.backend)
        token, account = other_process.login(' ALICE ', 'alice password 123')
        self.assertEqual(account, self.alice)
        self.assertEqual(other_process.resolve(token), account)
        self.assertNotIn(token.encode(), self.accounts.backend.get(self.accounts.session_key(token)))
        self.accounts.logout(token)
        self.assertIsNone(other_process.resolve(token))

    def test_bad_password_backoff_and_expiry(self):
        with patch('app.auth.time.time', return_value=100):
            self.assertIsNone(self.accounts.login('alice', 'wrong'))
            self.assertIsNone(self.accounts.login('alice', 'alice password 123'))
        with patch('app.auth.time.time', return_value=1000):
            token, _ = self.accounts.login('alice', 'alice password 123')
        with patch('app.auth.time.time', return_value=1000000):
            self.assertIsNone(self.accounts.resolve(token))
        self.assertIsNone(self.accounts.resolve('invented'))

    def test_duplicate_and_invalid_accounts(self):
        for username, password in [('Alice', 'another password'), ('../bob', 'long enough password'), ('bob', 'abc')]:
            with self.assertRaises(ValueError):
                self.accounts.create(username, password)
        with self.assertRaises(ValueError):
            self.accounts.create('bob', 'long enough password', self.alice['user_id'])

    def test_concurrent_registration_has_one_winner(self):
        def create(_):
            try:
                return Accounts(self.accounts.backend).create('new_learner', 'long enough password')
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(create, range(4)))
        winners = [result for result in results if result]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.accounts.login('new_learner', 'long enough password')[1], winners[0])

    def test_storage_errors_are_not_silently_successful(self):
        with patch.object(self.accounts.backend, 'put', side_effect=OSError('unavailable')):
            with self.assertRaises(OSError):
                self.accounts.login('alice', 'alice password 123')


class EndpointTests(AccountTests):
    def setUp(self):
        super().setUp()
        self.patcher = patch.object(main, 'accounts', self.accounts)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        self.headers = {'X-ChatKit-Request': '1', 'Origin': 'http://localhost:3000'}

    def login(self):
        return self.client.post('/auth/login', headers=self.headers,
                                json={'username': 'alice', 'password': 'alice password 123'})

    def test_public_registration_signs_in_and_preserves_existing_account(self):
        response = self.client.post('/auth/register', headers=self.headers,
                                    json={'username': 'New_Learner', 'password': 'my new password', 'existing_user_id': self.alice['user_id']})
        self.assertEqual(response.status_code, 201)
        account = response.json()
        self.assertEqual(account['username'], 'new_learner')
        self.assertNotEqual(account['user_id'], self.alice['user_id'])
        self.assertEqual(self.client.get('/auth/me').json(), account)
        duplicate = self.client.post('/auth/register', headers=self.headers,
                                     json={'username': 'NEW_LEARNER', 'password': 'different password'})
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(self.accounts.login('new_learner', 'my new password')[1], account)

    def test_registration_validation_and_csrf(self):
        for username, password in [('ab', 'valid password'), ('../bob', 'valid password'), ('bob', 'abc')]:
            response = self.client.post('/auth/register', headers=self.headers, json=dict(username=username, password=password))
            self.assertEqual(response.status_code, 400)
        body = {'username': 'bob', 'password': 'valid password'}
        self.assertEqual(self.client.post('/auth/register', json=body).status_code, 403)
        self.assertEqual(self.client.post('/auth/register', headers={**self.headers, 'Origin': 'https://evil.example'}, json=body).status_code, 403)

    def test_four_character_passwords(self):
        for index, password in enumerate(('abcd', '1234', 'ab12')):
            username = f'learner_{index}'
            response = self.client.post('/auth/register', headers=self.headers,
                                        json={'username': username, 'password': password})
            self.assertEqual(response.status_code, 201)
            self.assertEqual(self.accounts.login(username, password)[1], response.json())

    def test_identity_is_cookie_based_and_logout_revokes(self):
        self.assertEqual(self.client.get('/progress', headers={'userId': self.alice['user_id']}).status_code, 401)
        response = self.login()
        self.assertEqual(response.status_code, 200)
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('SameSite=lax', response.headers['set-cookie'])
        self.assertEqual(self.client.get('/auth/me').json(), self.alice)
        with patch.object(main.server.orch, 'progress_summary', return_value={}) as progress, patch.object(main, 'ACCESS_CODE', ''):
            self.assertEqual(self.client.get('/progress', headers={'userId': 'another-learner'}).status_code, 200)
            progress.assert_called_once_with(self.alice['user_id'])
        token = self.client.cookies.get(COOKIE_NAME)
        self.assertEqual(self.client.post('/auth/logout', headers=self.headers).status_code, 200)
        self.assertIsNone(self.accounts.resolve(token))
        self.assertEqual(self.client.get('/auth/me').status_code, 401)

    def test_csrf_and_static_privacy(self):
        body = {'username': 'alice', 'password': 'alice password 123'}
        self.assertEqual(self.client.post('/auth/login', json=body).status_code, 403)
        self.assertEqual(self.client.post('/auth/login', json=body, headers={**self.headers, 'Origin': 'https://evil.example'}).status_code, 403)
        self.login()
        for path in ('auth.py', 'data/auth/accounts/alice.json', 'evidence_log.jsonl', 'doctrine_pages.json'):
            self.assertEqual(self.client.get('/static/' + path).status_code, 404)
        self.assertEqual(self.client.get('/static/guide_fr.html').status_code, 200)

    def test_other_learner_upload_is_denied(self):
        self.login()
        with patch.dict(main.data_store._attachment_owners, {'owned-by-bob': 'bob'}), patch.object(main, 'ACCESS_CODE', ''):
            response = self.client.put('/attachments/owned-by-bob/upload', headers=self.headers, content=b'fake image')
        self.assertEqual(response.status_code, 404)

    def test_chatkit_uses_verified_identity_and_username_context(self):
        from app.auth import current_username
        self.login()
        observed = {}
        async def process(payload, context):
            observed.update(context)
            observed['username'] = current_username.get()
            return {}
        with patch.object(main.server, 'process', side_effect=process), patch.object(main, 'ACCESS_CODE', ''):
            response = self.client.post('/chatkit', content=b'{}', headers={**self.headers, 'userId': 'bob'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed['userId'], self.alice['user_id'])
        self.assertEqual(observed['username'], 'alice')

    def test_attachment_upload_and_read_require_owner(self):
        self.login()
        async def check():
            with patch('app.data_store.UPLOAD_DIR', Path(self.temp.name)):
                attachment = await main.data_store.create_attachment(
                    AttachmentCreateParams(name='example.png', mime_type='image/png', size=3),
                    {'userId': self.alice['user_id']})
            with self.assertRaises(NotFoundError):
                await main.data_store.load_attachment(attachment.id, {'userId': 'bob'})
            return attachment
        attachment = asyncio.run(check())
        try:
            with patch.object(main, 'ACCESS_CODE', ''):
                response = self.client.put('/attachments/' + attachment.id + '/upload', content=b'png', headers=self.headers)
            self.assertEqual(response.status_code, 204)
            loaded = asyncio.run(main.data_store.load_attachment(attachment.id, {'userId': self.alice['user_id']}))
            self.assertTrue(str(loaded.preview_url).startswith('data:image/png;base64,'))
        finally:
            asyncio.run(main.data_store.delete_attachment(attachment.id, {'userId': self.alice['user_id']}))

    def test_thread_lookup_does_not_cross_accounts(self):
        storage = MyDataStore()
        storage._persist = lambda user_id: None
        storage._state = type('MemoryState', (), {'get_json': lambda *args: None})()
        async def check():
            await storage.save_thread(ThreadMetadata(id='thread-alice', created_at=datetime.now()), {'userId': self.alice['user_id']})
            with self.assertRaises(NotFoundError):
                await storage.load_thread('thread-alice', {'userId': 'bob'})
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
