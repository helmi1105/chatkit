"""Learner accounts and revocable, opaque browser sessions.

Read/write the durable backend directly: auth must not use stale process caches
or the learning store's best-effort writes.
"""
import argparse
from contextvars import ContextVar
import getpass
import hashlib
import hmac
import json
import re
import secrets
import time
import uuid

from app.storage import store

COOKIE_NAME = 'chatkit_session'
SESSION_SECONDS = 7 * 24 * 60 * 60
current_username = ContextVar('current_username', default=None)


class UsernameTaken(ValueError):
    pass


def username_key(username):
    name = username.strip().lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{2,63}', name):
        raise ValueError('Username must contain 3–64 letters, digits, dots, underscores or hyphens.')
    return name


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


class Accounts:
    def __init__(self, backend=None):
        self.backend = backend if backend is not None else store().backend

    def read(self, key):
        raw = self.backend.get(key)
        return json.loads(raw) if raw else None

    def write(self, key, value):
        self.backend.put(key, json.dumps(value).encode())

    def create(self, username, password, existing_user_id=None):
        name = username_key(username)
        if not 4 <= len(password) <= 256:
            raise ValueError('Use a password of 4–256 characters.')
        if self.read(f'auth/accounts/{name}.json'):
            raise UsernameTaken('Ce nom d’utilisateur est déjà utilisé.')
        user_id = str(uuid.UUID(existing_user_id)) if existing_user_id else str(uuid.uuid4())
        if self.read(f'auth/identities/{user_id}.json'):
            raise ValueError('This learner ID already belongs to an account.')
        salt = secrets.token_hex(16)
        account = dict(username=name, user_id=user_id, salt=salt,
                       password_hash=password_hash(password, salt))
        self.write(f'auth/identities/{user_id}.json', dict(username=name, user_id=user_id))
        if not self.backend.put_if_absent(f'auth/accounts/{name}.json', json.dumps(account).encode()):
            self.backend.delete(f'auth/identities/{user_id}.json')
            raise UsernameTaken('Ce nom d’utilisateur est déjà utilisé.')
        return self.public(account)

    @staticmethod
    def public(account):
        return {key: account[key] for key in ('username', 'user_id')}

    def login(self, username, password):
        try:
            name = username_key(username)
        except ValueError:
            name = '_invalid'
        # Shared durable backoff survives restarts and multiple app workers.
        throttle_key = f'auth/attempts/{name}.json'
        attempts = self.read(throttle_key) or {}
        now = time.time()
        if attempts.get('blocked_until', 0) > now:
            return None
        account = self.read(f'auth/accounts/{name}.json')
        salt = account['salt'] if account else '00' * 16
        candidate = password_hash(password, salt)
        if not account or not hmac.compare_digest(candidate, account['password_hash']):
            failures = attempts.get('failures', 0) + 1 if now - attempts.get('updated', 0) < 900 else 1
            self.write(throttle_key, dict(failures=failures, updated=now,
                                         blocked_until=now + min(900, 2 ** min(failures, 10))))
            return None
        self.backend.delete(throttle_key)
        return self.start_session(account)

    def start_session(self, account):
        token = secrets.token_urlsafe(32)
        self.write(self.session_key(token), dict(**self.public(account), expires=time.time() + SESSION_SECONDS))
        return token, self.public(account)

    @staticmethod
    def session_key(token):
        return 'auth/sessions/' + hashlib.sha256(token.encode()).hexdigest() + '.json'

    def resolve(self, token):
        if not token or len(token) > 128:
            return None
        session = self.read(self.session_key(token))
        if not session or session['expires'] <= time.time():
            return None
        return self.public(session)

    def logout(self, token):
        if token:
            self.backend.delete(self.session_key(token))


def main():
    parser = argparse.ArgumentParser(description='Create a learner account in the configured state backend.')
    parser.add_argument('username')
    parser.add_argument('--existing-user-id', help='Administrator-approved UUID of an existing anonymous learner.')
    args = parser.parse_args()
    password = getpass.getpass('Password (minimum 4 characters): ')
    if password != getpass.getpass('Confirm password: '):
        parser.error('Passwords do not match.')
    try:
        account = Accounts().create(args.username, password, args.existing_user_id)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Created {account['username']} (ID: {account['user_id']})")


if __name__ == '__main__':
    main()
