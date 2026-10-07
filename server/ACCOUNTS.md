# Learner accounts

Username/password login is required for conversations, progress and uploads.
Anyone can select **Créer un compte** on the login screen and register with a
unique username and a password of at least 4 characters. Registration signs the
learner in immediately with a new permanent user ID. Duplicate usernames are
rejected case-insensitively, including simultaneous registration requests.
The optional shared `ACCESS_CODE` still applies to learning access when configured.

## Create an account from the command line (optional)

From Git Bash, in `server/`, with the virtual environment active:

```bash
python -m app.auth learner_001
```

Enter the password twice at the hidden prompts (4–256 characters). Do not put
passwords in command arguments, source code or Git. Usernames are case-insensitive
and use 3–64 letters, digits, dots, underscores or hyphens. The command prints the
permanent user ID. Run account creation sequentially, using the same storage
configuration as the application server.

Start the backend and frontend as usual, then open the frontend and log in.
Accounts persist under `auth/` in the existing state backend (local `app/data/`
for development, or the configured S3 bucket). They are not stored in Git.
For a deployed server, run the command with that server's S3 settings; a local
account in `app/data/` is not automatically copied to the deployed server.

## Existing anonymous progress

Old progress remains untouched. An administrator who has verified ownership can
link the old browser's `localStorage.userId` UUID when creating an account:

```bash
python -m app.auth learner_001 --existing-user-id EXISTING_UUID
```

The learner cannot claim another ID through the login interface or request
headers. Without this option, a new account starts with a fresh UUID.

## Browser and deployment settings

The frontend calls `/backend/*` through a Next.js rewrite, keeping cookies on the
frontend origin. Set `CHATKIT_BACKEND_URL` (or the existing
`NEXT_PUBLIC_CHATKIT_API_URL`) to the backend address **before building** the web
app. For local development the default is `http://127.0.0.1:8000`.

On the backend:

```bash
export ALLOWED_ORIGINS="https://YOUR-FRONTEND-HOST"
export AUTH_COOKIE_SECURE="true"
```

Use HTTPS in production. Local defaults allow `http://localhost:3000` and
`http://127.0.0.1:3000`, with non-Secure cookies for HTTP development. Wildcard
origins are rejected. An existing `ACCESS_CODE` remains an additional shared gate.

Sessions use opaque random tokens in HttpOnly, SameSite=Lax cookies and expire
after seven days. Logout revokes the session in durable storage. Session tokens
are hashed in storage; passwords are salted and hashed with scrypt. Failed
logins trigger per-username backoff. For public deployment, also use ingress
rate limits: the JSON backoff counter is not an atomic distributed rate limiter.

## Logs and access

Threads and progress retain their existing `threads/<user_id>.json` and
`sessions/<user_id>.json` locations. New learning and feedback events contain
`user_id` and `username`; older events are not rewritten. Account records map
usernames to IDs. Authentication data bypasses the learning store's process
cache and propagates backend failures rather than silently accepting writes.

Only public course PDFs, page PNGs and the learner guide are served under
`/static`. Source code, logs and account/state files are not public assets.
Attachments require their owner's session; image previews are embedded after
upload so the ChatKit iframe does not need access to the session cookie.

Existing container-local attachment lifetime is unchanged. This first version
does not include self-service password reset, account administration UI or MFA.

## Checks

```bash
python -m unittest test_auth
```
