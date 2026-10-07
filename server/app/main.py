"""FastAPI entry point: ChatKit endpoint, attachment upload, static assets,
health and progress endpoints, optional trainer access code."""
from __future__ import annotations

import hmac
import asyncio
import os
from pathlib import Path

from chatkit.server import StreamingResult
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from app.chatkit_server import MyChatKitServer
from app.data_store import MAX_IMAGE_ATTACHMENT_BYTES, USER_ID_KEY, MyDataStore
from app.auth import Accounts, COOKIE_NAME, SESSION_SECONDS, current_username, UsernameTaken
from pydantic import BaseModel, Field

app = FastAPI()
APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR.parent.parent / "docs"
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
if '*' in ALLOWED_ORIGINS:
    raise ValueError('ALLOWED_ORIGINS must list explicit frontend origins for account login.')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional shared code for the trainers' demonstrator: when ACCESS_CODE is
# set, every stateful endpoint requires the X-Access-Code header.
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()

data_store = MyDataStore()
server = MyChatKitServer(store=data_store, attachment_store=data_store)
accounts = Accounts()
COOKIE_SECURE = os.getenv('AUTH_COOKIE_SECURE', 'false').lower() == 'true'


@app.middleware('http')
async def identity(request: Request, call_next):
    path = request.url.path
    protected = path in ('/chatkit', '/progress', '/auth/me', '/auth/logout') or path.startswith('/attachments/') or path.startswith('/static/uploads/')
    if request.method == 'OPTIONS':
        return await call_next(request)
    if request.method not in ('GET', 'HEAD') and (protected or path in ('/auth/login', '/auth/register')):
        # A custom header prevents cross-site HTML forms from performing actions;
        # CORS restricts which origins may send that header with credentials.
        if request.headers.get('X-ChatKit-Request') != '1':
            return JSONResponse({'message': 'Invalid request origin.'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse({'message': 'Invalid request origin.'}, status_code=403)
    principal = None
    if protected:
        principal = await asyncio.to_thread(accounts.resolve, request.cookies.get(COOKIE_NAME))
        if not principal:
            return JSONResponse({'message': 'Veuillez vous connecter.', 'code': 'login_required'}, status_code=401)
        request.state.account = principal
    token = current_username.set(principal['username'] if principal else None)
    try:
        response = await call_next(request)
        if protected or path.startswith('/auth/'):
            response.headers['Cache-Control'] = 'no-store'
        return response
    finally:
        current_username.reset(token)


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@app.post('/auth/register')
async def register(body: Login, request: Request):
    try:
        account = await asyncio.to_thread(accounts.create, body.username, body.password)
    except UsernameTaken as exc:
        return JSONResponse({'message': str(exc)}, status_code=409)
    except ValueError:
        return JSONResponse({'message': 'Utilisez un nom de 3 à 64 caractères (lettres, chiffres, ., _ ou -) et un mot de passe de 4 à 256 caractères.'}, status_code=400)
    token, account = await asyncio.to_thread(accounts.start_session, account)
    await asyncio.to_thread(accounts.logout, request.cookies.get(COOKIE_NAME))
    response = JSONResponse(account, status_code=201)
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=COOKIE_SECURE, samesite='lax', path='/')
    return response


@app.post('/auth/login')
async def login(body: Login, request: Request):
    result = await asyncio.to_thread(accounts.login, body.username, body.password)
    if result is None:
        return JSONResponse({'message': 'Identifiants incorrects ou trop de tentatives. Réessayez plus tard.'}, status_code=401)
    await asyncio.to_thread(accounts.logout, request.cookies.get(COOKIE_NAME))
    token, account = result
    response = JSONResponse(account)
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=COOKIE_SECURE, samesite='lax', path='/')
    return response


@app.get('/auth/me')
async def me(request: Request):
    return request.state.account


@app.post('/auth/logout')
async def logout(request: Request):
    await asyncio.to_thread(accounts.logout, request.cookies.get(COOKIE_NAME))
    response = JSONResponse({'ok': True})
    response.delete_cookie(COOKIE_NAME, path='/', secure=COOKIE_SECURE, httponly=True, samesite='lax')
    return response


@app.get('/static/{asset_path:path}')
async def static_asset(asset_path: str, request: Request):
    target = (APP_DIR / asset_path).resolve()
    if APP_DIR not in target.parents or not target.is_file():
        return Response(status_code=404)
    if asset_path.startswith('uploads/'):
        owner = data_store._attachment_owners.get(target.stem)
        if owner != request.state.account['user_id']:
            return Response(status_code=404)
    elif not (asset_path == 'guide_fr.html' or
              (target.parent == APP_DIR and target.suffix.lower() == '.pdf') or
              (target.parent == APP_DIR / 'pages' and target.suffix.lower() == '.png')):
        return Response(status_code=404)
    return FileResponse(target)


def _access_denied(request: Request) -> JSONResponse | None:
    if not ACCESS_CODE:
        return None
    given = (request.headers.get("X-Access-Code") or "").strip()
    if not given:
        return JSONResponse(status_code=401, content={"message": "access code required"})
    if not hmac.compare_digest(given, ACCESS_CODE):
        return JSONResponse(status_code=401, content={"message": "invalid access code"})
    return None


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "access_code_required": bool(ACCESS_CODE),
        "doctrine_source": server.orch.doc.source,
        "storage": server.orch.store.backend.name,
        "bank_questions": sum(server.orch.bank.stats().values()),
    })


@app.get("/progress")
async def progress(request: Request) -> Response:
    denied = _access_denied(request)
    if denied:
        return denied
    user_id = request.state.account['user_id']
    if not user_id:
        return JSONResponse(status_code=400, content={"message": "UserId Missing"})
    return JSONResponse(server.orch.progress_summary(user_id))


@app.api_route("/attachments/{attachment_id}/upload", methods=["PUT", "POST"])
async def upload_attachment(attachment_id: str, request: Request) -> Response:
    denied = _access_denied(request)
    if denied:
        return denied
    if data_store._attachment_owners.get(attachment_id) != request.state.account['user_id']:
        return Response(status_code=404)
    # Anti-DoS guard: reject on the ANNOUNCED size before reading the body
    # (a ~150 MB unauthenticated request used to OOM the container).
    max_request_bytes = MAX_IMAGE_ATTACHMENT_BYTES + 1 * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_request_bytes:
                return JSONResponse(status_code=413, content={"message": "Upload too large."})
        except ValueError:
            return JSONResponse(status_code=400, content={"message": "Invalid Content-Length."})
    content_type = request.headers.get("content-type")
    content: bytes
    if content_type and content_type.lower().startswith("multipart/form-data"):
        form = await request.form()
        content = b""
        for value in form.values():
            if hasattr(value, "read") and hasattr(value, "filename"):
                content = await value.read()
                content_type = getattr(value, "content_type", None) or content_type
                break
        if not content:
            return JSONResponse(status_code=400, content={"message": "No image file found in multipart upload."})
    else:
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_request_bytes:
                return JSONResponse(status_code=413, content={"message": "Upload too large."})
            chunks.append(chunk)
        content = b"".join(chunks)
    await data_store.upload_attachment_bytes(attachment_id, content, content_type)
    return Response(status_code=204)


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    denied = _access_denied(request)
    if denied:
        return denied
    user_id = request.state.account['user_id']
    if user_id is None:
        return JSONResponse(status_code=400, content={"message": "UserId Missing"})
    provider = request.headers.get("X-Provider")
    api_key = request.headers.get("X-Provider-Api-Key")
    payload = await request.body()
    frontend_origin = request.headers.get('origin') or ALLOWED_ORIGINS[0]
    result = await server.process(payload, context={USER_ID_KEY: user_id, "provider": provider,
                                                  "api_key": api_key, "frontend_origin": frontend_origin})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")
    return JSONResponse(result)
