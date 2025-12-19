

from chatkit.server import StreamingResult
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.data_store import USER_ID_KEY, MyDataStore
from app.chatkit_server import MyChatKitServer

app = FastAPI()

# Add CORS to allow our server to be called from local front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = MyDataStore()
server = MyChatKitServer(store=data_store)

# forward HTTP requests to the server
@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    userId = request.headers.get(USER_ID_KEY)

    if userId is None:
        return JSONResponse(
            status_code=400,
            content={
                "message": "UserId Missing"
            }
        )

    payload = await request.body()
    result = await server.process(payload, context={USER_ID_KEY: userId})

    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")

    return JSONResponse(result)
