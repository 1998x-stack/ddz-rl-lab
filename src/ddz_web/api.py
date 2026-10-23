"""FastAPI REST/WS API. Operator-local by default; not an internet-facing auth system."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from .arena import ArenaManager

STATIC_DIR = Path(__file__).resolve().parent / 'static'


class NewGame(BaseModel):
    model_ids: tuple[str, str, str]
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    delay: float = Field(.7, ge=.05, le=3.0)


class Control(BaseModel):
    action: Literal['play', 'pause', 'step', 'speed']
    delay: float | None = None


def _origin_allowed(origin: str | None, host: str, trusted: set[str]) -> bool:
    # Modern browsers attach Origin to cross-site POST and WebSockets.
    # Nonbrowser local clients may omit it. Do not claim this is full authentication.
    if not origin:
        return True
    return origin in trusted or origin in {f'http://{host}', f'https://{host}'}


def create_app(registry_dir: str | Path = 'runs/registry',
               replay_dir: str | Path = 'runs/arena-replays',
               *, allow_baseline: bool = False,
               trusted_origins: set[str] | None = None,
               max_sessions: int = 12,
               allowed_hosts: list[str] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.manager.close()

    app = FastAPI(lifespan=lifespan, title='DDZ RL Lab Arena', version='0.7.0',
                  description='Local no-stakes, registered-checkpoint spectator arena')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or [
        'localhost', '127.0.0.1', 'testserver'])
    app.state.manager = ArenaManager(Path(registry_dir), Path(replay_dir),
                                     allow_baseline=allow_baseline, max_sessions=max_sessions)
    origins = trusted_origins or set()

    @app.middleware('http')
    async def same_origin_write(request: Request, call_next):
        if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            if not _origin_allowed(request.headers.get('origin'),
                                   request.headers.get('host', ''), origins):
                return JSONResponse(
                    status_code=403, content={'detail': 'cross-origin writes blocked'})
        return await call_next(request)

    @app.get('/')
    def page():
        return FileResponse(STATIC_DIR / 'index.html')

    @app.get('/static/app.js')
    def javascript():
        return FileResponse(STATIC_DIR / 'app.js', media_type='text/javascript')

    @app.get('/static/style.css')
    def stylesheet():
        return FileResponse(STATIC_DIR / 'style.css', media_type='text/css')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'schema': 'ddz-arena-v1'}

    @app.get('/api/models')
    def models():
        return app.state.manager.available_models()

    @app.post('/api/games', status_code=201)
    def create(data: NewGame):
        try:
            return app.state.manager.create(data.model_ids, seed=data.seed,
                                            delay=data.delay).snapshot()
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            logger.bind(component='web').warning('game_create_rejected: {}', type(exc).__name__)
            raise HTTPException(400, detail=str(exc)) from exc

    def get_game(game_id: str):
        try:
            return app.state.manager.get(game_id)
        except KeyError as exc:
            raise HTTPException(404, detail='unknown game') from exc

    @app.get('/api/games/{game_id}')
    def state(game_id: str):
        return get_game(game_id).snapshot()

    @app.post('/api/games/{game_id}/control')
    async def control(game_id: str, data: Control):
        try:
            return await get_game(game_id).control(data.action, data.delay)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.get('/api/games/{game_id}/replay')
    def replay(game_id: str):
        game = get_game(game_id)
        if not game.env.done:
            raise HTTPException(409, detail='replay is only available after game ends')
        path = game.replay_dir / f'{game.game_id}.json'
        if not path.is_file():
            raise HTTPException(404, detail='replay missing')
        return FileResponse(path, media_type='application/json', filename=f'{game.game_id}.json')

    @app.websocket('/api/games/{game_id}/events')
    async def events(ws: WebSocket, game_id: str):
        if not _origin_allowed(ws.headers.get('origin'), ws.headers.get('host', ''), origins):
            await ws.close(code=1008)
            return
        try:
            game = app.state.manager.get(game_id)
        except KeyError:
            await ws.close(code=1008)
            return
        await ws.accept()
        queue = game.subscribe()
        try:
            await ws.send_json(game.snapshot())
            while True:
                packet = await queue.get()
                await ws.send_json(packet)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            game.unsubscribe(queue)

    return app
