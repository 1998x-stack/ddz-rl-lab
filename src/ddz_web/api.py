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
from .lab import LabStore, LabError
from .auth import OperatorAuth, COOKIE
from .debug_hands import hand_debug_payload

STATIC_DIR = Path(__file__).resolve().parent / 'static'
NO_CACHE = {'Cache-Control': 'no-store'}


class NewGame(BaseModel):
    model_ids: tuple[str, str, str]
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    delay: float = Field(.7, ge=.05, le=3.0)
    random_bidding: bool = False


class Control(BaseModel):
    action: Literal['play', 'pause', 'step', 'speed']
    delay: float | None = None


class HumanAction(BaseModel):
    index: int
    expected_step: int


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
               allowed_hosts: list[str] | None = None,
               runs_dir: str | Path = 'runs',
               reports_dir: str | Path = 'runs/reports',
               log_dir: str | Path = 'logs',
               jobs_dir: str | Path | None = None,
               operator_token: str | None = None,
               secure_auth_cookie: bool = False,
               allow_debug_hands: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.manager.close()

    app = FastAPI(lifespan=lifespan, title='DDZ RL Lab Arena', version='0.10.0',
                  description='Local no-stakes, registered-checkpoint spectator arena')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or [
        'localhost', '127.0.0.1', 'testserver'])
    app.state.manager = ArenaManager(Path(registry_dir), Path(replay_dir),
                                     allow_baseline=allow_baseline, max_sessions=max_sessions)
    app.state.lab = LabStore(Path(runs_dir), Path(reports_dir), Path(replay_dir), Path(log_dir),
                             jobs_dir=Path(jobs_dir) if jobs_dir else None)
    origins = trusted_origins or set()
    auth = OperatorAuth(operator_token, secure_cookie=secure_auth_cookie)

    @app.middleware('http')
    async def same_origin_write(request: Request, call_next):
        if request.url.path.startswith('/api/') and request.url.path not in {
            '/api/health', '/api/auth/status', '/api/auth/login'}:
            if not auth.valid_cookie(request.cookies.get(COOKIE)):
                return JSONResponse(status_code=401, content={'detail':'operator login required'})
        if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            if not _origin_allowed(request.headers.get('origin'),
                                   request.headers.get('host', ''), origins):
                return JSONResponse(
                    status_code=403, content={'detail': 'cross-origin writes blocked'})
        return await call_next(request)

    @app.get('/')
    def page():
        return FileResponse(STATIC_DIR / 'index.html', headers=NO_CACHE)

    @app.get('/static/app.js')
    def javascript():
        return FileResponse(STATIC_DIR / 'app.js', media_type='text/javascript', headers=NO_CACHE)

    @app.get('/static/style.css')
    def stylesheet():
        return FileResponse(STATIC_DIR / 'style.css', media_type='text/css', headers=NO_CACHE)

    @app.get('/static/auth.js')
    def auth_js():
        return FileResponse(STATIC_DIR / 'auth.js', media_type='text/javascript', headers=NO_CACHE)

    @app.get('/static/lab.js')
    def lab_javascript():
        return FileResponse(STATIC_DIR / 'lab.js', media_type='text/javascript', headers=NO_CACHE)

    @app.get('/api/lab/overview')
    def overview():
        models = app.state.manager.available_models()
        return {'schema': 'ddz-lab-v1', 'runs': app.state.lab.runs(),
                'reports': app.state.lab.reports(), 'replays': app.state.lab.replays(),
                'jobs': app.state.lab.jobs(), 'matrices': app.state.lab.matrices(),
                'models': models}

    @app.get('/api/lab/runs')
    def runs():
        return {'runs': app.state.lab.runs()}

    @app.get('/api/lab/runs/{run_id}/metrics')
    def metrics(run_id: str, limit: int = 300):
        if not 1 <= limit <= 500:
            raise HTTPException(422, detail='limit out of range')
        try:
            return app.state.lab.metrics(run_id, limit=limit)
        except (LabError, FileNotFoundError) as exc:
            raise HTTPException(404, detail='invalid run or missing run') from exc

    @app.get('/api/lab/compare')
    def compare_runs(left: str, right: str):
        try:
            return app.state.lab.compare_runs(left, right)
        except (LabError, FileNotFoundError) as exc:
            raise HTTPException(404, detail='invalid run comparison') from exc

    @app.get('/api/lab/jobs')
    def job_records():
        return {'jobs': app.state.lab.jobs()}

    @app.get('/api/lab/matrices')
    def matrix_reports():
        return {'matrices': app.state.lab.matrices()}

    @app.get('/api/lab/matrices/{report_id}')
    def matrix_report(report_id: str):
        try:
            return app.state.lab.matrix(report_id)
        except (LabError, FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(404, detail='matrix report unavailable') from exc

    @app.get('/api/lab/reports')
    def reports():
        return {'reports': app.state.lab.reports()}

    @app.get('/api/lab/reports/{report_id}')
    def report(report_id: str):
        try:
            return app.state.lab.report(report_id)
        except (LabError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, detail='report unavailable') from exc

    @app.get('/api/lab/replays')
    def replays():
        return {'replays': app.state.lab.replays()}

    @app.get('/api/lab/replays/{replay_id}')
    def replay_frames(replay_id: str):
        try:
            return app.state.lab.replay(replay_id)
        except (LabError, FileNotFoundError, ValueError, AssertionError, KeyError) as exc:
            logger.bind(component='replay', replay_id=replay_id).warning('replay_validation_failed')
            raise HTTPException(404, detail='completed replay not available or invalid') from exc

    @app.get('/api/lab/logs')
    def logs(limit: int = 120):
        if not 1 <= limit <= 200:
            raise HTTPException(422, detail='limit out of range')
        return app.state.lab.logs(limit=limit)

    @app.get('/api/auth/status')
    def auth_status(request: Request):
        return {'required': auth.enabled, 'authenticated': auth.valid_cookie(request.cookies.get(COOKIE))}

    @app.post('/api/auth/login')
    def auth_login(request: Request):
        if not auth.enabled:
            raise HTTPException(400, detail='operator login is not enabled')
        if not auth.login(request.headers.get('x-operator-token')):
            logger.bind(component='auth').warning('operator_login_rejected')
            raise HTTPException(401, detail='invalid operator token')
        response = JSONResponse(content={'authenticated': True})
        response.set_cookie(COOKIE, auth.issue_cookie(), httponly=True, secure=auth.secure_cookie,
                            samesite='strict', max_age=3600 * 8, path='/')
        logger.bind(component='auth').info('operator_login_success')
        return response

    @app.post('/api/auth/logout')
    def auth_logout():
        response = JSONResponse(content={'authenticated': False})
        response.delete_cookie(COOKIE, path='/')
        return response

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'schema': 'ddz-arena-v1'}

    @app.get('/static/handview.js')
    def handview_javascript():
        return FileResponse(STATIC_DIR / 'handview.js', media_type='text/javascript', headers=NO_CACHE)

    def local_debug_allowed(request: Request) -> bool:
        # Reject remote peers; Origin and Host are never authentication.
        peer = request.client.host if request.client else ''
        return bool(allow_debug_hands and peer in {'127.0.0.1', '::1', 'testclient'})

    @app.get('/api/debug/config')
    def debug_config(request: Request):
        return {'enabled': local_debug_allowed(request)}

    @app.get('/api/games/{game_id}/debug/hands')
    def debug_hands(request: Request, game_id: str):
        if not local_debug_allowed(request):
            raise HTTPException(404, detail='debug hands unavailable')
        return JSONResponse(hand_debug_payload(get_game(game_id)),
                            headers={'Cache-Control': 'private, no-store'})

    @app.get('/api/games/{game_id}/spectator/hands')
    def spectator_hands(game_id: str):
        game = get_game(game_id)
        if '@human' in game.policy_ids:
            raise HTTPException(403, detail='opponent hands unavailable in human games')
        payload = hand_debug_payload(game)
        payload['schema'] = 'ddz-spectator-hands-v1'
        return JSONResponse(payload,
                            headers={'Cache-Control': 'private, no-store'})

    @app.get('/api/models')
    def models():
        return app.state.manager.available_models()

    @app.post('/api/games', status_code=201)
    def create(data: NewGame):
        try:
            return app.state.manager.create(data.model_ids, seed=data.seed,
                                            delay=data.delay,
                                            random_bidding=data.random_bidding).snapshot()
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

    @app.get('/api/games/{game_id}/hand')
    def hand(game_id: str):
        try:
            return JSONResponse(get_game(game_id).human_view(),
                                headers={'Cache-Control': 'private, no-store'})
        except ValueError as exc:
            raise HTTPException(404, detail=str(exc)) from exc

    @app.post('/api/games/{game_id}/action')
    async def human_action(game_id: str, data: HumanAction):
        try:
            return await get_game(game_id).human_action(data.index, data.expected_step)
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

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
        if not (auth.valid_cookie(ws.cookies.get(COOKIE)) and
                _origin_allowed(ws.headers.get('origin'), ws.headers.get('host', ''), origins)):
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
