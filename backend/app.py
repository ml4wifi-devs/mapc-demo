"""FastAPI app: REST + WebSocket + static frontend.

Run with: uvicorn backend.app:app (see run_demo.sh).
"""

import backend  # noqa: F401 — sets JAX env vars before any JAX import

import asyncio
import logging
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import order defines the method dropdown order in the UI.
from .methods import hmab, optimal, mh, fm, surrogate, dcf, rand  # noqa: F401 — populates the registry
from .methods.base import method_catalog
from .runs import manager
from .scenarios import catalog, scenario_preview

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('demo')

app = FastAPI(title='MAPC Co-SR Demo')

FRONTEND = Path(__file__).parent.parent / 'frontend'

_clients: set[WebSocket] = set()


@app.get('/api/catalog')
def get_catalog():
    return {**catalog(), 'methods': method_catalog()}


@app.post('/api/scenario/preview')
def post_preview(config: dict):
    try:
        return scenario_preview(config)
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


@app.post('/api/run/start')
def post_run_start(payload: dict):
    run_id = manager.start(payload['method'], payload.get('params', {}), payload['scenario'])
    return {'run_id': run_id}


@app.post('/api/run/stop')
def post_run_stop(payload: dict):
    manager.stop(payload.get('run_id'))
    return {'ok': True}


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive pings; content ignored
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


async def _broadcaster():
    while True:
        try:
            msg = await asyncio.to_thread(manager.messages.get, True, 1.0)
        except queue.Empty:
            continue
        dead = []
        for ws in _clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)


@app.on_event('startup')
async def startup():
    asyncio.create_task(_broadcaster())


@app.get('/')
def index():
    return FileResponse(FRONTEND / 'index.html')


app.mount('/static', StaticFiles(directory=FRONTEND), name='static')
