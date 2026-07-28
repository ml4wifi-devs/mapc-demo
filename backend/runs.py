"""Run manager: executes method runners in worker threads and streams
messages to WebSocket clients through a thread-safe queue."""

import logging
import queue
import threading
import traceback
from itertools import count

from .methods.base import METHODS
from .scenarios import build_scenario

log = logging.getLogger('demo')


class RunManager:
    def __init__(self):
        self.messages: queue.Queue = queue.Queue()
        self._runs: dict[int, threading.Event] = {}
        self._ids = count(1)
        self._lock = threading.Lock()

    def start(self, method_id: str, method_params: dict, scenario_config: dict) -> int:
        method = METHODS[method_id]
        run_id = next(self._ids)
        stop_event = threading.Event()

        with self._lock:
            self._runs[run_id] = stop_event

        thread = threading.Thread(
            target=self._worker,
            args=(run_id, method, method_params, scenario_config, stop_event),
            daemon=True,
            name=f'run-{run_id}-{method_id}',
        )
        thread.start()
        return run_id

    def stop(self, run_id: int | None = None) -> None:
        with self._lock:
            targets = [run_id] if run_id is not None else list(self._runs)
            for rid in targets:
                if rid in self._runs:
                    self._runs[rid].set()

    def _emit(self, run_id: int, msg: dict) -> None:
        self.messages.put({'run_id': run_id, **msg})

    def _worker(self, run_id, method, method_params, scenario_config, stop_event):
        emit = lambda msg: self._emit(run_id, msg)
        buffer = []

        def emit_buffered(msg):
            # Batch high-frequency points; pass everything else through immediately.
            if msg['type'] == 'point':
                buffer.append(msg)
                if len(buffer) >= 10:
                    emit({'type': 'points', 'points': [
                        {'step': m['step'], 'thr': m['thr'], 'config': m.get('config')} for m in buffer
                    ]})
                    buffer.clear()
            else:
                emit(msg)

        def flush():
            if buffer:
                emit({'type': 'points', 'points': [
                    {'step': m['step'], 'thr': m['thr'], 'config': m.get('config')} for m in buffer
                ]})
                buffer.clear()

        try:
            emit({'type': 'status', 'state': 'building', 'msg': 'building scenario'})
            scenario = build_scenario(scenario_config)
            emit({'type': 'status', 'state': 'running', 'msg': ''})
            method['run'](scenario, scenario_config.get('globals', {}), method_params, emit_buffered, stop_event)
            flush()
            state = 'stopped' if stop_event.is_set() else 'done'
            emit({'type': 'status', 'state': state, 'msg': ''})
        except Exception as e:
            log.error('run %s failed:\n%s', run_id, traceback.format_exc())
            flush()
            emit({'type': 'status', 'state': 'error', 'msg': f'{type(e).__name__}: {e}'})
        finally:
            with self._lock:
                self._runs.pop(run_id, None)


manager = RunManager()
