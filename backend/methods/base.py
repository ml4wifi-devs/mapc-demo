"""Method registry. Each method module registers:

{
  'id': str, 'name': str, 'kind': 'curve' | 'hline',
  'description': str, 'params': [param schema], 'run': callable
}

run(scenario, globals_cfg, params, emit, stop_event) executes in a worker
thread; emit(msg: dict) streams messages to the frontend, stop_event is a
threading.Event checked cooperatively.
"""

METHODS: dict[str, dict] = {}


def register(method: dict) -> None:
    METHODS[method['id']] = method


def method_catalog() -> list[dict]:
    return [
        {k: v for k, v in m.items() if k != 'run'}
        for m in METHODS.values()
    ]
