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


def step_config(scenario, tx, tx_power_levels, mcs):
    """Active Co-SR links for one step: transmitter, receiver, tx power [dBm], MCS index + rate [Mb/s]."""
    import numpy as np
    from mapc_sim.constants import DATA_RATES

    rates = DATA_RATES[scenario.channel_width]
    tx = np.asarray(tx)
    tx_power_levels = np.asarray(tx_power_levels)
    mcs = np.asarray(mcs)
    links = []
    for i, j in zip(*np.nonzero(tx)):
        m = int(mcs[i])
        links.append({
            'ap': int(i), 'sta': int(j),
            'tx_power': float(scenario.tx_power[i] - scenario.tx_power_delta * tx_power_levels[i]),
            'mcs': m, 'rate': float(rates[m]),
        })
    return {'links': links}
