"""T-Optimal / F-Optimal analytical upper bounds (mapc-optimal, MILP)."""

from itertools import chain

import numpy as np
from mapc_optimal import OptimizationType, Solver, positions_to_path_loss

from .base import register


def _run_optimal(opt_type):
    def run(scenario, globals_cfg, params, emit, stop_event):
        associations = {int(ap): [int(s) for s in stas] for ap, stas in scenario.associations.items()}
        access_points = list(associations.keys())
        stations = list(chain.from_iterable(associations.values()))
        path_loss = positions_to_path_loss(np.asarray(scenario.pos), np.asarray(scenario.walls))

        max_tx_power = float(np.max(np.asarray(scenario.tx_power)))
        levels = int(globals_cfg.get('tx_power_levels', 4))
        delta = float(scenario.tx_power_delta)
        min_tx_power = max_tx_power - delta * (levels - 1)

        solver = Solver(
            stations=stations,
            access_points=access_points,
            channel_width=scenario.channel_width,
            opt_type=opt_type,
            max_tx_power=max_tx_power,
            min_tx_power=min_tx_power,
        )
        _, total_rate = solver(path_loss, associations)
        emit({'type': 'hline', 'value': float(total_rate)})

    return run


_PARAMS = []

register({
    'id': 't_optimal',
    'name': 'T-Optimal',
    'kind': 'hline',
    'description': 'Upper bound on total network throughput from a mixed-integer linear program. '
                   'Assumes perfect channel knowledge and scheduling — no learning involved. '
                   'The solve time grows steeply with the number of APs and stations.',
    'params': _PARAMS,
    'run': _run_optimal(OptimizationType.SUM),
})

register({
    'id': 'f_optimal',
    'name': 'F-Optimal',
    'kind': 'hline',
    'description': 'Analytical optimum of the max-min (fairness) objective: maximizes the worst station '
                   'throughput, then reports the resulting total throughput. Lower than T-Optimal by design. '
                   'The solve time grows steeply with the number of APs and stations.',
    'params': _PARAMS,
    'run': _run_optimal(OptimizationType.MAX_MIN),
})
