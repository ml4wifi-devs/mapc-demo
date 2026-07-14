"""Metaheuristic approximate upper bounds (mapc-mh).

T-Meta approximates T-Optimal (max total throughput) with SA / RRHC / Tabu;
F-Meta approximates F-Optimal (max-min fairness) with SA / VNS / CG.
"""

from mapc_mh.methods.throughput import rrhc as t_rrhc
from mapc_mh.methods.throughput import sa as t_sa
from mapc_mh.methods.throughput import tabu as t_tabu
from mapc_mh.methods.fairness import cg as f_cg

from .base import register


def _p(name, label, type_, default, tooltip, **extra):
    return {'name': name, 'label': label, 'type': type_, 'default': default, 'tooltip': tooltip, **extra}


_SEED = _p('seed', 'Seed', 'number', 42,
           'Random seed. The number of search iterations is the scenario time horizon (Scenario tab) — '
           'each iteration evaluates candidate configurations in the simulator.',
           min=0, max=100_000, step=1)


def _emit_curve(history, emit, step_scale=1):
    for step, thr in enumerate(history, start=1):
        emit({'type': 'point', 'step': step * step_scale, 'thr': float(thr)})


def _run_t_meta(scenario, globals_cfg, params, emit, stop_event):
    algorithm = params.get('algorithm', 'sa')
    seed = int(params.get('seed', 42))
    n_steps = int(globals_cfg.get('n_steps', 600))

    if algorithm == 'sa':
        result = t_sa.run(scenario, seed=seed, n_steps=n_steps,
                          T_decay=float(params.get('t_decay', 0.999)))
    elif algorithm == 'rrhc':
        result = t_rrhc.run(scenario, seed=seed, n_steps=n_steps,
                            restart_threshold=int(params.get('restart_threshold', 100)))
    else:
        result = t_tabu.run(scenario, seed=seed, n_steps=n_steps,
                            tabu_size=int(params.get('tabu_size', 20)),
                            n_candidates=int(params.get('n_candidates', 10)))

    _emit_curve(result.best_history, emit)
    emit({'type': 'status_msg', 'msg': f'best rate: {float(result.best_rate):.1f} Mb/s'})


def _run_f_meta(scenario, globals_cfg, params, emit, stop_event):
    seed = int(params.get('seed', 42))
    n_steps = int(globals_cfg.get('n_steps', 600))

    # CG counts outer iterations; each spends inner_steps pricing evaluations,
    # so the shared horizon is split between the two to stay comparable.
    inner_steps = int(params.get('inner_steps', 100))
    result = f_cg.run(scenario, seed=seed,
                      n_steps=max(1, n_steps // inner_steps),
                      inner_steps=inner_steps,
                      tabu_size=int(params.get('tabu_size', 20)),
                      n_candidates=int(params.get('n_candidates', 10)))

    # The common chart shows aggregate throughput; fairness itself is reported below.
    _emit_curve(result.sum_history, emit, inner_steps)
    emit({'type': 'status_msg',
          'msg': f'min rate: {float(result.best_min_rate):.1f} Mb/s, '
                 f'total: {float(result.best_sum_rate):.1f} Mb/s, '
                 f"Jain's index: {float(result.best_fairness):.3f}"})


register({
    'id': 't_meta',
    'name': 'T-Meta',
    'kind': 'curve',
    'description': 'Metaheuristic search over Co-SR configurations (AP-STA selection, MCS, TX power) that '
                   'approximates the T-Optimal bound. Plots the best total rate found so far vs search step.',
    'params': [
        _p('algorithm', 'Algorithm', 'select', 'sa',
           'SA: simulated annealing with auto-calibrated initial temperature. RRHC: hill climbing with random '
           'restarts after stagnation. Tabu: evaluates several neighbors per step and forbids recently visited '
           'configurations.',
           options=[{'value': 'sa', 'label': 'Simulated Annealing'},
                    {'value': 'rrhc', 'label': 'Random-Restart Hill Climbing'},
                    {'value': 'tabu', 'label': 'Tabu Search'}]),
        _SEED,
        _p('t_decay', 'Temperature decay (SA)', 'number', 0.999,
           'SA only: multiplicative cooling factor per step.', min=0.9, max=1.0, step=0.001),
        _p('restart_threshold', 'Restart threshold (RRHC)', 'number', 100,
           'RRHC only: consecutive non-improving steps before a random restart.', min=10, max=5000, step=10),
        _p('tabu_size', 'Tabu list size (Tabu)', 'number', 20,
           'Tabu only: number of recently visited configurations kept forbidden.', min=1, max=200, step=1),
        _p('n_candidates', 'Candidates per step (Tabu)', 'number', 10,
           'Tabu only: neighbor candidates evaluated per iteration.', min=1, max=50, step=1),
    ],
    'run': _run_t_meta,
})

register({
    'id': 'f_meta',
    'name': 'F-Meta',
    'kind': 'curve',
    'description': 'Column generation that approximates F-Optimal (max-min fairness): iteratively builds a '
                   'time-shared schedule of Co-SR configurations maximizing the worst station\'s rate. The curve '
                   'shows the total rate of the best fair solution; the achieved min per-station rate and '
                   'Jain\'s index appear in the run status.',
    'params': [
        _SEED,
        _p('inner_steps', 'Pricing steps', 'number', 100,
           'Tabu pricing iterations per outer column-generation step. The time horizon is split into '
           'horizon / pricing steps outer iterations.', min=10, max=5000, step=10),
        _p('tabu_size', 'Tabu list size', 'number', 20,
           'Tabu list length used by the pricing search.', min=1, max=200, step=1),
        _p('n_candidates', 'Candidates per step', 'number', 10,
           'Neighbor candidates evaluated per pricing iteration.', min=1, max=50, step=1),
    ],
    'run': _run_f_meta,
})
