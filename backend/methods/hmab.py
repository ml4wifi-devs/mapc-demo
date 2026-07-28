"""H-MAB / Flat MAB / Random baseline runners (mapc-mab + reinforced-lib)."""

import threading

import jax
import numpy as np
from mapc_mab import MapcAgentFactory
from mapc_research.mab.random_agent import RandomMapcAgentFactory
from reinforced_lib.agents.mab import (
    EGreedy,
    Exp3,
    NormalThompsonSampling,
    Softmax,
    ThompsonSampling,
    UCB,
)

from .base import register, step_config

AGENT_TYPES = {
    'EGreedy': EGreedy,
    'UCB': UCB,
    'Softmax': Softmax,
    'Exp3': Exp3,
    'ThompsonSampling': ThompsonSampling,
    'NormalThompsonSampling': NormalThompsonSampling,
}

# Per-level defaults tuned in mapc_research/mab/default_config.json; Exp3 and
# ThompsonSampling defaults are the library ones.
AGENT_DEFAULTS = {
    'EGreedy': {
        'lvl1': {'e': 0.07, 'optimistic_start': 3.3, 'alpha': 0.7},
        'lvl2': {'e': 0.02, 'optimistic_start': 95.0, 'alpha': 0.9},
        'lvl3': {'e': 0.03, 'optimistic_start': 30.0, 'alpha': 0.85},
        'flat': {'e': 0.03, 'optimistic_start': 0.3, 'alpha': 0.6},
    },
    'UCB': {
        'lvl1': {'c': 1.5, 'gamma': 0.5},
        'lvl2': {'c': 0.5, 'gamma': 0.5},
        'lvl3': {'c': 0.2, 'gamma': 0.8},
        'flat': {'c': 2.5, 'gamma': 0.25},
    },
    'Softmax': {
        'lvl1': {'lr': 5.2, 'tau': 1.3, 'alpha': 0.35},
        'lvl2': {'lr': 0.3, 'tau': 0.35, 'alpha': 0.95},
        'lvl3': {'lr': 3.3, 'tau': 0.75, 'alpha': 0.2},
        'flat': {'lr': 2.9, 'tau': 0.15, 'alpha': 0.85},
    },
    'Exp3': {
        'lvl1': {'gamma': 0.15, 'min_reward': 0.0, 'max_reward': 1.0},
        'lvl2': {'gamma': 0.15, 'min_reward': 0.0, 'max_reward': 1.0},
        'lvl3': {'gamma': 0.15, 'min_reward': 0.0, 'max_reward': 1.0},
        'flat': {'gamma': 0.15, 'min_reward': 0.0, 'max_reward': 1.0},
    },
    'ThompsonSampling': {
        'lvl1': {'decay': 2.0},
        'lvl2': {'decay': 2.0},
        'lvl3': {'decay': 2.0},
        'flat': {'decay': 2.0},
    },
    'NormalThompsonSampling': {
        'lvl1': {'alpha': 6.4, 'beta': 0.9, 'lam': 1.0, 'mu': 0.5},
        'lvl2': {'alpha': 5.0, 'beta': 0.5, 'lam': 1.0, 'mu': 1.5},
        'lvl3': {'alpha': 8.5, 'beta': 0.5, 'lam': 1.0, 'mu': 1.0},
        'flat': {'alpha': 9.0, 'beta': 0.07, 'lam': 1.0, 'mu': 0.2},
    },
}

AGENT_PARAM_TOOLTIPS = {
    'e': 'Exploration probability ε — chance of trying a random arm.',
    'e_min': 'Lower bound on ε when decaying.',
    'e_decay': 'Multiplicative ε decay per step.',
    'optimistic_start': 'Initial optimistic value of every arm — encourages early exploration.',
    'alpha': 'For EGreedy/Softmax: recency weight of the reward average (0 = plain average). '
             'For NormalThompsonSampling: prior inverse-gamma shape.',
    'c': 'UCB exploration degree — higher c explores more.',
    'gamma': 'For UCB: discount factor for non-stationary environments. For Exp3: exploration rate.',
    'lr': 'Softmax gradient step size.',
    'tau': 'Softmax temperature — higher τ means more uniform (exploratory) sampling.',
    'multiplier': 'Softmax preference multiplier.',
    'min_reward': 'Smallest reward used for Exp3 normalization.',
    'max_reward': 'Largest reward used for Exp3 normalization.',
    'decay': 'Exponential decay of past observations (Thompson sampling).',
    'beta': 'Prior inverse-gamma scale.',
    'lam': 'Prior precision scaling λ.',
    'mu': 'Prior mean of the reward distribution.',
}


# mapc-mab and mapc_research's random agent pick the sharing AP and its station
# with the *process-global* numpy RNG (and reseed it in the agent factory). The
# demo runs several methods concurrently in one process, so runs would consume one
# shared stream and a newly started run would reseed a running one. Each run
# therefore keeps its own stream and swaps it into the global RNG around every
# call into those agents, which keeps a run's results identical whether it runs
# alone or next to others.
_np_rng_lock = threading.Lock()


class _NumpyRngStream:
    def __init__(self, seed):
        self._state = None
        self.call(np.random.seed, seed)

    def call(self, fn, *args, **kwargs):
        with _np_rng_lock:
            outer = np.random.get_state()
            if self._state is not None:
                np.random.set_state(self._state)
            try:
                return fn(*args, **kwargs)
            finally:
                self._state = np.random.get_state()
                np.random.set_state(outer)


def _mab_loop(scenario, make_factory, n_steps, seed, emit, stop_event):
    key = jax.random.PRNGKey(seed)
    rng = _NumpyRngStream(seed)
    agent = rng.call(lambda: make_factory().create_mapc_agent())
    scenario.reset()
    reward = 0.0

    for step in range(n_steps):
        if stop_event.is_set():
            return
        key, scenario_key = jax.random.split(key)
        tx, tx_power_levels = rng.call(agent.sample, reward)
        thr, reward, internals = scenario(scenario_key, tx, tx_power_levels, return_internals=True)
        config = step_config(scenario, tx, tx_power_levels, internals.mcs)
        emit({'type': 'point', 'step': step, 'thr': float(thr), 'config': config})


def _run_mab(hierarchical):
    def run(scenario, globals_cfg, params, emit, stop_event):
        agent_name = params.get('agent_type', 'UCB')
        agent_type = AGENT_TYPES[agent_name]
        defaults = AGENT_DEFAULTS[agent_name]
        seed = int(params.get('seed', 42))

        if hierarchical:
            lvl1 = {**defaults['lvl1'], **(params.get('params_lvl1') or {})}
            lvl2 = {**defaults['lvl2'], **(params.get('params_lvl2') or {})}
            lvl3 = {**defaults['lvl3'], **(params.get('params_lvl3') or {})}
        else:
            # The flat action space grows as (1 + n_sta·L)^(n_ap−1) — guard
            # against configurations that would exhaust memory.
            levels = int(globals_cfg.get('tx_power_levels', 4))
            n_ap = len(scenario.associations)
            max_sta = max(len(s) for s in scenario.associations.values())
            if (1 + max_sta * levels) ** (n_ap - 1) > 200_000:
                raise ValueError(
                    f'Flat MAB action space is too large for this network ({n_ap} APs, '
                    f'up to {max_sta} STAs each, {levels} power levels). Use H-MAB or a smaller topology.'
                )
            lvl1 = {**defaults['flat'], **(params.get('params_lvl1') or {})}
            lvl2 = lvl3 = None

        def make_factory():
            return MapcAgentFactory(
                scenario.associations,
                agent_type=agent_type,
                agent_params_lvl1=lvl1,
                agent_params_lvl2=lvl2,
                agent_params_lvl3=lvl3,
                hierarchical=hierarchical,
                tx_power_levels=int(globals_cfg.get('tx_power_levels', 4)),
                seed=seed,
            )

        _mab_loop(scenario, make_factory, int(globals_cfg.get('n_steps', 600)), seed, emit, stop_event)

    return run


def _run_random(scenario, globals_cfg, params, emit, stop_event):
    seed = int(params.get('seed', 42))
    _mab_loop(scenario, lambda: RandomMapcAgentFactory(scenario.associations, seed=seed),
              int(globals_cfg.get('n_steps', 600)), seed, emit, stop_event)


def _p(name, label, type_, default, tooltip, **extra):
    return {'name': name, 'label': label, 'type': type_, 'default': default, 'tooltip': tooltip, **extra}


_COMMON = [
    _p('seed', 'Seed', 'number', 42, 'Random seed of the agent and the channel realizations. '
       'The number of steps is the scenario time horizon (Scenario tab).', min=0, max=100_000, step=1),
]

_AGENT_SELECT = _p(
    'agent_type', 'MAB agent', 'select', 'UCB',
    'Bandit algorithm used by every level of the hierarchy. UCB balances exploration via confidence '
    'bounds, ε-greedy explores randomly, Softmax samples proportionally to estimated value, '
    'Thompson sampling maintains a posterior over rewards.',
    options=list(AGENT_TYPES.keys()),
)

register({
    'id': 'hmab',
    'name': 'H-MAB',
    'kind': 'curve',
    'description': 'Hierarchical multi-armed bandits: level 1 selects the set of Co-SR APs, '
                   'level 2 the recipient stations, level 3 the transmit powers.',
    'params': [_AGENT_SELECT, *_COMMON],
    'agent_defaults': AGENT_DEFAULTS,
    'agent_param_tooltips': AGENT_PARAM_TOOLTIPS,
    'levels': ['lvl1', 'lvl2', 'lvl3'],
    'run': _run_mab(hierarchical=True),
})

register({
    'id': 'fmab',
    'name': 'Flat MAB',
    'kind': 'curve',
    'description': 'Single-level MAB baseline: one bandit over the full joint action space '
                   '(all AP-STA pair subsets × power levels). Converges slower than H-MAB and the action '
                   'space explodes combinatorially — only feasible for small networks (roughly ≤4 APs).',
    'params': [_AGENT_SELECT, *_COMMON],
    'agent_defaults': AGENT_DEFAULTS,
    'agent_param_tooltips': AGENT_PARAM_TOOLTIPS,
    'levels': ['lvl1'],
    'run': _run_mab(hierarchical=False),
})

# The Random baseline is registered in rand.py so it lands at the bottom of the method list.
