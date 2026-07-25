"""DCF / 802.11ax Spatial Reuse baselines (mapc-dcf discrete event simulator)."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import simpy
from mapc_dcf import AccessPoint, Channel, Logger
from mapc_sim.constants import TAU

from .base import register


def _run_single(key, run_number, scenario, sr, tx_power, sim_time, logger):
    key, key_channel = jax.random.split(key)
    des_env = simpy.Environment()
    channel = Channel(key_channel, sr, scenario.channel_width, scenario.pos, scenario.walls)

    for ap in scenario.associations:
        key, key_ap = jax.random.split(key)
        clients = jnp.array(scenario.associations[ap])
        ap_node = AccessPoint(key_ap, int(ap), scenario.pos, tx_power, clients, channel, des_env, logger)
        ap_node.start_operation(run_number)

    des_env.run(until=(logger.warmup_length + sim_time))
    logger.dump_acumulators(run_number)


def _run_dcf(sr):
    def run(scenario, globals_cfg, params, emit, stop_event):
        # Same time horizon as the step-based methods: n_steps TXOPs of channel time.
        sim_time = int(globals_cfg.get('n_steps', 600)) * TAU
        warmup = float(params.get('warmup', 0.1))
        n_runs = int(params.get('n_runs', 1))
        seed = int(params.get('seed', 42))
        tx_power = float(np.max(np.asarray(scenario.tx_power)))

        key = jax.random.PRNGKey(seed)

        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, 'dcf')
            logger = Logger(sim_time, warmup_length=warmup, results_path=results_path)

            for run_number, k in enumerate(jax.random.split(key, n_runs), start=1):
                if stop_event.is_set():
                    logger.shutdown({})
                    return
                emit({'type': 'status_msg',
                      'msg': f'run {run_number}/{n_runs}: simulating {sim_time:.2f} s of channel time (may take minutes)…'})
                _run_single(k, run_number, scenario, sr, tx_power, sim_time, logger)
                emit({'type': 'status_msg', 'msg': f'run {run_number}/{n_runs} finished'})

            logger.shutdown({})
            df = pd.read_csv(results_path + '.csv')

        df = df[df.Collision == False]  # noqa: E712 — successful frames only
        rates = (df.groupby('RunNumber')['AMPDUSize'].sum() * 1e-6 / sim_time).reindex(
            range(1, n_runs + 1), fill_value=0.0)
        mean = float(rates.mean())
        std = float(rates.std(ddof=1)) if n_runs > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n_runs) if n_runs > 1 else 0.0
        emit({'type': 'hline', 'value': mean, 'ci_low': mean - ci, 'ci_high': mean + ci})

    return run


def _p(name, label, type_, default, tooltip, **extra):
    return {'name': name, 'label': label, 'type': type_, 'default': default, 'tooltip': tooltip, **extra}


_PARAMS = [
    _p('warmup', 'Warmup [s]', 'number', 0.1, 'Initial simulated time excluded from the statistics. '
       'The simulated channel time itself equals the scenario time horizon (n_steps × TXOP ≈ 5.5 ms each). '
       'Warning: this is a detailed per-frame discrete event simulator — expect minutes of computation '
       'per simulated second even for small topologies.',
       min=0.0, max=10.0, step=0.1),
    _p('n_runs', 'Independent runs', 'number', 1,
       'Number of independent simulation runs used to compute the mean and confidence interval.',
       min=1, max=10, step=1),
    _p('seed', 'Seed', 'number', 42, 'Random seed.', min=0, max=100_000, step=1),
]

register({
    'id': 'dcf',
    'name': 'DCF (legacy 802.11)',
    'kind': 'hline',
    'description': 'Legacy 802.11 operation: CSMA/CA with binary exponential backoff, no coordination, '
                   'full transmit power. Shown as a horizontal line (mean over runs with 95% CI). '
                   'SLOW: per-frame discrete event simulation, and the cost follows the simulated channel '
                   'time rather than the step count. Keep the time horizon short.',
    'params': _PARAMS,
    'run': _run_dcf(sr=False),
})

register({
    'id': 'sr',
    'name': 'SR (802.11ax)',
    'kind': 'hline',
    'description': '802.11ax OBSS/PD Spatial Reuse: uncoordinated parallel transmissions allowed when the '
                   'detected OBSS signal is below the packet-detect threshold. Shown as a horizontal line. '
                   'As slow as DCF — same per-frame simulation. Keep the time horizon short.',
    'params': _PARAMS,
    'run': _run_dcf(sr=True),
})
