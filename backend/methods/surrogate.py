"""MAPC-Surrogate: draw random Co-SR configurations, score them with a GNN
surrogate model in a single batched forward pass, evaluate the best in the
simulator. No learning, no generation, no iterative optimization.

Follows the inference loop of mapc_surrogate.sim (run_scenario) using the
published mapc_surrogate API. The checkpoint is loaded lazily on first use and
cached for the process lifetime.
"""

import os
import threading
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .base import register

import mapc_surrogate

# The checkpoint ships inside the installed mapc_surrogate package.
SURROGATE_CHECKPOINT = os.path.join(os.path.dirname(mapc_surrogate.__file__), 'runs', 'surrogate_base')

_lock = threading.Lock()
_surrogate_fn = None


def _load_model():
    global _surrogate_fn
    with _lock:
        if _surrogate_fn is not None:
            return _surrogate_fn

        import orbax.checkpoint as ocp
        from omegaconf import OmegaConf
        from mapc_surrogate.model import SurrogateModel

        fallback = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        with ocp.CheckpointManager(os.path.abspath(SURROGATE_CHECKPOINT), item_names=['params']) as mgr:
            params = mgr.restore(None, args=ocp.args.Composite(
                params=ocp.args.StandardRestore(fallback_sharding=fallback))).params

        cfg = OmegaConf.load(os.path.join(SURROGATE_CHECKPOINT, 'config.yaml'))
        model = SurrogateModel(**cfg.model)
        _surrogate_fn = jax.jit(partial(model.apply, params, training=False))
        return _surrogate_fn


def _mdn_scores(logits, means, scales, risk_averse, risk_factor):
    # Expected value (optionally risk-adjusted) of the MDN prediction;
    # the last entry is the padding graph added by make_batch and is dropped.
    probs = jax.nn.softmax(logits, axis=-1)
    expected_value = jnp.sum(probs * means, axis=-1)

    if risk_averse:
        second_moment = (probs * (scales ** 2 + means ** 2)).sum(axis=-1)
        variance = second_moment - expected_value ** 2
        std_dev = jnp.sqrt(jnp.maximum(variance, 0.0))
        scores = expected_value - risk_factor * std_dev
    else:
        scores = expected_value

    return scores[:-1]


def _run_surrogate(scenario, globals_cfg, params, emit, stop_event):
    # tx_to_conf lives in mapc_surrogate.sim, but importing that module builds
    # its experiment scenario sets at import time — inline the conversion instead.
    from mapc_surrogate.dataset import Configuration, TxPair, random_tx
    from mapc_surrogate.graphs import conf_to_nx, nx_to_jraph, make_batch

    def tx_to_conf(tx, tx_power, mcs):
        ap, sta = np.where(tx)
        return Configuration([TxPair(a, s, mcs[a].item(), tx_power[a].item()) for a, s in zip(ap, sta)])

    emit({'type': 'status_msg', 'msg': 'loading surrogate checkpoint…'})
    surrogate_fn = _load_model()

    n_samples = int(params.get('n_samples_eval', 128))
    top_k = int(params.get('top_k', 1))
    risk_averse = bool(params.get('risk_averse', True))
    seed = int(params.get('seed', 42))
    batch_size = 128
    n_steps = int(globals_cfg.get('n_steps', 600))

    key = jax.random.PRNGKey(seed)

    with jax.default_device(jax.devices('cpu')[0]):
        for step in range(n_steps):
            if stop_event.is_set():
                return

            # 1. Draw N random Co-SR configurations and encode each as a graph.
            candidate_txs, candidate_graphs = [], []
            for _ in range(n_samples):
                key, tx_key = jax.random.split(key)
                tx, tx_power, mcs = random_tx(tx_key, scenario)
                candidate_txs.append((tx, tx_power, mcs))
                candidate_graphs.append(nx_to_jraph(conf_to_nx(scenario, tx_to_conf(tx, tx_power, mcs))))

            # 2. Score all candidates with the surrogate model (batched forward pass).
            scores = []
            for i in range(0, len(candidate_graphs), batch_size):
                batch = make_batch(candidate_graphs[i:i + batch_size])
                logits, means, scales = surrogate_fn(batch)
                scores.extend(np.asarray(_mdn_scores(logits, means, scales, risk_averse, 2.5)).tolist())
            scores = np.asarray(scores[:n_samples])

            # 3. Evaluate the top-k candidates in the simulator, report the best one.
            key, eval_key = jax.random.split(key)
            best_thr = 0.0
            for idx in np.argsort(-scores)[:top_k]:
                tx, tx_power, _ = candidate_txs[idx]
                data_rate, _ = scenario(eval_key, tx, tx_power)  # mcs=None → ideal MCS
                best_thr = max(best_thr, float(data_rate))

            emit({'type': 'point', 'step': step + 1, 'thr': best_thr})


def _p(name, label, type_, default, tooltip, **extra):
    return {'name': name, 'label': label, 'type': type_, 'default': default, 'tooltip': tooltip, **extra}


register({
    'id': 'mapc_surrogate',
    'name': 'MAPC-Surrogate',
    'kind': 'curve',
    'description': 'Surrogate-only scheduler: draws random Co-SR configurations, scores them all in one '
                   'batched forward pass of a GNN surrogate model (MDN head), and evaluates the best ones '
                   'in the simulator. No learning, no generative model, no iterative optimization. '
                   'Note: trained on specific topology distributions — custom hand-drawn topologies may be '
                   'out-of-distribution.',
    'params': [
        _p('n_samples_eval', 'Candidates per step', 'number', 128,
           'Number of random configurations drawn and scored by the surrogate per step.', min=8, max=1024, step=8),
        _p('top_k', 'Top-k evaluated', 'number', 1,
           'How many of the highest-scored candidates are evaluated in the full simulator.', min=1, max=64, step=1),
        _p('risk_averse', 'Risk-averse scoring', 'checkbox', True,
           'Penalize candidates with high predicted variance (expected value − 2.5·std) instead of '
           'ranking by expected value alone.'),
        _p('seed', 'Seed', 'number', 42, 'Random seed.', min=0, max=100_000, step=1),
    ],
    'run': _run_surrogate,
})
