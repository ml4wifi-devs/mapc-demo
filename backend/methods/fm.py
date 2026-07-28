"""FM4WiFi generative pipeline (lai4wifi): GNN autoencoder + flow matching + surrogate.

Follows the inference loop of lai4wifi.flow_matching.sim (run_scenario), using
only the published lai4wifi API. Checkpoints are loaded lazily on first use and
cached for the process lifetime.
"""

import os
import threading
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .base import register, step_config

import lai4wifi

# Checkpoints ship inside the installed lai4wifi package.
LAI4WIFI_ROOT = os.path.dirname(lai4wifi.__file__)
FM_CHECKPOINT = os.path.join(LAI4WIFI_ROOT, 'flow_matching', 'runs', 'fm_medium')
SURROGATE_CHECKPOINT = os.path.join(LAI4WIFI_ROOT, 'surrogate', 'runs', 'sur_medium')

_lock = threading.Lock()
_models = None


def _load_models():
    global _models
    with _lock:
        if _models is not None:
            return _models

        import orbax.checkpoint as ocp
        from omegaconf import OmegaConf
        from lai4wifi.flow_matching.model import FlowMatching
        from lai4wifi.graphs.model import GnnAutoencoder
        from lai4wifi.surrogate.model import SurrogateModel

        fallback = jax.sharding.SingleDeviceSharding(jax.devices()[0])

        fm_cfg = OmegaConf.load(os.path.join(FM_CHECKPOINT, 'config.yaml'))
        with ocp.CheckpointManager(os.path.abspath(FM_CHECKPOINT), item_names=['ema_params']) as mgr:
            fm_params = mgr.restore(None, args=ocp.args.Composite(
                ema_params=ocp.args.StandardRestore(fallback_sharding=fallback))).ema_params
        fm = FlowMatching(**fm_cfg.model)
        gen_fn = jax.jit(partial(fm.apply, fm_params, method='gen'))

        gnn_path = fm_cfg.gnn_path
        if not os.path.isabs(gnn_path):
            gnn_path = os.path.normpath(os.path.join(LAI4WIFI_ROOT, 'flow_matching', gnn_path))
        gnn_cfg = OmegaConf.load(os.path.join(gnn_path, 'config.yaml'))
        with ocp.CheckpointManager(os.path.abspath(gnn_path), item_names=['params']) as mgr:
            gnn_params = mgr.restore(None, args=ocp.args.Composite(
                params=ocp.args.StandardRestore(fallback_sharding=fallback))).params
        gnn = GnnAutoencoder(**gnn_cfg.model)
        encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
        decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

        s_cfg = OmegaConf.load(os.path.join(SURROGATE_CHECKPOINT, 'config.yaml'))
        with ocp.CheckpointManager(os.path.abspath(SURROGATE_CHECKPOINT), item_names=['params']) as mgr:
            s_params = mgr.restore(None, args=ocp.args.Composite(
                params=ocp.args.StandardRestore(fallback_sharding=fallback))).params
        surrogate = SurrogateModel(**s_cfg.model)
        surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))

        _models = {
            'gen_fn': gen_fn,
            'encode_fn': encode_fn,
            'decode_fn': decode_fn,
            'surrogate_fn': surrogate_fn,
            'n_timesteps': int(fm_cfg.train.n_timesteps),
        }
        return _models


def _run_fm(scenario, globals_cfg, params, emit, stop_event):
    import jraph
    from lai4wifi.flow_matching.utils import connections_to_tx, random_tx, tx_to_jraph
    from lai4wifi.graphs.utils import make_batch, unbatch
    from lai4wifi.surrogate.utils import select_best_configuration

    emit({'type': 'status_msg', 'msg': 'loading FM4WiFi checkpoints…'})
    models = _load_models()
    gen_fn, encode_fn, decode_fn = models['gen_fn'], models['encode_fn'], models['decode_fn']
    surrogate_fn = models['surrogate_fn']
    n_timesteps = models['n_timesteps']

    n_samples = int(params.get('n_samples_eval', 64))
    top_k = int(params.get('top_k', 8))
    n_eval_repeats = int(params.get('n_eval_repeats', 5))
    use_simulator = bool(params.get('use_simulator', False))
    batch_size = 64
    seed = int(params.get('seed', 42))

    # Shared time horizon: each observation-generation round consumes n_timesteps
    # TXOPs of channel probing, so the chart x-axis stays comparable across methods.
    horizon = int(globals_cfg.get('n_steps', 600))
    n_rounds = max(1, horizon // n_timesteps)

    key = jax.random.PRNGKey(seed)

    with jax.default_device(jax.devices('cpu')[0]):
        for step in range(n_rounds):
            if stop_event.is_set():
                return

            # 1. Observation context: random probing transmissions encoded to latents.
            ctx_graphs = []
            for _ in range(n_timesteps):
                key, tx_key, sim_key = jax.random.split(key, 3)
                tx = random_tx(tx_key, scenario)
                _, _, internals = scenario(sim_key, *tx, return_internals=True)
                ctx_graphs.append(tx_to_jraph(scenario, *tx, internals))

            encoded = unbatch(encode_fn(make_batch(ctx_graphs)))
            stacked = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *encoded)
            context = encoded[0]._replace(edges=stacked.edges, nodes=stacked.nodes, globals=stacked.globals)
            fm_inputs = [context] * n_samples

            # 2. Generate candidate Co-SR configurations with the flow-matching model.
            all_txs, all_z = [], []
            for i in range(0, len(fm_inputs), batch_size):
                key, fm_key = jax.random.split(key)
                z = gen_fn(make_batch(fm_inputs[i:i + batch_size]), rngs=fm_key)
                all_z.extend(unbatch(z))
                for zt in unbatch(decode_fn(z)):
                    key, tx_key = jax.random.split(key)
                    all_txs.append(connections_to_tx(scenario.associations, tx_key, zt))

            # 3. Score candidates: surrogate model (fast) or full simulator.
            scores = []
            if use_simulator:
                for tx in all_txs:
                    key, sim_key = jax.random.split(key)
                    data_rate, _, _ = scenario(sim_key, *tx, return_internals=True)
                    scores.append(float(data_rate))
            else:
                for i in range(0, len(all_z), batch_size):
                    batch = make_batch(all_z[i:i + batch_size])
                    pred = surrogate_fn(batch)
                    mask = jraph.get_graph_padding_mask(batch)
                    logits, means, scales = jax.tree.map(lambda x: x[mask], pred)
                    _, s = select_best_configuration(logits, means, scales)
                    scores.extend(np.asarray(s).tolist())

            # 4. Evaluate top-k candidates in the simulator, report the best one.
            best_thr, best_config = 0.0, {'links': []}
            for idx in np.argsort(-np.asarray(scores))[:top_k]:
                repeats, last_internals = [], None
                for _ in range(n_eval_repeats):
                    key, sim_key = jax.random.split(key)
                    data_rate, _, last_internals = scenario(sim_key, *all_txs[idx], return_internals=True)
                    repeats.append(float(data_rate))
                mean_thr = float(np.mean(repeats))
                if mean_thr >= best_thr:
                    best_thr = mean_thr
                    best_config = step_config(scenario, all_txs[idx][0], all_txs[idx][1], last_internals.mcs)

            emit({'type': 'point', 'step': (step + 1) * n_timesteps, 'thr': best_thr, 'config': best_config})


def _p(name, label, type_, default, tooltip, **extra):
    return {'name': name, 'label': label, 'type': type_, 'default': default, 'tooltip': tooltip, **extra}


register({
    'id': 'fm4wifi',
    'name': 'FM4WiFi',
    'kind': 'curve',
    'description': 'Generative model: a GNN encoder embeds the observed network state, a flow-matching model '
                   'proposes candidate Co-SR configurations, and a surrogate model pre-selects the best ones. '
                   'SLOW: generation plus top-k simulator evaluation costs seconds per TXOP. Lower the candidate count for a quicker run. '
                   'Note: trained on specific topology distributions — custom hand-drawn topologies may be '
                   'out-of-distribution.',
    'params': [
        _p('n_samples_eval', 'Candidates per step', 'number', 64,
           'Number of configurations generated by the flow-matching model per step.', min=8, max=512, step=8),
        _p('top_k', 'Top-k evaluated', 'number', 8,
           'How many of the highest-scored candidates are evaluated in the full simulator.', min=1, max=64, step=1),
        _p('n_eval_repeats', 'Eval repeats', 'number', 5,
           'Simulator evaluations per selected candidate (averaged).', min=1, max=20, step=1),
        _p('use_simulator', 'Score with simulator', 'checkbox', False,
           'Score every candidate with the full simulator instead of the fast surrogate model. '
           'More accurate but much slower.'),
        _p('seed', 'Seed', 'number', 42, 'Random seed.', min=0, max=100_000, step=1),
    ],
    'run': _run_fm,
})
