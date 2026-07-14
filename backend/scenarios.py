"""Scenario registry: parameter schemas (with tooltips) and builders.

Every scenario builder returns a StaticScenario from mapc_research. The
frontend renders forms directly from the schemas returned by /api/catalog.
"""

import inspect

import jax.numpy as jnp
from mapc_sim.constants import DEFAULT_SIGMA, DEFAULT_TX_POWER
from mapc_sim.utils import enterprise_tgax_path_loss, residential_tgax_path_loss
from mapc_research.envs.scenario_impl import (
    enterprise_scenario,
    indoor_small_bsss_scenario,
    residential_scenario,
    small_office_scenario,
    spatial_reuse_scenario,
    symm_residential_scenario,
    random_scenario,
    toy_scenario_1,
    toy_scenario_2,
)
from mapc_research.envs.static_scenario import StaticScenario

PATH_LOSS_FNS = {
    'enterprise': enterprise_tgax_path_loss,
    'residential': residential_tgax_path_loss,
}


def _p(name, label, type_, default, tooltip, **extra):
    return {'name': name, 'label': label, 'type': type_, 'default': default, 'tooltip': tooltip, **extra}


GLOBAL_PARAMS = [
    _p('n_steps', 'Time horizon [TXOPs]', 'number', 600,
       'Shared simulation length for every method: number of Co-SR transmission opportunities. '
       'Learning agents and metaheuristics run this many steps, FM4WiFi spends it on channel probing, '
       'and DCF/SR simulate the equivalent channel time (one TXOP ≈ 5.5 ms), so all curves and lines '
       'on the chart refer to the same time horizon.',
       min=10, max=100_000, step=1),
    _p('channel_width', 'Channel width [MHz]', 'select', 20,
       'Wi-Fi channel width. Wider channels offer higher data rates but also higher noise floor.',
       options=[20, 40, 80, 160]),
    _p('path_loss', 'Path loss model', 'select', 'enterprise',
       'TGax path loss model. Residential: breaking point 5 m, wall loss 5 dB. '
       'Enterprise: breaking point 10 m, wall loss 7 dB. Residential/apartment scenarios '
       'should use the residential model.',
       options=['enterprise', 'residential']),
    _p('sigma', 'Noise std σ [dB]', 'number', DEFAULT_SIGMA,
       'Standard deviation of the white Gaussian noise added to the SINR. '
       'Higher values make rewards noisier and learning harder.', min=0.0, max=10.0, step=0.1),
    _p('nakagami', 'Nakagami-m fading', 'checkbox', False,
       'Enable Nakagami-m multipath channel fading (m=1 is Rayleigh fading). '
       'Adds realistic small-scale fading to the channel.'),
    _p('nakagami_m', 'Nakagami m', 'number', 1.5,
       'Shape parameter of the Nakagami-m fading distribution. Lower m means stronger fading.',
       min=0.5, max=10.0, step=0.1),
    _p('default_tx_power', 'Max TX power [dBm]', 'number', round(DEFAULT_TX_POWER, 2),
       'Maximum (default) transmission power of the APs. Power levels selected by agents '
       'are reductions from this value.', min=10.0, max=20.0, step=0.1),
    _p('tx_power_delta', 'TX power step [dB]', 'number', 3.0,
       'Difference between consecutive transmission power levels. Effective power = max power − level × step.',
       min=0.5, max=6.0, step=0.5),
    _p('tx_power_levels', 'TX power levels', 'number', 4,
       'Number of discrete transmission power levels available to Co-SR agents. '
       'More levels enlarge the action space and slow down convergence.', min=1, max=8, step=1),
]


SCENARIOS = {
    'custom': {
        'name': 'Custom (draw on canvas)',
        'description': 'Place APs (right click) and their stations (left click) directly on the topology canvas.',
        'factory': None,
        'params': [],
    },
    'small_office': {
        'name': 'Small office (2x2 rooms)',
        'description': '4 APs in a 2x2 room grid, 4 STAs each, walls between BSSs (paper Fig. 2 style).',
        'factory': small_office_scenario,
        'params': [
            _p('d_ap', 'AP distance [m]', 'number', 25.0, 'Distance between neighboring APs.', min=5.0, max=100.0, step=1.0),
            _p('d_sta', 'STA distance [m]', 'number', 2.0, 'Distance of each STA from its AP.', min=0.5, max=20.0, step=0.5),
        ],
    },
    'residential': {
        'name': 'TGax residential',
        'description': 'Residential scenario from IEEE 802.11-14/0980r16: grid of apartments, random node placement. Use residential path loss.',
        'factory': residential_scenario,
        'params': [
            _p('seed', 'Topology seed', 'number', 42, 'Random seed for node placement.', min=0, max=10_000, step=1),
            _p('x_apartments', 'Apartments (x)', 'number', 4, 'Number of apartments along the x axis (suggested 2-10).', min=1, max=10, step=1),
            _p('y_apartments', 'Apartments (y)', 'number', 2, 'Number of apartments along the y axis (suggested 2).', min=1, max=4, step=1),
            _p('n_sta_per_ap', 'STAs per AP', 'number', 2, 'Number of stations per apartment (suggested 1-10).', min=1, max=10, step=1),
            _p('size', 'Apartment size [m]', 'number', 10.0, 'Side length of a square apartment (suggested 5-10).', min=5.0, max=20.0, step=1.0),
        ],
    },
    'symm_residential': {
        'name': 'Symmetric residential (paper topology)',
        'description': 'Symmetric variant of the residential scenario: AP centered in each apartment, STAs on a ring — the 2x4 room topology from the paper. Use residential path loss.',
        'factory': symm_residential_scenario,
        'params': [
            _p('seed', 'Topology seed', 'number', 42, 'Random seed (used for circle STA placement).', min=0, max=10_000, step=1),
            _p('x_apartments', 'Apartments (x)', 'number', 4, 'Number of apartments along the x axis.', min=1, max=10, step=1),
            _p('y_apartments', 'Apartments (y)', 'number', 2, 'Number of apartments along the y axis.', min=1, max=4, step=1),
            _p('n_sta_per_ap', 'STAs per AP', 'number', 4, 'Number of stations per AP.', min=1, max=10, step=1),
            _p('size', 'Apartment size [m]', 'number', 10.0, 'Side length of a square apartment.', min=5.0, max=20.0, step=1.0),
            _p('d_sta', 'STA radius [m]', 'number', 2.0, 'Distance of the stations from their AP.', min=0.5, max=10.0, step=0.5),
            _p('sta_positioning', 'STA placement', 'select', 0, 'Ring: STAs evenly spaced on a circle. Circle: STAs uniformly random within the radius.',
               options=[{'value': 0, 'label': 'ring'}, {'value': 1, 'label': 'circle'}]),
        ],
    },
    'enterprise': {
        'name': 'TGax enterprise',
        'description': 'Enterprise scenario from IEEE 802.11-14/0980r16: offices with cubicles. Large! Use enterprise path loss.',
        'factory': enterprise_scenario,
        'params': [
            _p('seed', 'Topology seed', 'number', 42, 'Random seed for STA placement.', min=0, max=10_000, step=1),
            _p('x_offices', 'Offices (x)', 'number', 1, 'Number of offices along the x axis (suggested 1-4).', min=1, max=4, step=1),
            _p('y_offices', 'Offices (y)', 'number', 1, 'Number of offices along the y axis (suggested 1-2).', min=1, max=2, step=1),
            _p('n_sta_per_cubicle', 'STAs per cubicle', 'number', 1, 'Number of stations per cubicle (suggested 1-4).', min=1, max=4, step=1),
        ],
    },
    'random': {
        'name': 'Random open space',
        'description': 'APs placed uniformly at random in a square, STAs scattered around their APs (Gaussian). No walls.',
        'factory': random_scenario,
        'params': [
            _p('seed', 'Topology seed', 'number', 42, 'Random seed for node placement.', min=0, max=10_000, step=1),
            _p('d_ap', 'Area size [m]', 'number', 75.0, 'Side length of the square in which APs are placed.', min=10.0, max=200.0, step=5.0),
            _p('n_ap', 'Number of APs', 'number', 4, 'Number of access points.', min=1, max=12, step=1),
            _p('d_sta', 'STA spread [m]', 'number', 5.0, 'Standard deviation of STA placement around their AP.', min=0.5, max=20.0, step=0.5),
            _p('n_sta_per_ap', 'STAs per AP', 'number', 4, 'Number of stations per AP.', min=1, max=10, step=1),
        ],
    },
    'spatial_reuse': {
        'name': 'Spatial reuse (2 BSS line)',
        'description': 'Classic two-BSS line topology: STA — AP — AP — STA. Minimal example where Co-SR pays off.',
        'factory': spatial_reuse_scenario,
        'params': [
            _p('d_ap', 'AP-AP distance [m]', 'number', 20.0, 'Distance between the two APs.', min=1.0, max=100.0, step=1.0),
            _p('d_sta', 'AP-STA distance [m]', 'number', 5.0, 'Distance between each AP and its STA.', min=0.5, max=50.0, step=0.5),
        ],
    },
    'toy_1': {
        'name': 'Toy: 2 BSS line (shared STAs)',
        'description': 'STA AP STA | STA AP STA in a line, distance d apart.',
        'factory': toy_scenario_1,
        'params': [
            _p('d', 'Distance [m]', 'number', 20.0, 'Spacing between consecutive nodes in the line.', min=1.0, max=100.0, step=1.0),
        ],
    },
    'toy_2': {
        'name': 'Toy: 4 BSS square',
        'description': '4 APs in a square (side d_ap), 4 STAs per AP at distance d_sta. No walls.',
        'factory': toy_scenario_2,
        'params': [
            _p('d_ap', 'AP distance [m]', 'number', 50.0, 'Side length of the AP square.', min=5.0, max=200.0, step=1.0),
            _p('d_sta', 'STA distance [m]', 'number', 2.0, 'Distance of each STA from its AP.', min=0.5, max=20.0, step=0.5),
        ],
    },
    'indoor_small_bsss': {
        'name': 'Indoor small BSSs (hex grid)',
        'description': 'Indoor Small BSSs scenario from IEEE 802.11-14/0980r16: hexagonal AP grid, no walls. Large!',
        'factory': indoor_small_bsss_scenario,
        'params': [
            _p('seed', 'Topology seed', 'number', 42, 'Random seed for STA placement.', min=0, max=10_000, step=1),
            _p('grid_layers', 'Grid layers', 'number', 2, 'Number of hexagonal grid layers (3 or 5 in the standard).', min=1, max=5, step=1),
            _p('n_sta_per_ap', 'STAs per AP', 'number', 5, 'Number of stations per AP (suggested 5-30).', min=1, max=30, step=1),
            _p('bss_radius', 'BSS radius [m]', 'number', 10.0, 'Radius of each hexagonal BSS cell.', min=5.0, max=30.0, step=1.0),
        ],
    },
}


def _scenario_kwargs(globals_cfg: dict) -> dict:
    kwargs = {
        'channel_width': int(globals_cfg.get('channel_width', 20)),
        'sigma': float(globals_cfg.get('sigma', DEFAULT_SIGMA)),
        'nakagami_m': float(globals_cfg['nakagami_m']) if globals_cfg.get('nakagami') else None,
        'default_tx_power': float(globals_cfg.get('default_tx_power', DEFAULT_TX_POWER)),
        'tx_power_delta': float(globals_cfg.get('tx_power_delta', 3.0)),
        'path_loss_fn': PATH_LOSS_FNS[globals_cfg.get('path_loss', 'enterprise')],
    }
    return kwargs


def build_scenario(config: dict) -> StaticScenario:
    """Build a StaticScenario from a frontend config:
    {"id": ..., "params": {...}, "custom": {...}, "globals": {...}}
    """
    scenario_id = config['id']
    globals_cfg = config.get('globals', {})
    kwargs = _scenario_kwargs(globals_cfg)

    if scenario_id == 'custom':
        return _build_custom(config.get('custom') or {}, kwargs)

    entry = SCENARIOS[scenario_id]
    factory = entry['factory']
    params = dict(config.get('params', {}))

    for schema in entry['params']:
        params.setdefault(schema['name'], schema['default'])
        if schema['type'] == 'number' and schema.get('step') == 1:
            params[schema['name']] = int(params[schema['name']])

    if scenario_id == 'random':
        params['randomize'] = False
    if scenario_id == 'symm_residential':
        params['n_steps'] = float('inf')  # required positional; unused by the demo loop

    # Some factories do not accept **kwargs — filter to what they take.
    sig = inspect.signature(factory)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return factory(**params, **kwargs)


def _build_custom(custom: dict, kwargs: dict) -> StaticScenario:
    aps = custom.get('aps') or []
    if not aps or not any(ap.get('stas') for ap in aps):
        raise ValueError('Custom topology needs at least one AP with at least one station.')

    n_ap = len(aps)
    pos, associations = [], {}

    for i, ap in enumerate(aps):
        pos.append([float(ap['x']), float(ap['y'])])

    sta_id = n_ap
    for i, ap in enumerate(aps):
        stas = ap.get('stas') or []
        if not stas:
            raise ValueError(f'AP {i + 1} has no stations — every AP needs at least one.')
        associations[i] = list(range(sta_id, sta_id + len(stas)))
        sta_id += len(stas)

    for ap in aps:
        for sta in ap.get('stas') or []:
            pos.append([float(sta['x']), float(sta['y'])])

    return StaticScenario(jnp.array(pos), associations, str_repr='custom', **kwargs)


def scenario_preview(config: dict) -> dict:
    scenario = build_scenario(config)
    associations = {int(ap): [int(s) for s in stas] for ap, stas in scenario.associations.items()}
    walls_pos = scenario.walls_pos if scenario.walls_pos is not None else []
    if hasattr(walls_pos, 'tolist'):
        walls_pos = walls_pos.tolist()
    return {
        'pos': [[float(x), float(y)] for x, y in scenario.pos.tolist()],
        'associations': associations,
        'walls_pos': [[float(v) for v in w] for w in walls_pos],
        'n_nodes': int(scenario.pos.shape[0]),
        'str_repr': scenario.str_repr,
    }


def catalog() -> dict:
    return {
        'scenarios': [
            {'id': sid, 'name': s['name'], 'description': s['description'], 'params': s['params']}
            for sid, s in SCENARIOS.items()
        ],
        'globals': GLOBAL_PARAMS,
    }
