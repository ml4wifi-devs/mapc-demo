"""Random single-TX baseline — registered last to sit at the bottom of the method list."""

from .base import register
from .hmab import _COMMON, _run_random

register({
    'id': 'random',
    'name': 'Random (single TX)',
    'kind': 'curve',
    'description': 'Legacy-like baseline: a random AP transmits to a random associated station at full power, '
                   'no parallel Co-SR transmissions.',
    'params': list(_COMMON),
    'run': _run_random,
})
