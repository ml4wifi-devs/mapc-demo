import os

# Must be set before the first JAX import (parity with mapc_research experiments).
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
