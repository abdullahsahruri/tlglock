"""
TLGLock: key-driven logic locking in threshold logic gates.

Rebuild of the flow described in Sahruri & Margala, "TLGLock: A New Approach
in Logic Locking Using Key-Driven Charge Recycling in Threshold Logic Gates,"
IFIP/IEEE VLSI-SoC 2025.
"""

from .thfile import ThGate, ThNetwork, ThParseError, parse_th, read_th, write_th
from .sim import simulate, outputs_of, truth_table, enumerate_assignments
from .locking import (
    LockReport, lock, embed_keys, generate_key,
    select_lock_gates, assign_key_weights,
)
from .metrics import (
    corruption_rate, is_equivalent_under_correct_key,
    equivalent_key_count, output_hamming_profile, key_weight_sweep,
)
from .opb import OpbEncoder, build_distinguishing_miter

__version__ = "0.2.0"

__all__ = [
    "ThGate", "ThNetwork", "ThParseError", "parse_th", "read_th", "write_th",
    "simulate", "outputs_of", "truth_table", "enumerate_assignments",
    "LockReport", "lock", "embed_keys", "generate_key",
    "select_lock_gates", "assign_key_weights",
    "corruption_rate", "is_equivalent_under_correct_key",
    "equivalent_key_count", "output_hamming_profile", "key_weight_sweep",
    "OpbEncoder", "build_distinguishing_miter",
]
