"""
TLGLock: key-driven logic locking in threshold logic gates.

Rebuild of the flow described in Sahruri & Margala, "TLGLock: A New Approach
in Logic Locking Using Key-Driven Charge Recycling in Threshold Logic Gates,"
IFIP/IEEE VLSI-SoC 2025.

The flow, following Fig. 3 of the paper:

    read_circuit  ->  map_to_tlg  ->  collapse  ->  lock  ->  sat_attack
      (step 1)        (steps 2-3)     (step 4)    (steps 5-7)  (evaluation)

Cell-level area, power and delay come from `characterize`, which drives
ngspice against ASU PTM model cards -- no commercial simulator or PDK.
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
from .encoding import (
    GateEncoding, KeyEquations, NetworkEncoding,
    format_report, gate_encoding, key_equations, network_encoding,
)
from .opb import OpbEncoder, build_distinguishing_miter
from .lp import Constraint, solve_feasibility
from .separable import (
    ThresholdRealisation, identify, is_threshold, gate_to_table, truth_bits,
)
from .abc import (
    Aig, MapStats, SynthError, enumerate_cuts, map_to_tlg,
    read_bench, read_blif, read_circuit, synthesize, abc_available,
)
from .collapse import CollapseStats, collapse, compose, equivalent
from .attack import (
    AppSatResult, AttackResult, ExternalSolver, PbSolver, Status,
    appsat_attack, estimate_key_error,
    oracle_from, sat_attack, verify_recovered_key,
)
from .spice import (
    CellSpec, Technology, TECHNOLOGIES, PTM45_HP, PTM45_LP, PTM130,
    area_um2, build_deck, subckt, worst_case_stimulus,
)
from .characterize import (
    CellResult, MeasurementError, SimulatorError,
    characterize, key_size_sweep, ngspice_available,
    parse_measurements, table_i_rows, write_csv,
)

__version__ = "0.4.0"

__all__ = [
    # netlist
    "ThGate", "ThNetwork", "ThParseError", "parse_th", "read_th", "write_th",
    # simulation
    "simulate", "outputs_of", "truth_table", "enumerate_assignments",
    # locking
    "LockReport", "lock", "embed_keys", "generate_key",
    "select_lock_gates", "assign_key_weights",
    # metrics
    "corruption_rate", "is_equivalent_under_correct_key",
    "equivalent_key_count", "output_hamming_profile", "key_weight_sweep",
    # encoding size
    "GateEncoding", "KeyEquations", "NetworkEncoding",
    "format_report", "gate_encoding", "key_equations", "network_encoding",
    # encoding
    "OpbEncoder", "build_distinguishing_miter",
    "Constraint", "solve_feasibility",
    # threshold identification
    "ThresholdRealisation", "identify", "is_threshold", "gate_to_table",
    "truth_bits",
    # synthesis
    "Aig", "MapStats", "SynthError", "enumerate_cuts", "map_to_tlg",
    "read_bench", "read_blif", "read_circuit", "synthesize", "abc_available",
    # collapse
    "CollapseStats", "collapse", "compose", "equivalent",
    # attack
    "AppSatResult", "AttackResult", "ExternalSolver", "PbSolver", "Status",
    "appsat_attack", "estimate_key_error",
    "oracle_from", "sat_attack", "verify_recovered_key",
    # characterization
    "CellSpec", "Technology", "TECHNOLOGIES", "PTM45_HP", "PTM45_LP", "PTM130",
    "area_um2", "build_deck", "subckt", "worst_case_stimulus",
    "CellResult", "MeasurementError", "SimulatorError",
    "characterize", "key_size_sweep", "ngspice_available",
    "parse_measurements", "table_i_rows", "write_csv",
]
