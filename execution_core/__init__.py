"""Framework-owned trial execution.

The old model ran four agent-written programs in sequence and inspected only
their exit codes, so nothing forced training to reach prediction, nothing
stopped the agent from writing its own score, and every failure mode had to be
caught by a checker bolted on afterwards. Here the framework owns the flow and
the agent fills in two functions; see 실행코어_재설계.md.
"""

from .contract import (
    ContractViolation,
    LOADER_MODULE,
    MODEL_MODULE,
    loader_contract_source,
    model_contract_source,
)
from .metrics import METRICS, MetricSpec, compute, register
from .orchestrator import run_trial
from .splitting import split_samples
from .submission_writer import write_submission
from .verification import read_template_ids, verify_test_ids

__all__ = [
    "ContractViolation",
    "LOADER_MODULE",
    "METRICS",
    "MODEL_MODULE",
    "MetricSpec",
    "compute",
    "loader_contract_source",
    "model_contract_source",
    "read_template_ids",
    "register",
    "run_trial",
    "split_samples",
    "verify_test_ids",
    "write_submission",
]
