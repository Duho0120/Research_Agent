"""What the agent is allowed to write, and what shape it must take.

Two files, two lifetimes:

* ``data_loader.py`` is generated once per competition and then locked. It is
  the only place that knows how this competition stores its files -- DACON
  lays every competition out differently, so no fixed strategy is handed down;
  the agent inspects the real directory and writes what it finds.
* ``model.py`` is rewritten every trial. It never touches ``data/``.

``predict`` takes ``fitted`` as an argument. That single signature choice is
what replaced a checker: under the old four-script model, ``train_step.py``
could skip saving its model and ``predict_step.py`` could skip loading one,
and both still exited 0.
"""

from __future__ import annotations


LOADER_MODULE = "data_loader"
MODEL_MODULE = "model"

DEFAULT_HOLDOUT_RATIO = 0.2
DEFAULT_SEED = 0


class ContractViolation(RuntimeError):
    """The agent's code does not honour the contract.

    Raised loudly rather than worked around: a silent fallback here would put
    a made-up number where a score belongs.
    """


def loader_contract_source() -> str:
    """The loader contract, as text to hand to the code writer."""
    return _LOADER_CONTRACT


def model_contract_source() -> str:
    """The per-trial model contract, as text to hand to the code writer."""
    return _MODEL_CONTRACT


_LOADER_CONTRACT = '''\
# data_loader.py -- written once for this competition, then locked.
# This is the ONLY file that may read the competition data directory.

def load_samples(data_dir: str, split: str) -> list[dict]:
    """Return every sample for `split` ("train" or "test").

    Each sample is a dict carrying at least "id" plus whatever features the
    model will need. Samples from the "train" split additionally carry every
    key named by label_keys().

    Inspect the actual directory and write what is really there. If the
    expected structure is missing, raise -- do not fall back to a guess.
    """

def label_keys() -> list[str]:
    """Which keys on a train sample hold the answer.

    The framework uses this to build the holdout split, so a split can never
    end up without labels. These must be the same names as the submission's
    non-id columns -- what the model is scored on and what gets submitted are
    then one set of names, and cannot drift apart.
    """

def submission_columns() -> list[str]:
    """The submission file's columns, in order, read from the competition's
    own submission template."""

def id_column() -> str:  # optional
    """The id column's name. Defaults to submission_columns()[0]."""
'''

_MODEL_CONTRACT = '''\
# model.py -- rewritten every trial. Never reads data/ directly.

def fit(train_samples: list[dict]) -> object:
    """Train on `train_samples` and return whatever predict() needs.

    A rule-based approach that needs no training may return None or a dict of
    parameters -- returning nothing useful is legitimate, ignoring the input
    entirely is not.
    """

def predict(fitted: object, sample: dict) -> dict:
    """Predict one sample, given what fit() returned.

    Return a dict keyed by the submission columns other than the id column.
    """
'''
