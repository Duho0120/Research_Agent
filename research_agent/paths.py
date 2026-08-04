from __future__ import annotations

from pathlib import Path


ROOT = Path.cwd()


def project_root() -> Path:
    return ROOT


def competition_dir(competition: str) -> Path:
    return project_root() / "competitions" / competition


def experiments_dir() -> Path:
    return project_root() / "experiments"


def experiment_dir(competition: str) -> Path:
    return experiments_dir() / competition



def trial_dir(competition: str, trial_id: str) -> Path:
    return experiment_dir(competition) / trial_id


def memory_dir() -> Path:
    return project_root() / "memory"


def competition_memory_dir(competition: str) -> Path:
    return memory_dir() / competition


def jobs_dir() -> Path:
    return project_root() / "jobs"


def competition_jobs_dir(competition: str) -> Path:
    return jobs_dir() / competition


def configs_dir() -> Path:
    return project_root() / "configs"


def competition_configs_dir(competition: str) -> Path:
    return configs_dir() / competition


def policies_dir() -> Path:
    return configs_dir() / "policies"


def submissions_dir() -> Path:
    return project_root() / "submissions"


def competition_submissions_dir(competition: str) -> Path:
    return submissions_dir() / competition


def data_dir() -> Path:
    return project_root() / "data"


def competition_data_dir(competition: str) -> Path:
    return data_dir() / competition
