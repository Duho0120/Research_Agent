from __future__ import annotations

import copy
import json
from typing import Any

from .. import simple_yaml
from ..paths import competition_configs_dir, configs_dir, trial_dir
from ..store import read_text, write_text


def prepare_patch_plan(competition: str, source_trial_id: str, next_trial_id: str) -> dict[str, Any]:
    """Turn a next-experiment recommendation into concrete config and code-edit work."""

    source_dir = trial_dir(competition, source_trial_id)
    next_dir = trial_dir(competition, next_trial_id)
    source_config = simple_yaml.load(source_dir / "config.yaml", default={})
    if not source_config:
        raise FileNotFoundError(f"Missing source config: {source_dir / 'config.yaml'}")

    strategy = _read_strategy(next_dir / "next_experiment.md")
    pipeline_plan = _read_pipeline_improvement_plan(source_dir)
    pipeline_axis = pipeline_plan.get("primary_axis")
    allowed = _load_allowed_space(competition)
    config = copy.deepcopy(source_config)
    config_changes: dict[str, Any] = {}
    target_files = [f"experiments/{competition}/{next_trial_id}/config.yaml"]
    implementation_steps: list[str] = []

    if strategy == "sota_architecture_attempt":
        _apply_sota_config(config, allowed, config_changes)
        target_files.append("scripts/demo_train.py")
        implementation_steps.extend(
            [
                "Add or select a training branch for the SOTA/model-architecture candidate.",
                "Keep the output contract unchanged: metrics.json and optional submission.csv.",
                "Run validation before any leaderboard submission.",
            ]
        )
    elif strategy == "model_family_change":
        _apply_model_family_change(config, allowed, config_changes)
        implementation_steps.append("Switch the model adapter while preserving the existing feature pipeline.")
    elif strategy == "validation_review":
        _apply_validation_review(config, allowed, config_changes)
        implementation_steps.append("Run a CV/LB alignment trial before model-family changes.")
    else:
        _apply_controlled_refinement(config, allowed, config_changes)
        implementation_steps.append("Run one controlled refinement and compare against known seed variance.")

    _apply_pipeline_axis(
        config,
        config_changes,
        target_files,
        implementation_steps,
        pipeline_axis,
        bool(pipeline_plan.get("requires_human_review")),
    )

    plan = {
        "competition": competition,
        "source_trial_id": source_trial_id,
        "next_trial_id": next_trial_id,
        "strategy": strategy,
        "pipeline_axis": pipeline_axis,
        "secondary_axes": pipeline_plan.get("secondary_axes", []),
        "protected_axes": pipeline_plan.get("protected_axes", []),
        "requires_user_approval": _requires_user_approval(pipeline_axis, pipeline_plan),
        "target_files": target_files,
        "config_changes": config_changes,
        "config": config,
        "implementation_steps": implementation_steps,
        "validation_commands": [
            f"python -B -m kaggle_research_agent.cli validate-config --competition {competition} --trial {next_trial_id}",
            "python -B -m unittest discover -s tests -v",
        ],
        "submit_gate": [
            "Do not submit until training writes metrics.json.",
            "Before submission, record current leaderboard score/rank.",
            "After submission, record submitted score/rank and preserve all trial artifacts.",
        ],
    }

    next_dir.mkdir(parents=True, exist_ok=True)
    simple_yaml.dump(config, next_dir / "config.yaml")
    write_text(next_dir / "code_patch_plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    write_text(next_dir / "code_patch_plan.md", render_patch_plan(plan))
    return plan


def render_patch_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"# {plan['next_trial_id']} Code Patch Plan",
        "",
        "## Strategy",
        "",
        plan["strategy"],
        "",
        "## Target Files",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["target_files"])
    lines.extend(["", "## Pipeline Axis", "", f"- primary_axis: {plan.get('pipeline_axis') or 'None'}"])
    lines.extend(f"- protected_axis: {item}" for item in plan.get("protected_axes", []) or ["None"])
    lines.extend(["", "## Approval", "", f"- requires_user_approval: {plan.get('requires_user_approval', False)}"])
    lines.extend(["", "## Config Changes", ""])
    if plan["config_changes"]:
        lines.extend(f"- {key}: {value}" for key, value in plan["config_changes"].items())
    else:
        lines.append("- No config change proposed.")
    lines.extend(["", "## Implementation Steps", ""])
    lines.extend(f"- {item}" for item in plan["implementation_steps"])
    lines.extend(["", "## Validation Commands", ""])
    lines.extend(f"- `{item}`" for item in plan["validation_commands"])
    lines.extend(["", "## Submit Gate", ""])
    lines.extend(f"- {item}" for item in plan["submit_gate"])
    lines.append("")
    return "\n".join(lines)


def _read_strategy(path) -> str:
    text = read_text(path)
    if not text.strip():
        return "controlled_refinement"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "## Strategy":
            for candidate in lines[index + 1 :]:
                stripped = candidate.strip()
                if stripped:
                    return stripped
    return "controlled_refinement"


def _read_pipeline_improvement_plan(source_dir) -> dict[str, Any]:
    path = source_dir / "pipeline_improvement_plan.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_allowed_space(competition: str) -> dict[str, Any]:
    competition_path = competition_configs_dir(competition) / "allowed_space.yaml"
    if competition_path.exists():
        return simple_yaml.load(competition_path, default={})
    return simple_yaml.load(configs_dir() / "allowed_space.yaml", default={})


def _apply_controlled_refinement(config: dict[str, Any], allowed: dict[str, Any], changes: dict[str, Any]) -> None:
    features = config.setdefault("features", {})
    if not allowed or _allowed(allowed, ["features", "use_frequency_encoding"], True):
        features["use_frequency_encoding"] = True
        changes["features.use_frequency_encoding"] = True
        return
    params = config.setdefault("model", {}).setdefault("params", {})
    lr = float(params.get("learning_rate", 0.03))
    new_lr = max(0.005, round(lr * 0.8, 5))
    params["learning_rate"] = new_lr
    changes["model.params.learning_rate"] = new_lr


def _apply_model_family_change(config: dict[str, Any], allowed: dict[str, Any], changes: dict[str, Any]) -> None:
    current = config.setdefault("model", {}).get("type")
    for candidate in ["catboost", "xgboost", "tabular_nn", "skeleton_transformer", "lightgbm"]:
        if candidate != current and (not allowed or _allowed(allowed, ["model", "type"], candidate)):
            config["model"]["type"] = candidate
            changes["model.type"] = candidate
            return


def _apply_sota_config(config: dict[str, Any], allowed: dict[str, Any], changes: dict[str, Any]) -> None:
    model = config.setdefault("model", {})
    if _allowed(allowed, ["model", "type"], "skeleton_transformer"):
        model["type"] = "skeleton_transformer"
        model["params"] = {
            "learning_rate": _middle(allowed, ["model", "params", "learning_rate"], 0.02),
            "d_model": _middle(allowed, ["model", "params", "d_model"], 128),
            "nhead": _choice(allowed, ["model", "params", "nhead"], 4),
            "num_layers": _middle(allowed, ["model", "params", "num_layers"], 3),
            "dropout": _middle(allowed, ["model", "params", "dropout"], 0.1),
        }
        changes["model.type"] = "skeleton_transformer"
    else:
        _apply_model_family_change(config, allowed, changes)

    features = config.setdefault("features", {})
    for key in ["use_view_aware_features", "use_bed_wandering_aux_head"]:
        if _allowed(allowed, ["features", key], True):
            features[key] = True
            changes[f"features.{key}"] = True
    if "training" in allowed:
        config["training"] = {
            "epochs": _middle(allowed, ["training", "epochs"], 30),
            "batch_size": _choice(allowed, ["training", "batch_size"], 32),
            "warmup_epochs": _middle(allowed, ["training", "warmup_epochs"], 3),
            "early_stopping_patience": _middle(allowed, ["training", "early_stopping_patience"], 10),
        }
        changes["training"] = config["training"]


def _apply_validation_review(config: dict[str, Any], allowed: dict[str, Any], changes: dict[str, Any]) -> None:
    cv = config.setdefault("cv", {})
    if _allowed(allowed, ["cv", "n_splits"], 10):
        cv["n_splits"] = 10
        changes["cv.n_splits"] = 10
    else:
        seed = int(cv.get("seed", 42)) + 1
        cv["seed"] = seed
        changes["cv.seed"] = seed


def _apply_pipeline_axis(
    config: dict[str, Any],
    changes: dict[str, Any],
    target_files: list[str],
    implementation_steps: list[str],
    axis: str | None,
    requires_human_review: bool,
) -> None:
    if axis == "augmentation":
        augmentation = config.setdefault("augmentation", {})
        augmentation["enabled"] = True
        augmentation["policy"] = "domain_safe_light"
        changes["augmentation.enabled"] = True
        changes["augmentation.policy"] = "domain_safe_light"
        _append_unique(target_files, "kaggle_research_agent/pipeline/augmentation.py")
        implementation_steps.append("Add or update the augmentation branch and keep validation fixed.")
    elif axis == "sampling":
        training = config.setdefault("training", {})
        training["sampler"] = "balanced"
        training["sampling_weight_source"] = "train_labels"
        changes["training.sampler"] = "balanced"
        changes["training.sampling_weight_source"] = "train_labels"
        _append_unique(target_files, "kaggle_research_agent/pipeline/dataset.py")
        implementation_steps.append("Add balanced or hard-example sampling without changing validation.")
    elif axis == "loss_metric_alignment":
        training = config.setdefault("training", {})
        training["loss"] = "metric_aligned"
        training["class_weights"] = "auto"
        post = config.setdefault("post_processing", {})
        post["threshold_sweep"] = True
        changes["training.loss"] = "metric_aligned"
        changes["training.class_weights"] = "auto"
        changes["post_processing.threshold_sweep"] = True
        _append_unique(target_files, "kaggle_research_agent/pipeline/losses.py")
        _append_unique(target_files, "kaggle_research_agent/pipeline/post_processing.py")
        implementation_steps.append("Align the training loss, calibration, or threshold sweep with the competition metric.")
    elif axis == "pretraining_strategy":
        model = config.setdefault("model", {})
        model["pretraining"] = {
            "mode": "partial_finetune",
            "source": "approved_pretrained_backbone",
            "freeze_backbone_epochs": 1,
        }
        changes["model.pretraining.mode"] = "partial_finetune"
        changes["model.pretraining.source"] = "approved_pretrained_backbone"
        _append_unique(target_files, "kaggle_research_agent/pipeline/model_adapter.py")
        implementation_steps.append("Confirm external pretrained model permission before implementation.")
        implementation_steps.append("Add a freeze/unfreeze schedule for partial fine-tuning.")
    elif axis == "post_processing":
        post = config.setdefault("post_processing", {})
        post["enabled"] = True
        post["strategy"] = "threshold_or_smoothing"
        changes["post_processing.enabled"] = True
        changes["post_processing.strategy"] = "threshold_or_smoothing"
        _append_unique(target_files, "kaggle_research_agent/pipeline/post_processing.py")
        implementation_steps.append("Add post-processing while preserving model training.")

    if requires_human_review:
        implementation_steps.append("Use the review pack or user approval before applying this axis.")


def _requires_user_approval(axis: str | None, pipeline_plan: dict[str, Any]) -> bool:
    return bool(pipeline_plan.get("requires_human_review")) or axis in {"pretraining_strategy", "model_family", "model_architecture"}


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _allowed(allowed: dict[str, Any], path: list[str], value: Any) -> bool:
    node: Any = allowed
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return isinstance(node, list) and value in node


def _choice(allowed: dict[str, Any], path: list[str], default: Any) -> Any:
    node = _node(allowed, path)
    if isinstance(node, list):
        return default if default in node else node[0]
    return default


def _middle(allowed: dict[str, Any], path: list[str], default: float | int) -> float | int:
    node = _node(allowed, path)
    if isinstance(node, dict) and "min" in node and "max" in node:
        value = (node["min"] + node["max"]) / 2
        return int(value) if isinstance(node["min"], int) and isinstance(node["max"], int) else round(value, 5)
    return default


def _node(allowed: dict[str, Any], path: list[str]) -> Any:
    node: Any = allowed
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node
