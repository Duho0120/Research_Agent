from __future__ import annotations

import argparse
from . import simple_yaml
from .baseline_generator import generate_baseline_pipeline
from .competition_inspector import inspect_competition
from .competition_onboarding import start_competition
from .config_validator import validate_config
from .data_onboarding import profile_competition_data
from .agents.experiment_runner import apply_patch_plan, create_job, run_local_job
from .agents.coding_handoff import prepare_coding_handoff
from .agents.memory import record_user_feedback, remember_trial, request_user_review
from .agents.model_advisor import advise_model_candidates
from .agents.orchestrator import run_auto_research_loop, run_cycle
from .agents.pipeline_planner import plan_pipeline_improvement
from .agents.pipeline_patch_planner import prepare_patch_plan
from .agents.patch_validator import validate_patch_plan
from .agents.policy_gate import decide_human_review, log_llm_decision
from .agents.research_planner import propose_next_experiment, propose_plan
from .agents.result_analyst import diagnose_trial, evaluate_trial
from .agents.review_pack import prepare_review_pack
from .agents.submission import prepare_submission, record_submission_result, submit_trial
from .graph.research_graph import run_graph_cycle
from .paths import competition_configs_dir, configs_dir, trial_dir
from .store import init_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-research-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--competition", required=True)
    p_init.add_argument("--metric", default="unknown")
    p_init.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")

    p_inspect = sub.add_parser("inspect-competition")
    p_inspect.add_argument("--competition", required=True)

    p_start = sub.add_parser("start-competition")
    p_start.add_argument("--competition", required=True)
    p_start.add_argument("--metric", default="unknown")
    p_start.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    p_start.add_argument("--trial", default="trial_001")

    p_profile_data = sub.add_parser("profile-data")
    p_profile_data.add_argument("--competition", required=True)

    p_generate_baseline = sub.add_parser("generate-baseline")
    p_generate_baseline.add_argument("--competition", required=True)
    p_generate_baseline.add_argument("--trial", default="trial_001")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--competition", required=True)
    p_plan.add_argument("--trial", default="trial_001")

    p_job = sub.add_parser("create-job")
    p_job.add_argument("--competition", required=True)
    p_job.add_argument("--trial", required=True)
    p_job.add_argument("--run-command", dest="run_command", default=None)
    p_job.add_argument("--backend", choices=["local", "colab"], default="local")

    p_run = sub.add_parser("run-local")
    p_run.add_argument("--competition", required=True)
    p_run.add_argument("--trial", required=True)
    p_run.add_argument("--run-command", dest="run_command", default=None)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--competition", required=True)
    p_eval.add_argument("--trial", required=True)

    p_remember = sub.add_parser("remember")
    p_remember.add_argument("--competition", required=True)
    p_remember.add_argument("--trial", required=True)

    p_validate = sub.add_parser("validate-config")
    p_validate.add_argument("--competition", required=True)
    p_validate.add_argument("--trial", required=True)

    p_cycle = sub.add_parser("cycle")
    p_cycle.add_argument("--competition", required=True)
    p_cycle.add_argument("--trial", required=True)
    p_cycle.add_argument("--no-job", action="store_true")
    p_cycle.add_argument("--backend", choices=["local", "colab"], default="local")
    p_cycle.add_argument("--run-now", action="store_true")
    p_cycle.add_argument("--run-command", dest="run_command", default=None)
    p_cycle.add_argument("--next-trial", default=None)
    p_cycle.add_argument("--prepare-next-patch", action="store_true")
    p_cycle.add_argument("--apply-next-patch", action="store_true")
    p_cycle.add_argument("--next-run-command", dest="next_run_command", default=None)

    p_graph_cycle = sub.add_parser("run-graph-cycle")
    p_graph_cycle.add_argument("--competition", required=True)
    p_graph_cycle.add_argument("--trial", required=True)
    p_graph_cycle.add_argument("--no-job", action="store_true")
    p_graph_cycle.add_argument("--backend", choices=["local", "colab"], default="local")
    p_graph_cycle.add_argument("--run-now", action="store_true")
    p_graph_cycle.add_argument("--run-command", dest="run_command", default=None)
    p_graph_cycle.add_argument("--next-trial", default=None)
    p_graph_cycle.add_argument("--prepare-next-patch", action="store_true")
    p_graph_cycle.add_argument("--apply-next-patch", action="store_true")
    p_graph_cycle.add_argument("--next-run-command", dest="next_run_command", default=None)

    p_auto = sub.add_parser("run-auto-loop")
    p_auto.add_argument("--competition", required=True)
    p_auto.add_argument("--start-trial", default="trial_001")
    p_auto.add_argument("--max-trials", type=int, default=3)
    p_auto.add_argument("--submit-policy", choices=["never", "prepare_only"], default="never")
    p_auto.add_argument("--stop-no-improvement", type=int, default=3)
    p_auto.add_argument("--run-now", action="store_true")
    p_auto.add_argument("--run-command", dest="run_command", default=None)
    p_auto.add_argument("--next-run-command", dest="next_run_command", default=None)

    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("--competition", required=True)
    p_diag.add_argument("--trial", required=True)

    p_review = sub.add_parser("request-review")
    p_review.add_argument("--competition", required=True)
    p_review.add_argument("--trial", required=True)

    p_llm = sub.add_parser("decide-llm")
    p_llm.add_argument("--competition", required=True)
    p_llm.add_argument("--trial", default=None)
    p_llm.add_argument("--reason", required=True)
    p_llm.add_argument("--trial-llm-calls", type=int, default=0)
    p_llm.add_argument("--strategy-calls-today", type=int, default=0)
    p_llm.add_argument("--prompt-summary-path", default=None)

    p_feedback = sub.add_parser("record-feedback")
    p_feedback.add_argument("--competition", required=True)
    p_feedback.add_argument("--trial", required=True)
    p_feedback.add_argument("--topic", required=True)
    p_feedback.add_argument("--question", required=True)
    p_feedback.add_argument("--feedback", required=True)
    p_feedback.add_argument("--decision", required=True)
    p_feedback.add_argument("--follow-up-action", required=True)

    p_submit = sub.add_parser("record-submission")
    p_submit.add_argument("--competition", required=True)
    p_submit.add_argument("--trial", required=True)
    p_submit.add_argument("--version-name", required=True)
    p_submit.add_argument("--submission-file", required=True)
    p_submit.add_argument("--cv-score", type=float, default=None)
    p_submit.add_argument("--previous-lb-score", type=float, default=None)
    p_submit.add_argument("--previous-rank", type=int, default=None)
    p_submit.add_argument("--submitted-lb-score", type=float, default=None)
    p_submit.add_argument("--submitted-rank", type=int, default=None)
    p_submit.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    p_submit.add_argument("--notes", default="")

    p_prepare_submit = sub.add_parser("prepare-submission")
    p_prepare_submit.add_argument("--competition", required=True)
    p_prepare_submit.add_argument("--trial", required=True)
    p_prepare_submit.add_argument("--version-name", required=True)
    p_prepare_submit.add_argument("--submission-file", required=True)
    p_prepare_submit.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    p_prepare_submit.add_argument("--notes", default="")

    p_submit_trial = sub.add_parser("submit-trial")
    p_submit_trial.add_argument("--competition", required=True)
    p_submit_trial.add_argument("--trial", required=True)
    p_submit_trial.add_argument("--version-name", required=True)
    p_submit_trial.add_argument("--submission-file", required=True)
    p_submit_trial.add_argument("--before-score", type=float, default=None)
    p_submit_trial.add_argument("--before-rank", type=int, default=None)
    p_submit_trial.add_argument("--after-score", type=float, default=None)
    p_submit_trial.add_argument("--after-rank", type=int, default=None)
    p_submit_trial.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    p_submit_trial.add_argument("--before-command", dest="before_command", default=None)
    p_submit_trial.add_argument("--submit-command", dest="submit_command", default=None)
    p_submit_trial.add_argument("--after-command", dest="after_command", default=None)
    p_submit_trial.add_argument("--kaggle-competition-slug", dest="kaggle_competition_slug", default=None)
    p_submit_trial.add_argument("--kaggle-team-name", dest="kaggle_team_name", default=None)
    p_submit_trial.add_argument("--kaggle-message", dest="kaggle_message", default=None)
    p_submit_trial.add_argument("--poll-leaderboard", action="store_true")
    p_submit_trial.add_argument("--poll-attempts", type=int, default=5)
    p_submit_trial.add_argument("--poll-interval-seconds", type=float, default=30.0)
    p_submit_trial.add_argument("--notes", default="")

    p_next = sub.add_parser("plan-next")
    p_next.add_argument("--competition", required=True)
    p_next.add_argument("--source-trial", required=True)
    p_next.add_argument("--next-trial", required=True)

    p_improve = sub.add_parser("plan-improvement")
    p_improve.add_argument("--competition", required=True)
    p_improve.add_argument("--trial", required=True)

    p_advise_models = sub.add_parser("advise-models")
    p_advise_models.add_argument("--competition", required=True)
    p_advise_models.add_argument("--trial", required=True)

    p_patch = sub.add_parser("prepare-patch")
    p_patch.add_argument("--competition", required=True)
    p_patch.add_argument("--source-trial", required=True)
    p_patch.add_argument("--next-trial", required=True)

    p_validate_patch = sub.add_parser("validate-patch")
    p_validate_patch.add_argument("--competition", required=True)
    p_validate_patch.add_argument("--trial", required=True)
    p_validate_patch.add_argument("--user-approved", action="store_true")

    p_apply = sub.add_parser("apply-patch")
    p_apply.add_argument("--competition", required=True)
    p_apply.add_argument("--trial", required=True)
    p_apply.add_argument("--run-command", dest="run_command", default=None)

    p_handoff = sub.add_parser("prepare-handoff")
    p_handoff.add_argument("--competition", required=True)
    p_handoff.add_argument("--trial", required=True)
    p_handoff.add_argument("--user-approved", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "init":
        init_project(args.competition, args.metric, args.objective)
        print(f"Initialized competition: {args.competition}")
        return 0

    if args.command == "inspect-competition":
        result = inspect_competition(args.competition)
        print(f"Inspected competition: {result['competition_slug']} status={result['status']}")
        return 0 if result["status"] == "ready" else 1

    if args.command == "start-competition":
        result = start_competition(args.competition, metric=args.metric, objective=args.objective, trial_id=args.trial)
        print(f"Started competition: {result['competition_slug']} status={result['status']}")
        return 0 if result["status"] == "ready" else 1

    if args.command == "profile-data":
        result = profile_competition_data(args.competition)
        print(f"Data profile: {result['competition']} status={result['status']} task_type={result['task_type']}")
        return 0 if result["status"] == "ready" else 1

    if args.command == "generate-baseline":
        result = generate_baseline_pipeline(args.competition, args.trial)
        print(f"Baseline pipeline: {result['competition']} {result['trial_id']} status={result['status']}")
        return 0 if result["status"] == "ready" else 1

    if args.command == "plan":
        plan = propose_plan(args.competition, args.trial)
        print(f"Created plan for {plan['trial_id']}")
        return 0

    if args.command == "create-job":
        job = create_job(args.competition, args.trial, args.run_command, backend=args.backend)
        print(f"Created job: {job['job_id']}")
        return 0

    if args.command == "run-local":
        job = run_local_job(args.competition, args.trial, args.run_command)
        print(f"Local job {job['status']}: {job['job_id']}")
        return 0

    if args.command == "evaluate":
        report = evaluate_trial(args.competition, args.trial)
        print(f"Recommendation: {report['recommendation']}")
        return 0

    if args.command == "remember":
        row = remember_trial(args.competition, args.trial)
        print(f"Remembered {row['trial_id']} best={row['is_best']}")
        return 0

    if args.command == "validate-config":
        config = simple_yaml.load(trial_dir(args.competition, args.trial) / "config.yaml", default={})
        allowed_path = competition_configs_dir(args.competition) / "allowed_space.yaml"
        if not allowed_path.exists():
            allowed_path = configs_dir() / "allowed_space.yaml"
        allowed = simple_yaml.load(allowed_path, default={})
        errors = validate_config(config, allowed)
        if errors:
            print("\n".join(errors))
            return 1
        print("Config is valid")
        return 0

    if args.command == "cycle":
        result = run_cycle(
            args.competition,
            args.trial,
            create_job_request=not args.no_job,
            backend=args.backend,
            run_now=args.run_now,
            command=args.run_command,
            next_trial_id=args.next_trial,
            prepare_next_patch=args.prepare_next_patch,
            apply_next_patch=args.apply_next_patch,
            next_run_command=args.next_run_command,
        )
        print(" -> ".join(result["steps"]))
        if result.get("config_errors"):
            print("\n".join(result["config_errors"]))
            return 1
        return 0

    if args.command == "run-graph-cycle":
        result = run_graph_cycle(
            args.competition,
            args.trial,
            create_job_request=not args.no_job,
            backend=args.backend,
            run_now=args.run_now,
            command=args.run_command,
            next_trial_id=args.next_trial,
            prepare_next_patch=args.prepare_next_patch,
            apply_next_patch=args.apply_next_patch,
            next_run_command=args.next_run_command,
        )
        print(" -> ".join(result["steps"]))
        if result.get("config_errors"):
            print("\n".join(result["config_errors"]))
            return 1
        return 0

    if args.command == "run-auto-loop":
        result = run_auto_research_loop(
            args.competition,
            start_trial_id=args.start_trial,
            max_trials=args.max_trials,
            submit_policy=args.submit_policy,
            stop_no_improvement=args.stop_no_improvement,
            run_now=args.run_now,
            command=args.run_command,
            next_run_command=args.next_run_command,
        )
        print(f"Auto loop: {result['competition']} status={result['status']} trials={len(result['trials'])}")
        return 0 if result["status"] in {"completed", "stopped_no_improvement"} else 1

    if args.command == "diagnose":
        diagnosis = diagnose_trial(args.competition, args.trial)
        print(f"Diagnosis: needs_user_review={diagnosis['needs_user_review']}")
        return 0

    if args.command == "request-review":
        diagnosis = diagnose_trial(args.competition, args.trial)
        human_review = decide_human_review(args.competition, args.trial, diagnosis)
        if human_review["decision"] == "prepare_review_pack":
            prepare_review_pack(args.competition, args.trial, diagnosis)
        path = request_user_review(args.competition, args.trial, diagnosis)
        print(f"Review request: {path.as_posix()}")
        return 0

    if args.command == "decide-llm":
        decision = log_llm_decision(
            args.competition,
            args.trial,
            args.reason,
            trial_llm_calls=args.trial_llm_calls,
            strategy_calls_today=args.strategy_calls_today,
            prompt_summary_path=args.prompt_summary_path,
        )
        print(f"LLM decision: {decision['decision']} reason={decision['reason']}")
        return 0

    if args.command == "record-feedback":
        row = record_user_feedback(
            args.competition,
            args.trial,
            topic=args.topic,
            question=args.question,
            user_feedback=args.feedback,
            decision=args.decision,
            follow_up_action=args.follow_up_action,
        )
        print(f"Recorded feedback: {row['decision']}")
        return 0

    if args.command == "record-submission":
        row = record_submission_result(
            competition=args.competition,
            trial_id=args.trial,
            version_name=args.version_name,
            submission_file=args.submission_file,
            cv_score=args.cv_score,
            previous_lb_score=args.previous_lb_score,
            previous_rank=args.previous_rank,
            submitted_lb_score=args.submitted_lb_score,
            submitted_rank=args.submitted_rank,
            objective=args.objective,
            notes=args.notes,
        )
        print(f"Recorded submission: {row['version_name']} best={row['is_best']}")
        return 0

    if args.command == "prepare-submission":
        manifest = prepare_submission(
            competition=args.competition,
            trial_id=args.trial,
            version_name=args.version_name,
            submission_file=args.submission_file,
            objective=args.objective,
            notes=args.notes,
        )
        print(f"Prepared submission: {manifest['trial_id']} status={manifest['status']}")
        return 0 if manifest["status"] == "ready" else 1

    if args.command == "submit-trial":
        result = submit_trial(
            competition=args.competition,
            trial_id=args.trial,
            version_name=args.version_name,
            submission_file=args.submission_file,
            before_score=args.before_score,
            before_rank=args.before_rank,
            after_score=args.after_score,
            after_rank=args.after_rank,
            objective=args.objective,
            before_command=args.before_command,
            submit_command=args.submit_command,
            after_command=args.after_command,
            kaggle_competition_slug=args.kaggle_competition_slug,
            kaggle_team_name=args.kaggle_team_name,
            kaggle_message=args.kaggle_message,
            poll_leaderboard=args.poll_leaderboard,
            poll_attempts=args.poll_attempts,
            poll_interval_seconds=args.poll_interval_seconds,
            notes=args.notes,
        )
        print(f"Submit trial: {result['trial_id']} status={result['status']}")
        return 0 if result["status"] in {"recorded", "submitted"} else 1

    if args.command == "plan-next":
        plan = propose_next_experiment(args.competition, args.source_trial, args.next_trial)
        print(f"Next experiment: {plan['next_trial_id']} strategy={plan['strategy']}")
        return 0

    if args.command == "plan-improvement":
        plan = plan_pipeline_improvement(args.competition, args.trial)
        print(f"Pipeline improvement: {plan['trial_id']} axis={plan['primary_axis']}")
        return 0

    if args.command == "advise-models":
        result = advise_model_candidates(args.competition, args.trial)
        print(f"Model candidates: {result['trial_id']} scope={result['recommendation_scope']}")
        return 0

    if args.command == "prepare-patch":
        plan = prepare_patch_plan(args.competition, args.source_trial, args.next_trial)
        print(f"Patch plan: {plan['next_trial_id']} strategy={plan['strategy']}")
        return 0

    if args.command == "validate-patch":
        result = validate_patch_plan(args.competition, args.trial, user_approved=args.user_approved)
        print(f"Patch validation: {result['trial_id']} status={result['status']}")
        return 0 if result["status"] == "ready" else 1

    if args.command == "apply-patch":
        result = apply_patch_plan(args.competition, args.trial, run_command=args.run_command)
        print(f"Code edit: {result['trial_id']} status={result['status']}")
        return 0 if result["status"] in {"ready", "executed"} else 1

    if args.command == "prepare-handoff":
        result = prepare_coding_handoff(args.competition, args.trial, user_approved=args.user_approved)
        print(f"Coding handoff: {result['trial_id']} status={result['status']}")
        return 0 if result["status"] == "ready" else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
