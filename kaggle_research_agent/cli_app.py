from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from . import simple_yaml
from .experiment_qa import answer_experiment_question
from .interface_contract import (
    get_experiment,
    list_experiments,
    list_pending_requests,
    respond_to_request,
    sync_state,
    submit_human_insight,
)
from .execution_profile import load_execution_profile
from .paths import competition_dir, project_root
from .state_db import default_db_path, state_db_connection
from .workspace_preparer import prepare_workspace
from .user_insight_policy import interpret_user_insight


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPETITION = "titanic"
TRIAL_IDS = [f"trial_{index:03d}" for index in range(1, 6)]
Input = Callable[[str], str]
Output = Callable[[str], None]
KNOWN_EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "titanic": {
        "topic": "Titanic - Machine Learning from Disaster",
        "metric": "accuracy",
        "objective": "maximize",
        "target_column": "Survived",
        "id_column": "PassengerId",
        "required_data_files": ["train.csv", "test.csv", "gender_submission.csv"],
    },
    "spaceship-titanic": {
        "topic": "Spaceship Titanic",
        "metric": "accuracy",
        "objective": "maximize",
        "target_column": "Transported",
        "id_column": "PassengerId",
        "required_data_files": ["train.csv", "test.csv", "sample_submission.csv"],
    },
    "house-prices-advanced-regression-techniques": {
        "topic": "House Prices - Advanced Regression Techniques",
        "metric": "rmsle",
        "objective": "minimize",
        "target_column": "SalePrice",
        "id_column": "Id",
        "required_data_files": ["train.csv", "test.csv", "sample_submission.csv"],
    },
    "digit-recognizer": {
        "topic": "Digit Recognizer",
        "metric": "accuracy",
        "objective": "maximize",
        "target_column": "label",
        "id_column": "ImageId",
        "required_data_files": ["train.csv", "test.csv", "sample_submission.csv"],
    },
}


def runtime_dir() -> Path:
    configured = os.environ.get("RESEARCH_AGENT_RUNTIME_DIR")
    return Path(configured).expanduser().resolve() if configured else ROOT / "demo_workspaces" / "_runtime"


def cli_state_path() -> Path:
    return runtime_dir() / "cli_state.json"


def loop_state_path() -> Path:
    return runtime_dir() / "auto_loop_state.json"


def pause_request_path() -> Path:
    return runtime_dir() / "pause.request"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_competition() -> str:
    return str(load_json(cli_state_path()).get("selected_competition") or DEFAULT_COMPETITION)


def select_competition(competition: str) -> None:
    state = load_json(cli_state_path())
    state["selected_competition"] = competition
    save_json_atomic(cli_state_path(), state)


def load_experiments(*, sync: bool) -> list[dict[str, Any]]:
    result = list_experiments(sync=sync)
    experiments = list(result.get("data", {}).get("experiments", [])) if result.get("ok") else []
    by_competition = {str(item.get("competition")): dict(item) for item in experiments if item.get("competition")}
    for item in _filesystem_experiments():
        key = str(item["competition"])
        by_competition[key] = by_competition.get(key, {}) | item
    if DEFAULT_COMPETITION not in by_competition:
        by_competition[DEFAULT_COMPETITION] = {
            "competition": DEFAULT_COMPETITION,
            "topic": "Titanic",
            "state": "local_workspace",
        }
    return sorted(by_competition.values(), key=lambda item: str(item.get("competition") or ""))


def experiment_snapshot(competition: str, *, sync: bool) -> dict[str, Any]:
    result = get_experiment(competition, sync=sync)
    experiment = dict(result.get("data", {}).get("experiment") or {}) if result.get("ok") else {}
    trials = list(result.get("data", {}).get("trials") or []) if result.get("ok") else []
    loop = _current_loop_state(competition)
    manual = _manual_trial_rows(competition)
    latest = _latest_trial(trials, manual)
    best = _best_trial(trials, manual)
    return {
        "competition": competition,
        "topic": experiment.get("topic") or _filesystem_topic(competition) or competition,
        "state": _display_state(experiment.get("state"), loop, manual),
        "current_trial": loop.get("current_trial"),
        "last_completed_trial": loop.get("last_completed_trial") or (latest or {}).get("trial_id"),
        "next_trial": loop.get("next_trial") or experiment.get("next_trial_id") or _infer_next_trial(manual),
        "latest": latest,
        "best": best,
        "pause_requested": bool(loop.get("pause_requested")),
        "pending_request_count": int(experiment.get("pending_request_count") or 0),
        "loop": loop,
    }


def _current_loop_state(competition: str) -> dict[str, Any]:
    loop = load_json(loop_state_path())
    if not loop:
        return {}
    loop_competition = str(loop.get("competition") or "")
    if loop_competition and loop_competition != competition:
        return {}
    status = str(loop.get("status") or "").casefold()
    if status in {"starting", "running", "resuming", "paused", "failed"}:
        return loop
    if status == "completed" and loop.get("next_trial"):
        return loop
    return {}


def render_snapshot(snapshot: dict[str, Any]) -> str:
    latest = snapshot.get("latest") or {}
    best = snapshot.get("best") or {}
    feedback = snapshot.get("pending_request_count") or 0
    lines = [
        f"선택된 실험: {snapshot['competition']} | {snapshot.get('topic') or '-'}",
        (
            f"상태: {snapshot.get('state') or '-'} | 현재 trial: "
            f"{snapshot.get('current_trial') or '-'} | 다음: {snapshot.get('next_trial') or '-'}"
        ),
        (
            f"최근 완료: {snapshot.get('last_completed_trial') or '-'} | 로컬: "
            f"{_score(latest.get('local_score'))} | 제출: {_score(latest.get('lb_score'))}"
        ),
        (
            f"베스트: {best.get('trial_id') or '-'} | 로컬: "
            f"{_score(best.get('local_score'))} | 제출: {_score(best.get('lb_score'))}"
        ),
        f"다음 실험 기준 base: {_next_base_trial_text(snapshot)}",
        f"피드백 요청: {feedback}",
    ]
    progress = _progress_lines(snapshot)
    if progress:
        lines.extend(["", *progress])
    return "\n".join(lines)


def _next_base_trial_text(snapshot: dict[str, Any]) -> str:
    best = snapshot.get("best") or {}
    base = best.get("trial_id") if isinstance(best, dict) else None
    if base:
        if best.get("lb_score") is not None:
            return f"{base} (제출 점수 기준 베스트)"
        return f"{base} (로컬 점수 기준 베스트)"
    latest = snapshot.get("latest") or {}
    latest_trial = latest.get("trial_id") if isinstance(latest, dict) else None
    if latest_trial:
        return f"{latest_trial} (최근 완료 trial)"
    return "-"


def _progress_lines(snapshot: dict[str, Any]) -> list[str]:
    loop = snapshot.get("loop") or {}
    status = _progress_status_text(loop)
    log_lines = _recent_log_lines(loop.get("log_path"), limit=4)
    if not status and not log_lines:
        return []
    lines: list[str] = []
    if status:
        lines.extend(["진행 상태:", status])
    if log_lines:
        if lines:
            lines.append("")
        lines.append("최근 로그:")
        lines.extend(f"- {line}" for line in log_lines)
    return lines


def _progress_status_text(loop: dict[str, Any]) -> str:
    status = str(loop.get("status") or "").casefold()
    current = loop.get("current_trial") or loop.get("next_trial") or "-"
    if status in {"running", "starting", "resuming"}:
        return f"{current} 진행 중"
    if status == "paused":
        return f"중단 대기 완료. 다음 trial: {loop.get('next_trial') or '-'}"
    if status == "completed":
        return f"완료. 최근 완료 trial: {loop.get('last_completed_trial') or '-'}"
    if status == "failed":
        return f"실패: {loop.get('error') or '원인 미상'}"
    return ""


def _recent_log_lines(log_path: Any, *, limit: int) -> list[str]:
    path = Path(str(log_path)) if log_path else runtime_dir() / "auto_loop.log"
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    visible = _summarize_log_text(text)
    return visible[-limit:]


def _summarize_log_text(text: str) -> list[str]:
    visible: list[str] = []
    current_header = ""
    json_buffer: list[str] = []
    json_depth = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("===") and line.endswith("==="):
            if json_buffer:
                _append_json_log_summary(visible, current_header, json_buffer)
                json_buffer = []
                json_depth = 0
            current_header = _clean_log_line(line)
            continue

        if json_buffer:
            json_buffer.append(raw_line)
            json_depth += raw_line.count("{") - raw_line.count("}")
            if json_depth <= 0:
                _append_json_log_summary(visible, current_header, json_buffer)
                current_header = ""
                json_buffer = []
                json_depth = 0
            continue

        if line.startswith("{"):
            json_buffer = [raw_line]
            json_depth = raw_line.count("{") - raw_line.count("}")
            if json_depth <= 0:
                _append_json_log_summary(visible, current_header, json_buffer)
                current_header = ""
                json_buffer = []
                json_depth = 0
            continue

        cleaned = _clean_log_line(raw_line)
        if cleaned:
            visible.append(cleaned)

    if json_buffer:
        _append_json_log_summary(visible, current_header, json_buffer)
    elif current_header:
        visible.append(current_header)
    return visible


def _append_json_log_summary(visible: list[str], header: str, json_lines: list[str]) -> None:
    try:
        payload = json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        if header:
            visible.append(header)
        visible.extend(line for line in (_clean_log_line(line) for line in json_lines) if line)
        return

    if header:
        visible.append(header)
    if not isinstance(payload, dict):
        return

    status = payload.get("status")
    if status:
        visible.append(f"loop status: {status}")
    error = payload.get("error")
    if error:
        visible.append(f"error: {error}")

    trials = payload.get("trials")
    if isinstance(trials, list):
        for trial in trials:
            if isinstance(trial, dict):
                trial_id = trial.get("trial_id")
                trial_status = trial.get("status")
                if trial_id and trial_status:
                    visible.append(f"{trial_id}: {trial_status}")
                elif trial_id:
                    visible.append(str(trial_id))
    elif payload.get("trial_id"):
        trial_id = payload.get("trial_id")
        trial_status = payload.get("status")
        visible.append(f"{trial_id}: {trial_status}" if trial_status else str(trial_id))


def _clean_log_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("===") and line.endswith("==="):
        return line.strip("= ").strip()
    if line in {"{", "}", "[", "]", "},", "],"}:
        return ""
    jsonish = re.fullmatch(r'"([^"]+)"\s*:\s*"?([^",}]+)"?,?', line)
    if jsonish:
        return f"{jsonish.group(1)}: {jsonish.group(2)}"
    return line[:160]


def _clear_screen_if_interactive(input_fn: Input, output: Output) -> None:
    if input_fn is input and output is print and sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def _render_home(snapshot: dict[str, Any], recent_message: str | None = None) -> str:
    feedback = int(snapshot.get("pending_request_count") or 0)
    lines = [
        "Research Agent",
        "",
        render_snapshot(snapshot),
        "",
        "무엇을 할까요?",
        "",
        "1. 실험 바꾸기",
        "2. 새 실험 등록",
        "3. 자동 실험 시작",
        "4. 현재 실험 중단 요청",
        "5. 상태 다시 보기",
        "6. 에이전트에게 질문",
        "7. 다음 실험에 반영할 인사이트 남기기",
        f"8. 피드백 요청 확인 ({feedback})",
        "9. Trial 비교표 보기",
        "10. 폴더/DB 위치 열기",
        "11. 종료",
    ]
    if recent_message:
        lines.extend(["", "최근 메시지:", recent_message.strip()])
    return "\n".join(lines)


def start_experiment(competition: str, *, trial_count: int | None = None, continuous: bool = False) -> str:
    active = load_json(loop_state_path())
    active_competition = str(active.get("competition") or "")
    active_status = str(active.get("status") or "")
    if active_status in {"running", "starting", "resuming"}:
        if active_competition == competition:
            return "\n".join(
                [
                    "이미 자동 실험이 실행 중입니다.",
                    "",
                    f"- 선택된 실험: {competition}",
                    f"- 현재 trial: {active.get('current_trial') or active.get('next_trial') or '-'}",
                    '- 중단하려면 4번 "현재 실험 중단 요청"을 선택하세요.',
                ]
            )
        request_experiment_stop(active_competition)
        return f"{active_competition} 실험에 중단을 요청했습니다. 중단 완료 후 {competition} 실험을 시작해주세요."
    resumable_loop = _current_loop_state(competition)
    start_trial = str(resumable_loop.get("next_trial") or _infer_start_trial(competition) or "")
    if not start_trial:
        return "남은 trial이 없습니다. 다음 trial 계획을 먼저 생성해야 합니다."
    profile = _load_profile_safely(competition)
    if not profile:
        return "실행 프로필이 아직 없습니다. 새 실험 추가에서 워크스페이스를 만들거나 기존 프로젝트 경로를 연결해주세요."
    platform = str(profile.get("platform") or "").casefold()
    max_trials = "999" if continuous else str(trial_count or 5)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generic_workspace_auto_loop.py"),
        "--competition",
        competition,
        "--start-trial",
        start_trial,
        "--max-trials",
        max_trials,
        "--code-writer",
        "--allow-api",
    ]
    end_trial = None
    planned_count = max_trials
    if platform == "kaggle":
        command.extend(["--submit", "--kaggle-slug", competition])

    log_path = runtime_dir() / "auto_loop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pause_request_path().unlink()
    except FileNotFoundError:
        pass
    starting_state = active | {
        "competition": competition,
        "status": "starting",
        "next_trial": start_trial,
        "pause_requested": False,
        "pid": None,
        "log_path": str(log_path),
    }
    save_json_atomic(loop_state_path(), starting_state)
    log_file = log_path.open("a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    env = os.environ.copy()
    env["RESEARCH_AGENT_RUNTIME_DIR"] = str(runtime_dir())
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                close_fds=True,
            )
        except Exception as error:
            save_json_atomic(loop_state_path(), starting_state | {"status": "failed", "error": str(error)})
            raise
    finally:
        log_file.close()
    state = load_json(loop_state_path()) | {"pid": process.pid}
    save_json_atomic(loop_state_path(), state)
    return "\n".join(
        [
            "자동 실험을 시작했습니다.",
            "",
            f"- 선택된 실험: {competition}",
            f"- 시작 trial: {start_trial}",
            f"- 실행 범위: {_run_scope_text(start_trial, end_trial, planned_count, continuous)}",
            "- 진행 방식: 로컬 실행 -> Kaggle 제출 -> 제출 점수 기록 -> 다음 trial 계획",
            "- 중단 요청 시: 현재 trial 완료와 다음 trial 계획 생성까지 마친 뒤 멈춤",
            f"- PID: {process.pid}",
            f"- 로그: {log_path}",
        ]
    )


def _run_scope_text(start_trial: str, end_trial: str | None, planned_count: int | str | None, continuous: bool) -> str:
    if continuous:
        return "중단 요청 전까지 계속 진행"
    if end_trial:
        suffix = f" ({planned_count}회)" if planned_count else ""
        return f"{start_trial} -> {end_trial}{suffix}"
    return f"{planned_count or '-'}회"


def request_experiment_stop(competition: str) -> str:
    state = load_json(loop_state_path())
    if not state or state.get("status") not in {"running", "starting", "resuming"}:
        return "이미 대기 중입니다."
    if state.get("pause_requested"):
        return "이미 중단 대기 중입니다. 현재 trial 완료와 다음 trial 계획 생성이 끝나면 멈춥니다."
    pause_request_path().parent.mkdir(parents=True, exist_ok=True)
    pause_request_path().write_text("requested\n", encoding="utf-8")
    state["pause_requested"] = True
    save_json_atomic(loop_state_path(), state)
    return "중단을 요청했습니다. 현재 trial 완료와 다음 trial 계획 생성이 끝나면 멈춥니다."


def run_menu(*, sync_on_start: bool = True, input_fn: Input = input, output: Output = print) -> int:
    competition = selected_competition()
    recent_message: str | None = None
    try:
        load_experiments(sync=sync_on_start)
    except Exception as error:
        recent_message = f"상태 동기화 경고: {error}"
    while True:
        try:
            snapshot = experiment_snapshot(competition, sync=False)
        except Exception as error:
            snapshot = {"competition": competition, "topic": competition, "state": "확인 실패", "latest": {}, "best": {}}
            recent_message = f"상태 확인 경고: {error}"
        _clear_screen_if_interactive(input_fn, output)
        output(_render_home(snapshot, recent_message))
        choice = input_fn("\n선택 > ").strip().lower()
        if choice == "1":
            competition = _choose_experiment(competition, input_fn, output)
            recent_message = f"{competition} 실험을 선택했습니다."
        elif choice == "2":
            created = _new_experiment_dialog(input_fn, output)
            if created:
                competition = created
                recent_message = f"{competition} 실험을 등록하고 선택했습니다."
            else:
                recent_message = "새 실험 등록을 취소했습니다."
        elif choice == "3":
            recent_message = _start_experiment_dialog(competition, input_fn, output)
        elif choice == "4":
            recent_message = request_experiment_stop(competition)
        elif choice == "5":
            try:
                snapshot = experiment_snapshot(competition, sync=True)
                recent_message = "상태를 다시 불러왔습니다."
            except Exception as error:
                recent_message = f"상태 새로고침 실패: {error}"
        elif choice == "6":
            recent_message = _question_dialog(competition, snapshot, input_fn, output)
        elif choice == "7":
            recent_message = _insight_dialog(competition, snapshot, input_fn, output)
        elif choice == "8":
            recent_message = _feedback_dialog(competition, input_fn, output)
        elif choice == "9":
            output(render_trial_comparison_table(competition))
            recent_message = "Trial 비교표를 표시했습니다."
            if _confirm_previous(input_fn, output, competition):
                continue
        elif choice == "10":
            recent_message = _open_paths_dialog(competition, snapshot, input_fn, output)
        elif choice in {"11", "q", "quit", "exit"}:
            output("종료합니다. 실행 중인 실험은 백그라운드에서 계속됩니다.")
            return 0
        else:
            recent_message = "1~11 중 하나를 입력해주세요."


def _open_paths_dialog(competition: str, snapshot: dict[str, Any], input_fn: Input, output: Output) -> str:
    paths = _artifact_locations(competition, snapshot)
    last_message = ""
    while True:
        output(
            "\n".join(
                [
                    "어떤 폴더/DB를 볼까요?",
                    "",
                    "1. 사용자용 산출물 보기",
                    "2. 실행 워크스페이스 열기",
                    "3. 실험 전체 기록 폴더 열기",
                    "4. 현재/최근 trial 내부 기록 열기",
                    "5. 제출 파일 폴더 열기",
                    "6. SQLite DB",
                    "q. 돌아가기",
                    "",
                ]
            )
        )
        raw = input_fn("선택 (q: 이전으로 돌아가기)> ").strip().lower()
        if raw == "q":
            return last_message or "폴더/DB 위치 보기를 종료했습니다."
        if raw == "1":
            last_message = _user_artifacts_dialog(competition, snapshot, input_fn, output)
            output(last_message)
            if _confirm_previous(input_fn, output, competition):
                return last_message
            continue
        if raw == "2":
            last_message = _open_folder_message(
                paths["workspace"],
                "실행 워크스페이스",
                note=(
                    "에이전트가 실제로 코드를 수정하고 실행하는 작업 공간입니다.\n"
                    "- data: 사용자가 넣은 원본/입력 데이터\n"
                    "- src 또는 실행 스크립트: 에이전트가 수정하는 코드\n"
                    "- outputs: 현재 워크스페이스의 최신 실행 산출물\n"
                    "주의: outputs는 trial별 아카이브가 아니라 최신 작업 결과일 수 있습니다."
                ),
            )
            output(last_message)
            if _confirm_previous(input_fn, output, competition):
                return last_message
            continue
        if raw == "3":
            last_message = _open_folder_message(
                paths["experiment_root"],
                "실험 전체 기록 폴더",
                note=(
                    "trial별 실행 기록, 판단, 제출 기록, 내부 메타데이터를 모아둔 공식 기록 공간입니다.\n"
                    "사용자용 요약만 보려면 1번 사용자용 산출물 보기를 사용하세요."
                ),
            )
            output(last_message)
            if _confirm_previous(input_fn, output, competition):
                return last_message
            continue
        if raw == "4":
            last_message = _internal_records_dialog(competition, snapshot, input_fn, output)
            output(last_message)
            if _confirm_previous(input_fn, output, competition):
                return last_message
            continue
        if raw == "5":
            last_message = _submission_folder_dialog(competition, snapshot, input_fn, output)
            output(last_message)
            if _confirm_previous(input_fn, output, competition):
                return last_message
            continue
        if raw == "6":
            output(render_sqlite_trial_table(competition))
            last_message = "SQLite DB 요약을 표시했습니다."
            if _confirm_previous(input_fn, output, competition):
                return last_message
            continue
        output("1~6 또는 q를 입력해주세요.")


def _confirm_previous(input_fn: Input, output: Output, competition: str) -> bool:
    while True:
        raw = input_fn("확인완료 (Enter: 계속 보기, q: 이전으로 돌아가기, trial 조회 : 원하는 trial_번호 입력)> ").strip()
        lowered = raw.lower()
        if not raw:
            return False
        if lowered == "q":
            return True
        if re.fullmatch(r"trial_\d+", lowered):
            output(render_sqlite_trial_detail(competition, lowered))
            continue
        output("Enter, q, 또는 trial_001 같은 trial 번호를 입력해주세요.")


def _user_artifacts_dialog(competition: str, snapshot: dict[str, Any], input_fn: Input, output: Output) -> str:
    best_trial = _best_trial_id(snapshot)
    recent_trial = _recent_trial_id(snapshot)
    while True:
        output(
            "\n".join(
                [
                    "어떤 사용자용 산출물을 볼까요?",
                    "",
                    f"1. 베스트 trial 요약 보기: {best_trial or '-'}",
                    f"2. 최근 완료 trial 요약 보기: {recent_trial or '-'}",
                    "3. trial 번호 직접 입력해서 요약 보기",
                    f"4. 베스트 trial 산출물 폴더 열기: {best_trial or '-'}",
                    f"5. 최근 완료 trial 산출물 폴더 열기: {recent_trial or '-'}",
                    "6. 전체 trial 산출물 위치 열기",
                    "q. 돌아가기",
                    "",
                ]
            )
        )
        raw = input_fn("선택 (q: 이전으로 돌아가기)> ").strip().lower()
        if raw == "q":
            return "사용자용 산출물 보기를 취소했습니다."
        if raw == "1":
            return render_user_artifact_summary(competition, best_trial, label="베스트 trial")
        if raw == "2":
            return render_user_artifact_summary(competition, recent_trial, label="최근 완료 trial")
        if raw == "3":
            trial_id = input_fn("trial 번호 입력 (예: trial_003, q: 취소)> ").strip().lower()
            if trial_id == "q":
                return "trial 선택을 취소했습니다."
            if not re.fullmatch(r"trial_\d+", trial_id):
                output("trial_001 같은 trial 번호를 입력해주세요.")
                continue
            return render_user_artifact_summary(competition, trial_id, label="선택 trial")
        if raw == "4":
            return _open_trial_user_artifacts(competition, best_trial, "베스트 trial 사용자용 산출물")
        if raw == "5":
            return _open_trial_user_artifacts(competition, recent_trial, "최근 완료 trial 사용자용 산출물")
        if raw == "6":
            return _open_user_artifacts_roots(competition)
        output("1~6 또는 q를 입력해주세요.")


def _submission_folder_dialog(competition: str, snapshot: dict[str, Any], input_fn: Input, output: Output) -> str:
    best_trial = _best_trial_id(snapshot)
    recent_trial = _recent_trial_id(snapshot)
    workspace = _workspace_path(competition)
    while True:
        output(
            "\n".join(
                [
                    "어떤 제출 파일 폴더를 볼까요?",
                    "",
                    f"1. 최근 완료 trial 제출 파일 폴더 열기: {recent_trial or '-'}",
                    f"2. 베스트 trial 제출 파일 폴더 열기: {best_trial or '-'}",
                    "3. 실행 워크스페이스 최신 outputs 폴더 열기",
                    "4. trial 번호 직접 입력해서 제출 파일 폴더 열기",
                    "q. 돌아가기",
                    "",
                ]
            )
        )
        raw = input_fn("선택 (q: 이전으로 돌아가기)> ").strip().lower()
        if raw == "q":
            return "제출 파일 폴더 보기를 취소했습니다."
        if raw == "1":
            return _open_trial_submission_folder(competition, recent_trial, "최근 완료 trial 제출 파일 폴더")
        if raw == "2":
            return _open_trial_submission_folder(competition, best_trial, "베스트 trial 제출 파일 폴더")
        if raw == "3":
            return _open_folder_message(
                _submission_folder(competition, workspace),
                "실행 워크스페이스 최신 outputs 폴더",
                note="이 위치는 trial별 제출 아카이브가 아니라 현재 워크스페이스의 최신 산출물 위치입니다.",
            )
        if raw == "4":
            trial_id = input_fn("trial 번호 입력 (예: trial_003, q: 취소)> ").strip().lower()
            if trial_id == "q":
                return "trial 선택을 취소했습니다."
            if not re.fullmatch(r"trial_\d+", trial_id):
                output("trial_001 같은 trial 번호를 입력해주세요.")
                continue
            return _open_trial_submission_folder(competition, trial_id, f"{trial_id} 제출 파일 폴더")
        output("1~4 또는 q를 입력해주세요.")


def _internal_records_dialog(competition: str, snapshot: dict[str, Any], input_fn: Input, output: Output) -> str:
    current_trial = str(snapshot.get("current_trial") or "") or None
    recent_trial = _recent_trial_id(snapshot)
    best_trial = _best_trial_id(snapshot)
    while True:
        output(
            "\n".join(
                [
                    "어떤 trial 내부 기록을 볼까요?",
                    "",
                    f"1. 현재 실행 중 trial 내부 기록 열기: {current_trial or '-'}",
                    f"2. 최근 완료 trial 내부 기록 열기: {recent_trial or '-'}",
                    f"3. 베스트 trial 내부 기록 열기: {best_trial or '-'}",
                    "4. trial 번호 직접 입력해서 내부 기록 열기",
                    "q. 돌아가기",
                    "",
                ]
            )
        )
        raw = input_fn("선택 (q: 이전으로 돌아가기)> ").strip().lower()
        if raw == "q":
            return "trial 내부 기록 보기를 취소했습니다."
        if raw == "1":
            return _open_trial_internal_records(competition, current_trial, "현재 실행 중 trial 내부 기록")
        if raw == "2":
            return _open_trial_internal_records(competition, recent_trial, "최근 완료 trial 내부 기록")
        if raw == "3":
            return _open_trial_internal_records(competition, best_trial, "베스트 trial 내부 기록")
        if raw == "4":
            trial_id = input_fn("trial 번호 입력 (예: trial_003, q: 취소)> ").strip().lower()
            if trial_id == "q":
                return "trial 선택을 취소했습니다."
            if not re.fullmatch(r"trial_\d+", trial_id):
                output("trial_001 같은 trial 번호를 입력해주세요.")
                continue
            return _open_trial_internal_records(competition, trial_id, f"{trial_id} 내부 기록")
        output("1~4 또는 q를 입력해주세요.")


def _artifact_locations(competition: str, snapshot: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    active_trial = _active_or_recent_trial_id(snapshot) or "trial_001"
    trial_root = _trial_record_root(competition, active_trial) or root / "experiments" / competition / active_trial
    workspace = _workspace_path(competition)
    return {
        "user_view": _trial_user_view_path(competition, active_trial) or trial_root / "user_view",
        "workspace": workspace,
        "experiment_root": root / "experiments" / competition,
        "internal": trial_root / "internal",
        "submission": _submission_folder(competition, workspace),
    }


def _best_trial_id(snapshot: dict[str, Any]) -> str | None:
    best = snapshot.get("best") or {}
    value = best.get("trial_id") if isinstance(best, dict) else None
    return str(value) if value else None


def _recent_trial_id(snapshot: dict[str, Any]) -> str | None:
    latest = snapshot.get("latest") or {}
    value = snapshot.get("last_completed_trial") or (latest.get("trial_id") if isinstance(latest, dict) else None)
    return str(value) if value else None


def _active_or_recent_trial_id(snapshot: dict[str, Any]) -> str | None:
    value = snapshot.get("current_trial") or _recent_trial_id(snapshot)
    return str(value) if value else None


def _open_trial_user_artifacts(competition: str, trial_id: str | None, label: str) -> str:
    if not trial_id:
        return f"{label}를 열 수 없습니다. 대상 trial이 없습니다."
    path = _trial_user_view_path(competition, trial_id)
    if path is None:
        return f"{label}가 아직 없습니다.\n\ntrial:\n{trial_id}"
    return _open_folder_message(path, label)


def render_user_artifact_summary(competition: str, trial_id: str | None, *, label: str = "trial") -> str:
    if not trial_id:
        return f"{label} 요약을 볼 수 없습니다. 대상 trial이 없습니다."

    normalized_trial = trial_id.lower()
    row = _sqlite_trial_row(competition, normalized_trial) or {}
    comparison = _sqlite_trial_comparison_row(competition, normalized_trial) or {}
    user_view = _trial_user_view_path(competition, normalized_trial)
    files = _user_artifact_preview_files(competition, normalized_trial, user_view)
    plan = _preview_artifact_file(files.get("plan"))
    pipeline = _preview_artifact_file(files.get("pipeline"))
    scores = _preview_artifact_file(files.get("scores"))
    artifacts = _sqlite_trial_artifact_paths(competition, normalized_trial, user_facing=True)
    base_trial = _comparison_base_label(comparison) if comparison else "-"
    if base_trial == "-":
        base_trial = _infer_base_trial_from_text(plan) or "-"

    lines = [
        f"{normalized_trial} 사용자용 산출물 요약",
        "",
        f"- 구분: {label}",
        f"- 로컬 점수: {_compact_score(row.get('local_score'))}",
        f"- 제출 점수: {_compact_score(row.get('lb_score'))}",
        f"- 개선축: {row.get('change_axis') or '-'}",
        f"- base trial: {base_trial}",
        f"- 판단: {_detail_best_label(row)}",
        "",
        "실험 계획 요약:",
        plan or "- 확인 가능한 계획서가 없습니다.",
        "",
        "파이프라인 구조 요약:",
        pipeline or "- 확인 가능한 파이프라인 구조도가 없습니다.",
        "",
        "점수/결과 요약:",
        scores or "- 확인 가능한 점수/결과 문서가 없습니다.",
        "",
        "산출물 위치:",
        str(user_view) if user_view else "- 사용자용 산출물 폴더를 찾지 못했습니다.",
    ]
    if artifacts:
        lines.extend(["", "사용자용 파일:"])
        lines.extend(f"- {path}" for path in artifacts[:6])
    return "\n".join(lines)


def _sqlite_trial_row(competition: str, trial_id: str) -> dict[str, Any] | None:
    return next(
        (row for row in _sqlite_trial_rows(competition) if str(row.get("trial_id") or "").lower() == trial_id.lower()),
        None,
    )


def _sqlite_trial_comparison_row(competition: str, trial_id: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in _sqlite_trial_comparison_rows(competition)
            if str(row.get("trial_id") or "").lower() == trial_id.lower()
        ),
        None,
    )


def _user_artifact_preview_files(competition: str, trial_id: str, user_view: Path | None) -> dict[str, Path]:
    candidates = _sqlite_trial_artifact_paths(competition, trial_id, user_facing=True)
    if user_view and user_view.exists():
        candidates.extend(path for path in sorted(user_view.glob("*.md")) if path not in candidates)

    selected: dict[str, Path] = {}
    for path in candidates:
        key = _artifact_preview_key(path)
        if key and key not in selected and path.exists():
            selected[key] = path
    return selected


def _sqlite_trial_artifact_paths(competition: str, trial_id: str, *, user_facing: bool) -> list[Path]:
    db_path = default_db_path()
    if not db_path.exists():
        return []
    with state_db_connection(db_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT artifact_type, path
                FROM trial_artifacts
                WHERE competition_id = ? AND trial_id = ? AND is_user_facing = ?
                ORDER BY artifact_type, path
                """,
                [competition, trial_id, 1 if user_facing else 0],
            )
        ]
    root = project_root()
    paths: list[Path] = []
    for row in rows:
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        paths.append(path if path.is_absolute() else root / path)
    return paths


def _artifact_preview_key(path: Path) -> str | None:
    name = path.name.lower()
    if "plan" in name:
        return "plan"
    if "pipeline" in name or "structure" in name:
        return "pipeline"
    if "score" in name or "result" in name or "metric" in name:
        return "scores"
    return None


def _preview_artifact_file(path: Path | None, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    preview = "\n".join(lines[:max_lines]).strip()
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3].rstrip() + "..."
    if len(lines) > max_lines:
        preview += "\n..."
    return preview


def _infer_base_trial_from_text(text: str) -> str | None:
    match = re.search(r"(?:base|기준)\s*trial\s*`?\s*[:|]\s*`?\s*(trial_\d+)", text, re.IGNORECASE)
    return match.group(1) if match else None


def _open_user_artifacts_roots(competition: str) -> str:
    roots = _user_artifacts_roots(competition)
    if not roots:
        return "사용자용 산출물 위치가 아직 없습니다."
    primary = _preferred_user_artifacts_root(competition, roots)
    try:
        _open_folder(primary.resolve())
        opened = primary.resolve()
    except Exception as error:
        return f"전체 trial 사용자용 산출물 대표 위치를 열지 못했습니다. {error}\n\n경로:\n{primary.resolve()}"
    extra_roots = [path.resolve() for path in roots if path.resolve() != opened]
    lines = [
        "전체 trial 사용자용 산출물 대표 위치를 열었습니다.",
        "",
        "열린 경로:",
        str(opened),
    ]
    if extra_roots:
        lines.extend(["", "추가 사용자용 산출물 위치:"])
        lines.extend(str(path) for path in extra_roots)
    return "\n".join(lines)


def _preferred_user_artifacts_root(competition: str, roots: list[Path]) -> Path:
    canonical = (project_root() / "experiments" / competition).resolve()
    for path in roots:
        if path.resolve() == canonical:
            return path
    return roots[0]


def _open_trial_submission_folder(competition: str, trial_id: str | None, label: str) -> str:
    if not trial_id:
        return f"{label}를 열 수 없습니다. 대상 trial이 없습니다."
    path = _trial_submission_path(competition, trial_id)
    if path is None:
        return f"{label}가 아직 없습니다.\n\ntrial:\n{trial_id}"
    return _open_folder_message(path.parent if path.is_file() else path, label)


def _open_trial_internal_records(competition: str, trial_id: str | None, label: str) -> str:
    if not trial_id:
        return f"{label}을 열 수 없습니다. 대상 trial이 없습니다."
    trial_root = _trial_record_root(competition, trial_id)
    if trial_root is None:
        return f"{label}이 아직 없습니다.\n\ntrial:\n{trial_id}"
    internal = trial_root / "internal"
    if internal.exists():
        return _open_folder_message(internal, label)
    return _open_folder_message(
        trial_root,
        f"{label} 대체 위치",
        note=f"{trial_id}의 internal 폴더가 없어 trial 기록 루트를 열었습니다.",
    )


def _trial_user_view_path(competition: str, trial_id: str) -> Path | None:
    root = project_root()
    candidates = [
        root / "runs" / competition / trial_id,
        root / "experiments" / competition / trial_id / "user_view",
        _workspace_path(competition) / "manual_trials" / trial_id / "user_view",
    ]
    return next((path for path in candidates if path.exists()), candidates[1])


def _trial_record_root(competition: str, trial_id: str) -> Path | None:
    root = project_root()
    candidates = [
        root / "experiments" / competition / trial_id,
        _workspace_path(competition) / "manual_trials" / trial_id,
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _trial_submission_path(competition: str, trial_id: str) -> Path | None:
    root = project_root()
    record_paths = [
        _workspace_path(competition) / "manual_trials" / trial_id / "metrics.json",
        root / "experiments" / competition / trial_id / "submission_run.json",
        root / "experiments" / competition / trial_id / "submit_manifest.json",
        root / "experiments" / competition / trial_id / "metrics.json",
    ]
    for record_path in record_paths:
        record = load_json(record_path)
        submission_file = record.get("submission_file") or record.get("file_path")
        if submission_file:
            path = Path(str(submission_file)).expanduser()
            if not path.is_absolute():
                path = root / path
            if path.exists():
                return path.resolve()
    candidates = [
        _workspace_path(competition) / "manual_trials" / trial_id / "submission.csv",
        root / "experiments" / competition / trial_id / "submission.csv",
        root / "experiments" / competition / trial_id / "submission_run.json",
        root / "experiments" / competition / trial_id / "submit_manifest.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def _user_artifacts_roots(competition: str) -> list[Path]:
    root = project_root()
    candidates = [
        root / "runs" / competition,
        root / "experiments" / competition,
        _workspace_path(competition) / "manual_trials",
    ]
    seen: set[Path] = set()
    roots: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            roots.append(path)
    return roots


def _workspace_path(competition: str) -> Path:
    profile = _load_profile_safely(competition)
    if profile.get("project_root"):
        return Path(str(profile["project_root"])).expanduser().resolve()
    return (project_root() / "demo_workspaces" / competition).resolve()


def _submission_folder(competition: str, workspace: Path) -> Path:
    profile = _load_profile_safely(competition)
    artifacts = profile.get("artifacts", {}) if isinstance(profile, dict) else {}
    submissions = artifacts.get("submission", []) if isinstance(artifacts, dict) else []
    if submissions:
        return (workspace / str(submissions[0])).resolve().parent
    return (workspace / "outputs").resolve()


def _open_folder_message(path: Path, label: str, *, note: str | None = None) -> str:
    path = path.expanduser().resolve()
    if not path.exists():
        return f"{label}가 아직 없습니다.\n\n경로:\n{path}"
    if not path.is_dir():
        path = path.parent
    try:
        _open_folder(path)
    except Exception as error:
        return f"{label}를 열지 못했습니다: {error}\n\n경로:\n{path}"
    note_lines = ["", note] if note else []
    return "\n".join([f"{label}를 열었습니다.", *note_lines, "", "경로:", str(path)])


def _open_folder(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_trial_comparison_table(competition: str) -> str:
    db_path = default_db_path()
    if not db_path.exists():
        return f"SQLite DB가 아직 없습니다.\n\nDB 경로:\n{db_path}"
    rows = _sqlite_trial_comparison_rows(competition)
    if not rows:
        return f"SQLite DB에 {competition} trial 기록이 없습니다.\n\nDB 경로:\n{db_path}"

    table_rows: list[list[str]] = []
    previous_local: Any = None
    previous_submit: Any = None
    for row in rows:
        local_score = row.get("local_score")
        submit_score = row.get("lb_score")
        table_rows.append(
            [
                str(row.get("trial_id") or "-"),
                _compact_cell(_comparison_base_label(row), 10),
                _compact_score(local_score),
                _compact_score(submit_score),
                _compact_delta(local_score, previous_local),
                _compact_delta(submit_score, previous_submit),
                _compact_cell(str(row.get("change_axis") or "-"), 22),
                _compact_cell(str(row.get("decision") or "-"), 18),
                _best_label(row),
            ]
        )
        if local_score is not None:
            previous_local = local_score
        if submit_score is not None:
            previous_submit = submit_score

    return "\n".join(
        [
            "Trial 비교표",
            f"DB 경로: {db_path}",
            "",
            _render_table(
                ["trial", "base", "local", "submit", "localΔ", "submitΔ", "axis", "decision", "best"],
                table_rows,
            ),
            "",
            "참고: delta는 이전 표시 trial의 점수 대비 변화입니다. base가 '-'이면 기록된 기준 trial이 아직 없습니다.",
        ]
    )


def _sqlite_trial_comparison_rows(competition: str) -> list[dict[str, Any]]:
    db_path = default_db_path()
    if not db_path.exists():
        return []
    with state_db_connection(db_path) as connection:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    t.trial_id,
                    t.source_trial_id,
                    COALESCE(t.recommended_base_trial, d.recommended_base_trial, '') AS recommended_base_trial,
                    s.local_score,
                    s.lb_score,
                    COALESCE(NULLIF(d.active_axis, ''), NULLIF(d.change_axis, ''), NULLIF(t.primary_change_axis, ''), '') AS change_axis,
                    COALESCE(d.decision, t.plan_type, '') AS decision,
                    s.is_best_local,
                    s.is_best_lb
                FROM trials t
                LEFT JOIN trial_scores s
                    ON s.competition_id = t.competition_id AND s.trial_id = t.trial_id
                LEFT JOIN trial_decisions d
                    ON d.competition_id = t.competition_id AND d.trial_id = t.trial_id
                WHERE t.competition_id = ?
                ORDER BY t.trial_id
                """,
                [competition],
            )
        ]


def _comparison_base_label(row: dict[str, Any]) -> str:
    return str(row.get("source_trial_id") or row.get("recommended_base_trial") or "-")


def _compact_delta(value: Any, previous: Any) -> str:
    if value is None or previous is None:
        return "-"
    try:
        delta = float(value) - float(previous)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.5f}"


def render_sqlite_trial_table(competition: str) -> str:
    db_path = default_db_path()
    if not db_path.exists():
        return f"SQLite DB가 아직 없습니다.\n\nDB 경로:\n{db_path}"
    rows = _sqlite_trial_rows(competition)
    if not rows:
        return f"SQLite DB에 {competition} trial 기록이 없습니다.\n\nDB 경로:\n{db_path}"
    table_rows = [
        [
            str(row.get("trial_id") or "-"),
            _trial_status_label(row.get("status")),
            _compact_score(row.get("local_score")),
            _compact_score(row.get("lb_score")),
            _compact_cell(str(row.get("change_axis") or "-"), 18),
            _compact_cell(str(row.get("improvement_plan") or "-"), 22),
            _best_label(row),
        ]
        for row in rows
    ]
    return "\n".join(
        [
            "SQLite DB trial 요약",
            f"DB 경로: {db_path}",
            "",
            _render_table(
                ["trial", "status", "local", "submit", "axis", "plan", "best"],
                table_rows,
            ),
            "",
            "참고: SQLite 내부는 단일 표가 아니라 trials, trial_scores, trial_decisions 등을 조인해 위 표로 보여줍니다.",
        ]
    )


def render_sqlite_trial_detail(competition: str, trial_id: str) -> str:
    rows = [row for row in _sqlite_trial_rows(competition) if str(row.get("trial_id") or "").lower() == trial_id.lower()]
    if not rows:
        return f"{trial_id} 기록을 SQLite DB에서 찾지 못했습니다."
    row = rows[0]
    artifacts = _sqlite_trial_artifact_lines(competition, str(row.get("trial_id") or trial_id))
    return "\n".join(
        [
            "",
            f"trial : {row.get('trial_id') or '-'}",
            f"status : {_trial_status_label(row.get('status'))}",
            f"local : {_compact_score(row.get('local_score'))}",
            f"submit : {_compact_score(row.get('lb_score'))}",
            f"axis : {row.get('change_axis') or '-'}",
            f"plan : {row.get('improvement_plan') or '-'}",
            f"best : {_detail_best_label(row)}",
            *artifacts,
            "",
        ]
    )


def _sqlite_trial_rows(competition: str) -> list[dict[str, Any]]:
    db_path = default_db_path()
    if not db_path.exists():
        return []
    with state_db_connection(db_path) as connection:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    t.trial_id,
                    t.status,
                    s.local_score,
                    s.lb_score,
                    COALESCE(NULLIF(d.active_axis, ''), NULLIF(d.change_axis, ''), NULLIF(t.primary_change_axis, ''), '') AS change_axis,
                    COALESCE(NULLIF(t.plan_summary, ''), NULLIF(t.plan_type, ''), NULLIF(d.decision, ''), '') AS improvement_plan,
                    s.is_best_local,
                    s.is_best_lb
                FROM trials t
                LEFT JOIN trial_scores s
                    ON s.competition_id = t.competition_id AND s.trial_id = t.trial_id
                LEFT JOIN trial_decisions d
                    ON d.competition_id = t.competition_id AND d.trial_id = t.trial_id
                WHERE t.competition_id = ?
                ORDER BY t.trial_id
                """,
                [competition],
            )
        ]


def _trial_status_label(status: Any) -> str:
    value = str(status or "").casefold()
    return {
        "planned": "계획 완료",
        "ready": "계획 완료",
        "completed": "완료",
        "blocked": "중단",
    }.get(value, str(status or "-"))


def _sqlite_trial_artifact_lines(competition: str, trial_id: str) -> list[str]:
    db_path = default_db_path()
    if not db_path.exists():
        return []
    with state_db_connection(db_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT artifact_type, path, is_user_facing
                FROM trial_artifacts
                WHERE competition_id = ? AND trial_id = ?
                ORDER BY is_user_facing DESC, artifact_type, path
                """,
                [competition, trial_id],
            )
        ]
    if not rows:
        return []
    user_paths = [row for row in rows if row.get("is_user_facing")]
    internal_paths = [row for row in rows if not row.get("is_user_facing")]
    lines = ["user_artifacts :"]
    lines.extend(f"- {row.get('artifact_type')}: {row.get('path')}" for row in user_paths[:6])
    if not user_paths:
        lines.append("- 없음")
    lines.append("model_artifacts :")
    lines.extend(f"- {row.get('artifact_type')}: {row.get('path')}" for row in internal_paths[:6])
    if not internal_paths:
        lines.append("- 없음")
    return lines


def _best_label(row: dict[str, Any]) -> str:
    if row.get("is_best_lb"):
        return "Best"
    return "-"


def _detail_best_label(row: dict[str, Any]) -> str:
    if row.get("is_best_lb"):
        return "BEST"
    return "-"


def _compact_score(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def _compact_cell(value: str, limit: int) -> str:
    value = str(value or "-")
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)] + "…"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |")
    return "\n".join(lines)


def _start_experiment_dialog(competition: str, input_fn: Input, output: Output) -> str:
    output(
        "\n".join(
            [
                "실험을 시작하겠습니다. 몇 회 진행할까요?",
                "",
                "- 숫자 입력: 입력한 횟수만큼 진행",
                "- c 입력: 중단 요청 전까지 계속 진행",
                "- q 입력: 시작하지 않고 메뉴로 돌아가기",
                "",
            ]
        )
    )
    raw = input_fn("진행 횟수/c/q> ").strip().lower()
    if raw == "q":
        return "자동 실험 시작을 취소했습니다."
    if raw in {"c", "continue", "계속"}:
        return start_experiment(competition, continuous=True)
    try:
        trial_count = int(raw)
    except ValueError:
        return "진행 횟수는 양의 정수로 입력하거나, 계속 진행은 c, 취소는 q를 입력해주세요."
    if trial_count < 1:
        return "진행 횟수는 1 이상이어야 합니다."
    return start_experiment(competition, trial_count=trial_count)


def _choose_experiment(current: str, input_fn: Input, output: Output) -> str:
    try:
        experiments = load_experiments(sync=False)
    except Exception as error:
        output(f"실험 목록을 불러오지 못했습니다: {error}")
        return current
    for index, item in enumerate(experiments, start=1):
        marker = " (현재)" if item.get("competition") == current else ""
        output(f"{index}. {item.get('competition')} | {item.get('topic') or '-'} | {item.get('state') or '-'}{marker}")
    raw = input_fn("선택 (q: 취소)> ").strip().lower()
    if raw == "q":
        return current
    try:
        selected = str(experiments[int(raw) - 1]["competition"])
    except (ValueError, IndexError, KeyError):
        output("올바른 번호를 입력해주세요.")
        return current
    select_competition(selected)
    output(f"{selected} 실험을 선택했습니다.")
    return selected


def _new_experiment_dialog(input_fn: Input, output: Output) -> str | None:
    output("새 실험을 시작합니다. 중간에 q를 입력하면 취소됩니다.")
    raw_description = input_fn("URL 또는 실험을 설명해주세요.\n> ").strip()
    if raw_description.lower() == "q" or not raw_description:
        return None
    research_direction = input_fn("추가 연구 방향이나 선호가 있으면 입력하세요. 없으면 Enter.\n> ").strip()
    if research_direction.lower() == "q":
        return None

    source_path = input_fn("기존 로컬 프로젝트 경로가 있으면 입력하세요. 없으면 Enter.\n> ").strip()
    if source_path.lower() == "q":
        return None
    source_path = source_path or None

    settings = _propose_new_experiment_settings(raw_description, research_direction, source_path)
    if not settings.get("competition"):
        output("실험 ID를 해석하지 못했습니다. 설명에 대회명이나 영문 식별자를 조금 더 넣어주세요.")
        return None
    settings = _confirm_new_experiment_settings(settings, input_fn, output)
    if settings is None:
        return None

    try:
        result = prepare_workspace(
            str(settings["competition"]),
            source_path=settings.get("source_path") or None,
            topic=str(settings.get("topic") or settings["competition"]),
            platform=str(settings.get("platform") or "kaggle"),
            metric=str(settings.get("metric") or "unknown"),
            objective=str(settings.get("objective") or "maximize"),
            create_workspace=bool(settings.get("create_workspace")),
            target_column=str(settings.get("target_column") or "") or None,
            id_column=str(settings.get("id_column") or "") or None,
            required_data_files=list(settings.get("required_data_files") or []),
        )
        sync_state(competition=str(settings["competition"]))
    except Exception as error:
        output(f"새 실험 등록 실패: {error}")
        return None

    competition = str(settings["competition"])
    select_competition(competition)
    status = result.get("status") or "created"
    source = result.get("source_path") or "-"
    output(f"{competition} 실험을 등록하고 선택했습니다. 상태: {status} | 워크스페이스: {source}")
    if status in {"needs_data", "needs_review", "needs_project_path"}:
        output("바로 자동 실행하려면 데이터/실행 프로필 확인이 먼저 필요합니다.")
    return competition


def _propose_new_experiment_settings(
    description: str,
    research_direction: str = "",
    source_path: str | None = None,
) -> dict[str, Any]:
    competition = _infer_experiment_id_from_description(description)
    source_path = _normalize_workspace_source_path(source_path)
    preset = _experiment_preset_for(competition, description)
    schema = _infer_schema_from_data(source_path)
    target_column = schema.get("target_column") or _extract_column_hint(description, "target") or preset.get("target_column")
    id_column = schema.get("id_column") or _extract_column_hint(description, "id") or preset.get("id_column")
    metric_hint = _infer_metric_from_description(description, target_column)
    target_kind = _infer_target_kind_from_data(source_path, target_column)
    if preset.get("metric") and not _description_has_explicit_metric(description):
        metric = str(preset["metric"])
    else:
        metric = metric_hint if metric_hint != "unknown" else str(preset.get("metric") or _metric_for_target_kind(target_kind) or "unknown")
    objective = _objective_for_metric(metric, preset.get("objective"))
    required_files = _infer_required_data_files(description, source_path)
    return {
        "competition": competition,
        "topic": _infer_topic_from_description(description, competition, preset),
        "platform": "kaggle",
        "source_path": source_path,
        "create_workspace": not bool(source_path),
        "target_column": target_column,
        "id_column": id_column,
        "metric": metric,
        "objective": objective,
        "required_data_files": required_files or list(preset.get("required_data_files") or []),
        "research_direction": research_direction,
        "description": description,
    }


def _confirm_new_experiment_settings(
    settings: dict[str, Any],
    input_fn: Input,
    output: Output,
) -> dict[str, Any] | None:
    while True:
        output(_render_new_experiment_settings(settings))
        raw = input_fn("선택 > ").strip().lower()
        if raw == "q":
            return None
        if raw == "":
            return settings
        updated = _edit_new_experiment_setting(settings, raw, input_fn, output)
        if updated is None:
            output("수정할 번호를 다시 선택해주세요.")
            continue
        settings = updated


def _render_new_experiment_settings(settings: dict[str, Any]) -> str:
    files = ", ".join(settings.get("required_data_files") or []) or "-"
    source = settings.get("source_path") or "-"
    scaffold = "예" if settings.get("create_workspace") else "아니오"
    direction = settings.get("research_direction") or "-"
    return "\n".join(
        [
            "",
            "에이전트 분석 결과:",
            "",
            f"- 실험 ID: {settings.get('competition') or '-'}",
            f"- 주제명: {settings.get('topic') or '-'}",
            f"- 플랫폼: {settings.get('platform') or '-'}",
            f"- 기존 프로젝트 경로: {source}",
            f"- 새 워크스페이스 생성: {scaffold}",
            f"- target 컬럼: {settings.get('target_column') or '-'}",
            f"- ID 컬럼: {settings.get('id_column') or '-'}",
            f"- 평가 지표: {settings.get('metric') or '-'}",
            f"- 최적화 방향: {settings.get('objective') or '-'}",
            f"- 필수 데이터 파일: {files}",
            f"- 연구 방향: {direction}",
            "",
            "Enter: 이대로 등록",
            "번호 입력: 수정",
            "q: 취소",
            "",
            "1. 실험 ID",
            "2. 주제명",
            "3. target/id 컬럼",
            "4. 평가 지표/최적화 방향",
            "5. 필수 데이터 파일",
            "6. 기존 프로젝트 경로/워크스페이스 생성",
            "7. 연구 방향",
        ]
    )


def _edit_new_experiment_setting(
    settings: dict[str, Any],
    raw: str,
    input_fn: Input,
    output: Output,
) -> dict[str, Any] | None:
    updated = dict(settings)
    if raw == "1":
        value = input_fn("실험 ID> ").strip()
        if value.lower() == "q":
            return updated
        updated["competition"] = _normalize_competition_id(value) or updated.get("competition")
        return updated
    if raw == "2":
        value = input_fn("주제명> ").strip()
        if value.lower() == "q":
            return updated
        updated["topic"] = value or updated.get("topic")
        return updated
    if raw == "3":
        target = input_fn("target 컬럼(없으면 Enter)> ").strip()
        if target.lower() == "q":
            return updated
        identifier = input_fn("ID 컬럼(없으면 Enter)> ").strip()
        if identifier.lower() == "q":
            return updated
        updated["target_column"] = target or None
        updated["id_column"] = identifier or None
        return updated
    if raw == "4":
        metric = input_fn("평가 지표(예: accuracy, roc_auc, rmse)> ").strip()
        if metric.lower() == "q":
            return updated
        objective = input_fn("최적화 방향(maximize/minimize, Enter: 자동)> ").strip().lower()
        if objective == "q":
            return updated
        updated["metric"] = metric or updated.get("metric") or "unknown"
        updated["objective"] = objective if objective in {"maximize", "minimize"} else _objective_for_metric(updated["metric"])
        return updated
    if raw == "5":
        files = input_fn("필수 데이터 파일(쉼표로 구분)> ").strip()
        if files.lower() == "q":
            return updated
        updated["required_data_files"] = [item.strip() for item in files.split(",") if item.strip()]
        return updated
    if raw == "6":
        source_path = input_fn("기존 프로젝트 경로(없으면 Enter)> ").strip()
        if source_path.lower() == "q":
            return updated
        scaffold = input_fn("새 워크스페이스를 생성할까요? (Y/n)> ").strip().lower()
        if scaffold == "q":
            return updated
        source_path = _normalize_workspace_source_path(source_path)
        updated["source_path"] = source_path or None
        updated["create_workspace"] = scaffold not in {"n", "no", "아니오"} and not bool(source_path)
        if source_path:
            schema = _infer_schema_from_data(source_path)
            updated["target_column"] = updated.get("target_column") or schema.get("target_column")
            updated["id_column"] = updated.get("id_column") or schema.get("id_column")
            updated["required_data_files"] = _infer_required_data_files(str(updated.get("description") or ""), source_path)
        return updated
    if raw == "7":
        value = input_fn("연구 방향(없으면 Enter)> ").strip()
        if value.lower() == "q":
            return updated
        updated["research_direction"] = value
        return updated
    return None


def _infer_experiment_id_from_description(description: str) -> str:
    value = description.strip()
    match = re.search(r"kaggle\.com/(?:c|competitions)/([^/?#\s]+)", value)
    if match:
        return _normalize_competition_id(match.group(1))
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,80}", value):
        return _normalize_competition_id(value)
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), value)
    subject = re.split(
        r"(?i)(?:\btarget\b|\blabel\b|\bid\s+column\b|\bmetric\b|\bevaluation\b|[.?!])",
        first_line,
        maxsplit=1,
    )[0]
    ascii_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", subject)
    if ascii_words:
        return _normalize_competition_id("-".join(ascii_words[:6]))
    return "research_experiment"


def _experiment_preset_for(competition: str, description: str = "") -> dict[str, Any]:
    if competition in KNOWN_EXPERIMENT_PRESETS:
        return dict(KNOWN_EXPERIMENT_PRESETS[competition])
    lowered = f"{competition} {description}".lower()
    if "spaceship" in lowered and "titanic" in lowered:
        return dict(KNOWN_EXPERIMENT_PRESETS["spaceship-titanic"])
    if "titanic" in lowered:
        return dict(KNOWN_EXPERIMENT_PRESETS["titanic"])
    if "house-prices" in lowered or ("house" in lowered and "price" in lowered):
        return dict(KNOWN_EXPERIMENT_PRESETS["house-prices-advanced-regression-techniques"])
    if "digit-recognizer" in lowered or "digit recognizer" in lowered:
        return dict(KNOWN_EXPERIMENT_PRESETS["digit-recognizer"])
    return {}


def _extract_column_hint(description: str, role: str) -> str | None:
    role_patterns = {
        "target": r"(?:target|label|타깃|타겟|목표\s*컬럼|정답\s*컬럼)",
        "id": r"(?:id|identifier|식별자|ID\s*컬럼)",
    }
    role_pattern = role_patterns.get(role)
    if not role_pattern:
        return None
    patterns = [
        rf"{role_pattern}\s*(?:column|컬럼|col)?\s*(?:is|는|은|:|=)\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        rf"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*(?:is|가|이)?\s*(?:the\s+)?{role_pattern}\s*(?:column|컬럼|col)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _infer_topic_from_description(description: str, competition: str, preset: dict[str, Any] | None = None) -> str:
    if preset and preset.get("topic"):
        return str(preset["topic"])
    text = description.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return competition.replace("-", " ").replace("_", " ").title()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first_line) > 80:
        first_line = first_line[:77].rstrip() + "..."
    return first_line or competition.replace("-", " ").replace("_", " ").title()


def _infer_metric_from_description(description: str, target_column: str | None = None) -> str:
    lowered = description.lower()
    if "roc" in lowered or "auc" in lowered or "area under" in lowered:
        return "roc_auc"
    if "logloss" in lowered or "log loss" in lowered or "cross entropy" in lowered:
        return "log_loss"
    if "rmse" in lowered:
        return "rmse"
    if "rmsle" in lowered:
        return "rmsle"
    if "mae" in lowered:
        return "mae"
    if "accuracy" in lowered or "정확도" in lowered:
        return "accuracy"
    if "f1" in lowered:
        return "f1"
    if "classification" in lowered or "분류" in lowered or "binary" in lowered or "multiclass" in lowered:
        return "accuracy"
    if "regression" in lowered or "회귀" in lowered or "price" in lowered or "가격" in lowered:
        return "rmse"
    if target_column and target_column.lower() in {"survived", "label", "target", "class", "exited"}:
        return "accuracy"
    return "unknown"


def _description_has_explicit_metric(description: str) -> bool:
    lowered = description.lower()
    return any(
        token in lowered
        for token in [
            "auc",
            "roc",
            "accuracy",
            "정확도",
            "logloss",
            "log loss",
            "cross entropy",
            "rmse",
            "rmsle",
            "mae",
            "mse",
            "f1",
        ]
    )


def _objective_for_metric(metric: str | None, fallback: Any = None) -> str:
    if fallback in {"maximize", "minimize"}:
        return str(fallback)
    return "minimize" if str(metric or "").lower() in {"rmse", "mae", "mse", "rmsle", "log_loss"} else "maximize"


def _infer_required_data_files(description: str, source_path: str | None) -> list[str]:
    source = Path(source_path).expanduser() if source_path else None
    if source and source.exists():
        csv_files = _discover_data_file_names(source)
        if csv_files:
            preferred = [name for name in ["train.csv", "test.csv", "sample_submission.csv", "gender_submission.csv"] if name in csv_files]
            rest = [name for name in csv_files if name not in preferred]
            return (preferred + rest)[:8]
    preset = _experiment_preset_for(_infer_experiment_id_from_description(description), description)
    if preset.get("required_data_files"):
        return list(preset["required_data_files"])
    lowered = description.lower()
    return ["train.csv", "test.csv", "sample_submission.csv"]


def _infer_schema_from_data(source_path: str | None) -> dict[str, str | None]:
    if not source_path:
        return {"target_column": None, "id_column": None}
    source = Path(source_path).expanduser()
    train_header = _csv_header(_find_data_file(source, ["train.csv"]))
    test_header = _csv_header(_find_data_file(source, ["test.csv"]))
    sample_header = _csv_header(_find_data_file(source, ["sample_submission.csv"]))
    if not sample_header:
        sample_header = _csv_header(_find_data_file(source, ["gender_submission.csv"]))
    return _infer_target_id_from_headers(train_header, test_header, sample_header)


def _infer_target_kind_from_data(source_path: str | None, target_column: Any) -> str | None:
    if not source_path or not target_column:
        return None
    source = Path(source_path).expanduser()
    train_path = _find_data_file(source, ["train.csv"])
    if train_path is None:
        return None
    values: list[str] = []
    try:
        with train_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = str(row.get(str(target_column)) or "").strip()
                if value:
                    values.append(value)
                if len(values) >= 200:
                    break
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if not values:
        return None
    unique = set(values)
    if unique <= {"0", "1", "0.0", "1.0", "False", "True", "false", "true"}:
        return "binary_classification"
    numeric_values: list[float] = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except ValueError:
            return "classification" if len(unique) <= 30 else None
    if len(set(numeric_values)) <= 30 and all(float(value).is_integer() for value in numeric_values):
        return "classification"
    return "regression"


def _metric_for_target_kind(target_kind: str | None) -> str | None:
    if target_kind in {"binary_classification", "classification"}:
        return "accuracy"
    if target_kind == "regression":
        return "rmse"
    return None


def _normalize_workspace_source_path(source_path: str | None) -> str | None:
    if not source_path:
        return None
    source = Path(source_path).expanduser()
    if source.is_file() and source.suffix.lower() == ".csv":
        source = source.parent
    if source.name.casefold() in {"data", "input"} and _discover_data_file_names(source):
        return str(source.parent)
    return str(source)


def _candidate_data_roots(source: Path) -> list[Path]:
    roots = [source / "data", source]
    if source.name.casefold() in {"data", "input"}:
        roots = [source, source.parent / "data", source.parent]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _find_data_file(source: Path, names: list[str]) -> Path | None:
    lowered_names = {name.casefold() for name in names}
    for root in _candidate_data_roots(source):
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
        if root.exists():
            for candidate in root.glob("*.csv"):
                if candidate.name.casefold() in lowered_names:
                    return candidate
    return None


def _discover_data_file_names(source: Path) -> list[str]:
    discovered: list[str] = []
    for root in _candidate_data_roots(source):
        if not root.exists():
            continue
        for path in sorted(root.glob("*.csv"), key=lambda item: item.name.casefold()):
            if path.name not in discovered:
                discovered.append(path.name)
    return discovered


def _csv_header(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return next(csv.reader(handle), [])
    except (OSError, StopIteration, UnicodeDecodeError, csv.Error):
        return []


def _infer_target_id_from_headers(
    train_header: list[str],
    test_header: list[str],
    sample_header: list[str],
) -> dict[str, str | None]:
    train = [column for column in train_header if column]
    test = [column for column in test_header if column]
    sample = [column for column in sample_header if column]
    train_only = [column for column in train if column not in set(test)]
    target_preferences = ["target", "label", "Survived", "Exited", "SalePrice", "y"]
    target = next((name for name in target_preferences if name in train_only), None)
    if target is None and len(sample) >= 2:
        target = sample[1]
    if target is None and train_only:
        target = train_only[0]

    common = [column for column in train if column in set(test)]
    id_preferences = ["id", "ID", "PassengerId", "CustomerId", "Id"]
    identifier = next((name for name in id_preferences if name in common or name in sample), None)
    if identifier is None:
        identifier = next((name for name in common if name.lower().endswith("id") or "id" in name.lower()), None)
    if identifier is None and sample:
        identifier = sample[0]
    return {"target_column": target, "id_column": identifier}


def _question_dialog(competition: str, snapshot: dict[str, Any], input_fn: Input, output: Output) -> str:
    output("에이전트 질문 모드입니다. 메뉴로 돌아가려면 q를 입력하세요.")
    answered = 0
    while True:
        question = input_fn("\n질문 (q: 메뉴로 돌아가기)> ").strip()
        if question.lower() == "q":
            return "에이전트 질문 모드를 종료했습니다." if answered else "질문을 취소했습니다."
        if not question:
            output("질문을 입력하거나 q로 메뉴에 돌아가세요.")
            continue
        trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial")
        result = answer_experiment_question(competition, trial_id, question)
        lines = []
        if result.get("warning"):
            lines.append(str(result["warning"]))
        lines.append(f"[{result.get('mode')}]\n{result.get('answer')}")
        output("\n".join(["", "답변:", *lines]))
        answered += 1


def _insight_dialog(competition: str, snapshot: dict[str, Any], input_fn: Input, output: Output) -> str:
    trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial") or "trial_001"
    next_trial = snapshot.get("next_trial") or _next_trial_after(str(trial_id)) or "다음"
    existing = _latest_user_insight(competition, str(trial_id))
    if existing:
        output(
            "\n".join(
                [
                    "이미 인사이트가 제공 되었습니다",
                    f": {existing}",
                    "",
                    _format_insight_plan_message(existing, str(next_trial), include_insight=False),
                    "",
                    "기존 인사이트 유지: q, 인사이트 변경: 내용을 입력하세요",
                ]
            )
        )
    insight = input_fn("인사이트 > ").strip()
    if insight.lower() == "q" or not insight:
        return "인사이트 입력을 취소했습니다. 기존 인사이트는 유지됩니다." if existing else "인사이트 입력을 취소했습니다."
    result = submit_human_insight(competition, str(trial_id), insight=insight)
    if not result.get("ok"):
        return result.get("message", "기록 실패")
    _keep_latest_user_insight(competition, str(trial_id), result.get("data", {}).get("feedback"))
    record = result.get("data", {}).get("insight") or {}
    return "\n".join(
        [
            _format_insight_plan_message(insight, str(next_trial)),
            f"- 인사이트 상태: {record.get('status') or 'pending'}",
            f"- 개선축: {record.get('axis') or '해석 대기'}",
        ]
    )


def _format_insight_plan_message(insight: str, next_trial: str, *, include_insight: bool = True) -> str:
    lines = [
        "다음 실험 계획 단계에서 에이전트가 인사이트를 개선축으로 해석해 반영합니다.",
        "",
    ]
    if include_insight:
        lines.extend([insight, ""])
    lines.extend(
        [
            f"- {next_trial} 실험 반영 예정",
            f"- 적용 개선안 : {_planned_improvement_from_insight(insight)}",
        ]
    )
    return "\n".join(lines)


def _user_feedback_path(competition: str) -> Path:
    return project_root() / "memory" / competition / "user_feedback.jsonl"


def _read_user_feedback_rows(competition: str) -> list[dict[str, Any]]:
    path = _user_feedback_path(competition)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _is_next_trial_user_insight(row: dict[str, Any], competition: str, trial_id: str) -> bool:
    return (
        row.get("competition") == competition
        and row.get("trial_id") == trial_id
        and row.get("topic") == "user_insight"
        and row.get("scope") == "next_trial"
    )


def _latest_user_insight(competition: str, trial_id: str) -> str | None:
    for row in reversed(_read_user_feedback_rows(competition)):
        if _is_next_trial_user_insight(row, competition, trial_id):
            value = str(row.get("user_feedback") or "").strip()
            return value or None
    return None


def _keep_latest_user_insight(competition: str, trial_id: str, latest_feedback: dict[str, Any] | None) -> None:
    if not latest_feedback:
        return
    path = _user_feedback_path(competition)
    rows = _read_user_feedback_rows(competition)
    kept = [row for row in rows if not _is_next_trial_user_insight(row, competition, trial_id)]
    kept.append(latest_feedback)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n", encoding="utf-8")


def _next_trial_after(trial_id: str) -> str | None:
    match = re.fullmatch(r"trial_(\d+)", trial_id)
    if not match:
        return None
    return f"trial_{int(match.group(1)) + 1:03d}"


def _planned_improvement_from_insight(insight: str) -> str:
    result = interpret_user_insight(insight)
    intent = result.get("implementation_intent") if isinstance(result.get("implementation_intent"), dict) else {}
    return str(intent.get("change") or "계획 에이전트가 다음 trial의 구체적인 변경안으로 해석합니다.")


def _feedback_dialog(competition: str, input_fn: Input, output: Output) -> str:
    result = list_pending_requests(competition, sync=False)
    requests = list(result.get("data", {}).get("requests") or [])
    if not requests:
        return "피드백 요청이 없습니다."
    for index, request in enumerate(requests, start=1):
        output(f"{index}. {request.get('title') or request.get('type')} | {request.get('question') or request.get('message')}")
    raw = input_fn("답변할 요청 (q: 취소)> ").strip().lower()
    if raw == "q":
        return "피드백 요청 확인을 취소했습니다."
    try:
        request = requests[int(raw) - 1]
    except (ValueError, IndexError):
        return "올바른 번호를 입력해주세요."
    answer = input_fn("답변 (q: 취소)> ").strip()
    if answer.lower() == "q" or not answer:
        return "피드백 답변 입력을 취소했습니다."
    response = respond_to_request(str(request["request_id"]), free_text=answer)
    return "답변을 기록했습니다. 다음 실험에 반영하겠습니다." if response.get("ok") else response.get("message", "기록 실패")


def _manual_trial_rows(competition: str) -> list[dict[str, Any]]:
    root = project_root() / "demo_workspaces" / competition / "manual_trials"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("trial_*/metrics.json")):
        row = load_json(path)
        if row:
            rows.append(
                {
                    "trial_id": row.get("trial_id") or path.parent.name,
                    "status": "completed",
                    "local_score": row.get("local_score") or row.get("cv_score"),
                    "lb_score": row.get("kaggle_lb_score") or row.get("lb_score"),
                    "change_axis": row.get("change_axis"),
                }
            )
    return rows


def _latest_trial(db_trials: list[dict[str, Any]], manual: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed_db = [
        row
        for row in db_trials
        if str(row.get("status") or "").casefold() not in {"planned", "ready"}
        and (
            row.get("local_score") is not None
            or row.get("lb_score") is not None
            or str(row.get("status") or "").casefold() == "completed"
        )
    ]
    if completed_db:
        return max(completed_db, key=lambda row: str(row.get("trial_id") or ""))
    submitted = [row for row in manual if row.get("lb_score") is not None]
    if submitted:
        return max(submitted, key=lambda row: str(row.get("trial_id") or ""))
    merged = {str(row.get("trial_id")): row for row in db_trials if row.get("trial_id")}
    merged.update({str(row.get("trial_id")): row for row in manual if row.get("trial_id")})
    return merged[sorted(merged)[-1]] if merged else None


def _best_trial(db_trials: list[dict[str, Any]], manual: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [*db_trials, *manual]
    scored = [row for row in rows if row.get("lb_score") is not None]
    if scored:
        return max(scored, key=lambda row: float(row["lb_score"]))
    return None


def _infer_next_trial(manual: list[dict[str, Any]]) -> str | None:
    completed = {str(row.get("trial_id")) for row in manual if row.get("lb_score") is not None}
    manual_next = next((trial_id for trial_id in TRIAL_IDS if trial_id not in completed), None)
    if manual_next:
        return manual_next
    trial_ids = sorted(str(row.get("trial_id")) for row in manual if row.get("trial_id"))
    return next_trial_id(trial_ids[-1]) if trial_ids else None


def _infer_start_trial(competition: str) -> str | None:
    try:
        snapshot = get_experiment(competition, sync=False)
    except Exception:
        snapshot = {}
    experiment = dict((snapshot.get("data") or {}).get("experiment") or {}) if snapshot.get("ok") else {}
    if experiment.get("next_trial_id"):
        return str(experiment["next_trial_id"])
    trials = list((snapshot.get("data") or {}).get("trials") or []) if snapshot.get("ok") else []
    planned = [
        row
        for row in trials
        if str(row.get("status") or "").casefold() in {"planned", "ready"}
        and row.get("trial_id")
    ]
    if planned:
        return str(max(planned, key=lambda row: str(row.get("trial_id") or "")).get("trial_id"))
    trial_ids = sorted(str(row.get("trial_id")) for row in trials if row.get("trial_id"))
    if not trial_ids:
        manual = _manual_trial_rows(competition)
        manual_next = _infer_next_trial(manual)
        return manual_next or "trial_001"
    return next_trial_id(trial_ids[-1])


def next_trial_id(trial_id: str) -> str:
    prefix, _, number = trial_id.rpartition("_")
    try:
        return f"{prefix}_{int(number) + 1:03d}"
    except ValueError:
        return "trial_002"


def _display_state(db_state: Any, loop: dict[str, Any], manual: list[dict[str, Any]]) -> str:
    state = str(loop.get("status") or "")
    labels = {
        "starting": "시작 중",
        "running": "실행 중",
        "resuming": "재개 중",
        "paused": "대기 중",
        "completed": "완료",
        "failed": "오류",
    }
    if loop.get("pause_requested") and state in {"starting", "running", "resuming"}:
        return "중단 대기 중"
    if state:
        return labels.get(state, state)
    if _infer_next_trial(manual):
        return "대기 중"
    return str(db_state or "대기 중")


def _score(value: Any) -> str:
    return "-" if value is None else str(value)


def _load_profile_safely(competition: str) -> dict[str, Any]:
    try:
        return load_execution_profile(competition)
    except (FileNotFoundError, ValueError):
        return {}


def _filesystem_experiments() -> list[dict[str, Any]]:
    root = project_root() / "competitions"
    if not root.exists():
        return []
    experiments = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        state = simple_yaml.load(path / "state.yaml", default={})
        competition = state.get("competition") or {}
        source = load_json(path / "workspace_source.json")
        experiments.append(
            {
                "competition": path.name,
                "topic": competition.get("topic") or source.get("topic") or path.name,
                "platform": competition.get("platform") or source.get("platform"),
                "metric": competition.get("metric"),
                "objective": competition.get("objective"),
                "state": source.get("status") or competition.get("status") or "registered",
            }
        )
    return experiments


def _filesystem_topic(competition: str) -> str | None:
    state = simple_yaml.load(competition_dir(competition) / "state.yaml", default={})
    source = load_json(competition_dir(competition) / "workspace_source.json")
    return (state.get("competition") or {}).get("topic") or source.get("topic")


def _normalize_competition_id(raw: str) -> str:
    value = raw.strip().rstrip("/")
    match = re.search(r"kaggle\.com/(?:c|competitions)/([^/?#]+)", value)
    if match:
        value = match.group(1)
    value = value.split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_").lower()
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research Agent 대화형 CLI")
    parser.add_argument("--no-sync", action="store_true", help="시작 시 상태 DB 동기화를 건너뜁니다.")
    parser.add_argument("--status", action="store_true", help="선택된 실험 상태만 출력합니다.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        print(render_snapshot(experiment_snapshot(selected_competition(), sync=not args.no_sync)))
        return 0
    return run_menu(sync_on_start=not args.no_sync)


if __name__ == "__main__":
    raise SystemExit(main())
