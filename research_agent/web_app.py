from __future__ import annotations

import html
import json
import os
import re
import zipfile
from io import BytesIO
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from .cli_app import (
    _artifact_locations,
    _best_label,
    _compact_cell,
    _compact_score,
    _filesystem_topic,
    _format_insight_plan_message,
    _keep_latest_user_insight,
    _latest_user_insight,
    _load_profile_safely,
    _normalize_competition_id,
    _normalize_workspace_source_path,
    _propose_new_experiment_settings,
    _sqlite_trial_rows,
    check_dacon_submission_limit,
    delete_experiment,
    experiment_snapshot,
    list_pending_requests,
    load_experiments,
    render_sqlite_trial_detail,
    render_snapshot,
    request_experiment_stop,
    select_competition,
    selected_competition,
    set_dacon_submission_limit_override,
    dacon_auto_submit_allowed,
    refresh_dacon_competition_docs,
    set_dacon_auto_submit,
    set_dacon_team_name,
    start_experiment,
    submit_trial_manually,
)
from .chat_history import (
    answer_chat_question,
    chat_history_snapshot,
    start_new_chat,
)
from .interface_contract import respond_to_request, submit_human_insight
from .paths import competition_dir, project_root
from .state_db import default_db_path, state_db_connection
from .state_db_sync import sync_state_db
from .workspace_preparer import prepare_workspace, refresh_workspace_inventory
from .user_insight_policy import latest_user_insight_record, user_insight_target_trial_ids


def app_state(*, sync: bool = False) -> dict[str, Any]:
    competition = selected_competition()
    try:
        snapshot = experiment_snapshot(competition, sync=sync)
        ok = True
        error = None
    except Exception as exc:
        snapshot = {
            "competition": competition,
            "topic": competition,
            "state": "상태 확인 실패",
            "latest": {},
            "best": {},
            "pending_request_count": 0,
        }
        ok = False
        error = str(exc)
    try:
        experiments = load_experiments(sync=False)
    except Exception:
        experiments = [{"competition": competition, "topic": competition, "state": "unknown"}]
    try:
        pending = list_pending_requests(competition, sync=False)
        pending_requests = list(pending.get("data", {}).get("requests") or [])
    except Exception:
        pending_requests = []
    snapshot["pending_request_count"] = len(pending_requests) if pending_requests else snapshot.get("pending_request_count", 0)
    return {
        "ok": ok,
        "error": error,
        "snapshot": snapshot,
        "experiments": experiments,
        "pending_requests": pending_requests,
        "text": render_status_text(snapshot),
        "existing_insight": current_user_insight(snapshot),
    }


def dacon_submission_limit_snapshot(competition: str) -> dict[str, Any]:
    profile = _load_profile_safely(competition)
    platform = str(profile.get("platform") or "").casefold()
    if platform != "dacon":
        return {"ok": True, "applicable": False}
    try:
        result = check_dacon_submission_limit(competition)
    except Exception as exc:  # noqa: BLE001 - a live network/scrape call, must not break the dashboard
        return {
            "ok": True,
            "applicable": True,
            "status": "unknown",
            "daily_submission_limit": None,
            "remaining": None,
            "next_reset_estimate": None,
            "message": f"제출 한도 조회 중 오류가 발생했습니다: {exc}",
            "auto_submit": dacon_auto_submit_allowed(competition),
        }
    return {"ok": True, "applicable": True, "auto_submit": dacon_auto_submit_allowed(competition), **result}


def render_status_text(snapshot: dict[str, Any]) -> str:
    return render_snapshot(snapshot)


class ResearchAgentHandler(BaseHTTPRequestHandler):
    server_version = "ResearchAgentHTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/status":
            self._send_json(app_state())
            return
        if parsed.path == "/api/dacon-submission-limit":
            query = parse_qs(parsed.query)
            competition = clean((query.get("competition") or [""])[0]) or selected_competition()
            self._send_json(dacon_submission_limit_snapshot(competition))
            return
        if parsed.path == "/api/chat/history":
            query = parse_qs(parsed.query)
            session_id = clean((query.get("session_id") or [""])[0]) or None
            try:
                self._send_json(
                    {
                        "ok": True,
                        **chat_history_snapshot(
                            selected_competition(),
                            session_id=session_id,
                        ),
                    }
                )
            except ValueError as error:
                self._send_json(
                    {"ok": False, "message": str(error)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return
        if parsed.path == "/artifact":
            query = parse_qs(parsed.query)
            relative_path = (query.get("path") or [""])[0]
            try:
                self._send_html(render_artifact_page(relative_path))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid artifact path")
            return
        if parsed.path == "/api/artifact":
            query = parse_qs(parsed.query)
            relative_path = (query.get("path") or [""])[0]
            try:
                self._send_json({"ok": True, **load_artifact_content(relative_path)})
            except FileNotFoundError:
                self._send_json({"ok": False, "message": "산출물을 찾을 수 없습니다."}, status=HTTPStatus.NOT_FOUND)
            except ValueError:
                self._send_json({"ok": False, "message": "잘못된 산출물 경로입니다."}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/":
            query = parse_qs(parsed.query)
            self._send_html(
                render_home(
                    app_state(sync=(query.get("sync") or [""])[0] == "1"),
                    message=(query.get("message") or [""])[0],
                    answer=(query.get("answer") or [""])[0],
                )
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/action":
            self._handle_action()
            return
        if parsed.path == "/api/dacon-submission-limit":
            data = self._form_data()
            competition = clean((data.get("competition") or [""])[0]) or selected_competition()
            raw_value = clean((data.get("value") or [""])[0])
            try:
                value = int(raw_value)
                if value <= 0:
                    raise ValueError
            except ValueError:
                self._send_json(
                    {"ok": False, "message": "1 이상의 정수를 입력해주세요."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            set_dacon_submission_limit_override(competition, value)
            self._send_json(dacon_submission_limit_snapshot(competition))
            return
        if parsed.path == "/api/submit-trial":
            data = self._form_data()
            competition = clean((data.get("competition") or [""])[0]) or selected_competition()
            trial_id = clean((data.get("trial_id") or [""])[0])
            if not trial_id:
                self._send_json({"ok": False, "message": "trial_id가 필요합니다."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = submit_trial_manually(competition, trial_id)
            except Exception as error:
                self._send_json(
                    {"ok": False, "message": f"제출 중 오류가 발생했습니다: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            # submit_trial_manually writes submission_run.json (and the lb_score
            # it just read back from DACON) straight to disk; nothing else in
            # this request pulls that into the state DB the dashboard reads
            # from. Without this, the score sat correctly on disk but stayed
            # invisible until someone happened to click the manual refresh
            # link (real incident this fixes).
            try:
                sync_state_db(competition)
            except Exception:
                pass  # a submission that already succeeded must not be reported as failed over a display refresh
            self._send_json(result, status=HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/dacon-auto-submit":
            data = self._form_data()
            competition = clean((data.get("competition") or [""])[0]) or selected_competition()
            enabled = (data.get("enabled") or [""])[0] in {"1", "true", "on"}
            set_dacon_auto_submit(competition, enabled)
            self._send_json(dacon_submission_limit_snapshot(competition))
            return
        if parsed.path == "/api/insight":
            data = self._form_data()
            result = record_insight((data.get("insight") or [""])[0])
            self._send_json(result, status=HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/question":
            data = self._form_data()
            question = clean((data.get("question") or [""])[0])
            if not question:
                self._send_json({"ok": False, "message": "질문을 입력해주세요."}, status=HTTPStatus.BAD_REQUEST)
                return
            session_id = clean((data.get("session_id") or [""])[0]) or None
            try:
                result = ask_agent_exchange(question, session_id=session_id)
            except Exception as error:
                self._send_json(
                    {"ok": False, "message": f"답변 또는 대화 기록 저장에 실패했습니다: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_json(
                {
                    "ok": True,
                    "question": question,
                    "answer": result["rendered_answer"],
                    "history": result["history"],
                }
            )
            return
        if parsed.path == "/api/chat/session":
            try:
                self._send_json({"ok": True, **start_new_chat(selected_competition())})
            except Exception as error:
                self._send_json(
                    {"ok": False, "message": f"새 대화를 만들지 못했습니다: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        if parsed.path == "/api/upload-data":
            self._handle_upload_data()
            return
        if parsed.path == "/api/register-experiment":
            data = self._form_data()
            message, competition = create_experiment_from_form(data)
            self._send_json({"ok": competition is not None, "message": message, "competition": competition})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_upload_data(self) -> None:
        competition = selected_competition()
        if not competition:
            self._send_json({"ok": False, "message": "선택된 실험이 없습니다."}, status=HTTPStatus.BAD_REQUEST)
            return
        target_dir = workspace_data_dir(competition)
        if target_dir is None:
            self._send_json(
                {"ok": False, "message": "이 실험은 아직 워크스페이스 경로가 없습니다. 실험을 다시 등록해주세요."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        files = parse_multipart_files(self.headers.get("Content-Type") or "", body)
        if not files:
            self._send_json({"ok": False, "message": "업로드할 파일이 없습니다."}, status=HTTPStatus.BAD_REQUEST)
            return
        result = save_uploaded_files(files, target_dir)
        if result.get("ok") and result.get("saved"):
            refresh_workspace_inventory(competition, target_dir.parent)
        self._send_json(result)

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args), flush=True)

    def _handle_action(self) -> None:
        data = self._form_data()
        action = (data.get("action") or [""])[0]
        message = ""
        answer = ""
        try:
            if action == "select":
                competition = clean((data.get("competition") or [""])[0])
                if competition:
                    select_competition(competition)
                    message = f"{competition} 실험을 선택했습니다."
            elif action == "start":
                message = start_from_form(data)
            elif action == "stop":
                message = request_experiment_stop(selected_competition())
            elif action == "refresh":
                message = "상태를 다시 불러왔습니다."
            elif action == "question":
                question = clean((data.get("question") or [""])[0])
                answer = ask_agent(question)
                message = "에이전트 답변을 생성했습니다." if answer else "질문을 입력해주세요."
            elif action == "insight":
                result = record_insight((data.get("insight") or [""])[0])
                message = result["message"]
            elif action == "feedback":
                message = record_feedback_response(data)
            elif action == "new_experiment_analyze":
                settings, message = analyze_new_experiment_from_form(data)
                self._send_html(render_home(app_state(), message=message, new_experiment_settings=settings))
                return
            elif action == "new_experiment":
                message, _ = create_experiment_from_form(data)
            elif action == "delete_experiment":
                message = delete_experiment_from_form(data)
            else:
                message = "알 수 없는 요청입니다."
        except Exception as exc:
            message = f"요청 처리 중 오류가 발생했습니다: {exc}"
        self._redirect_home(message=message, answer=answer)

    def _form_data(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return parse_qs(raw, keep_blank_values=True)

    def _redirect_home(self, *, message: str = "", answer: str = "") -> None:
        query = urlencode({"message": message, "answer": answer})
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/?{query}")
        self.end_headers()

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def start_from_form(data: dict[str, list[str]]) -> str:
    continuous = (data.get("continuous") or [""])[0] == "1"
    if continuous:
        return start_experiment(selected_competition(), continuous=True)
    raw_count = clean((data.get("trial_count") or ["1"])[0]) or "1"
    try:
        trial_count = int(raw_count)
    except ValueError:
        return "실험 횟수는 숫자로 입력해주세요."
    if trial_count < 1:
        return "실험 횟수는 1 이상이어야 합니다."
    return start_experiment(selected_competition(), trial_count=trial_count)


def record_insight(insight: str) -> dict[str, Any]:
    insight = insight.strip()
    if not insight:
        return {"ok": False, "message": "인사이트를 입력해주세요."}
    snapshot = app_state()["snapshot"]
    trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial") or "trial_001"
    next_trial = snapshot.get("next_trial") or _next_trial_after(str(trial_id)) or "다음"
    result = submit_human_insight(str(snapshot["competition"]), str(trial_id), insight=insight)
    if not result.get("ok"):
        return {"ok": False, "message": result.get("message", "기록 실패")}
    _keep_latest_user_insight(str(snapshot["competition"]), str(trial_id), result.get("data", {}).get("feedback"))
    record = result.get("data", {}).get("insight") or {}
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    intent = interpretation.get("implementation_intent") if isinstance(interpretation.get("implementation_intent"), dict) else {}
    return {
        "ok": True,
        "message": "\n".join(
            [
                "인사이트:",
                insight,
                "",
                f"- 반영 예정: {next_trial}",
                f"- 적용 개선안: {intent.get('change') or '다음 계획 단계에서 구체화'}",
                f"- 상태: {record.get('status') or 'pending'}",
                f"- 개선축: {record.get('axis') or '해석 대기'}",
            ]
        ),
    }


def record_feedback_response(data: dict[str, list[str]]) -> str:
    request_id = clean((data.get("request_id") or [""])[0])
    answer = clean((data.get("answer") or [""])[0])
    decision = clean((data.get("decision") or [""])[0])
    if not request_id:
        return "피드백 요청이 없습니다."
    if not answer and not decision:
        return "선택지 또는 추가 의견을 입력해주세요."
    answers = {"decision": decision} if decision else {}
    result = respond_to_request(request_id, answers=answers, free_text=answer)
    return "답변을 기록했습니다. 다음 실험에 반영하겠습니다." if result.get("ok") else result.get("message", "기록 실패")


def ask_agent(question: str) -> str:
    return str(ask_agent_exchange(question).get("rendered_answer") or "")


def ask_agent_exchange(question: str, *, session_id: str | None = None) -> dict[str, Any]:
    question = question.strip()
    if not question:
        return {"rendered_answer": "", "history": {}}
    snapshot = app_state()["snapshot"]
    trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial")
    return answer_chat_question(
        str(snapshot["competition"]),
        trial_id,
        question,
        session_id=session_id,
    )


def analyze_new_experiment_from_form(data: dict[str, list[str]]) -> tuple[dict[str, Any] | None, str]:
    description = clean((data.get("description") or [""])[0])
    if not description:
        return None, "URL 또는 실험 설명을 입력해주세요."
    research_direction = clean((data.get("research_direction") or [""])[0])
    source_path = clean((data.get("source_path") or [""])[0]) or None
    settings = _propose_new_experiment_settings(description, research_direction, source_path)
    settings = _new_experiment_settings_with_overrides(settings, data)
    return settings, "에이전트가 새 실험 설정을 분석했습니다. 필요한 항목을 수정한 뒤 등록하세요."


def create_experiment_from_form(data: dict[str, list[str]]) -> tuple[str, str | None]:
    """Register a new experiment from the analysis form.

    Returns (message, competition) -- competition is None on failure, so
    callers (the JSON registration endpoint) can tell success from failure
    without parsing the message text.
    """
    settings, error = _settings_for_new_experiment_registration(data)
    if error:
        return error, None
    assert settings is not None
    competition = str(settings["competition"])
    result = prepare_workspace(
        competition,
        source_path=settings.get("source_path") or None,
        topic=str(settings.get("topic") or competition),
        platform=str(settings.get("platform") or "kaggle"),
        metric=str(settings.get("metric") or "unknown"),
        objective=str(settings.get("objective") or "maximize"),
        create_workspace=bool(settings.get("create_workspace")),
        target_column=str(settings.get("target_column") or "") or None,
        id_column=str(settings.get("id_column") or "") or None,
        required_data_files=list(settings.get("required_data_files") or []),
    )
    dacon_team_name = clean((data.get("dacon_team_name") or [""])[0])
    if dacon_team_name:
        set_dacon_team_name(competition, dacon_team_name)
    if str(settings.get("platform") or "").casefold() == "dacon":
        try:
            refresh_dacon_competition_docs(competition)
        except Exception:  # noqa: BLE001 - best-effort; must never block registration
            pass
    select_competition(competition)
    return f"{competition} 실험을 등록하고 선택했습니다. 상태: {result.get('status')}", competition


def delete_experiment_from_form(data: dict[str, list[str]]) -> str:
    competition = clean((data.get("competition") or [""])[0])
    confirm_text = clean((data.get("confirm_text") or [""])[0])
    if not competition:
        return "삭제할 실험을 찾을 수 없습니다."
    topic = _filesystem_topic(competition) or competition
    expected = f"{topic} 지우기"
    if confirm_text != expected:
        return f'확인 문구가 일치하지 않습니다. "{expected}"를 정확히 입력해주세요.'
    result = delete_experiment(competition)
    return f"{competition} | {topic} 실험을 삭제했습니다." if result.get("ok") else "실험 삭제에 실패했습니다."


def workspace_data_dir(competition: str) -> Path | None:
    """The data/ folder inside a registered experiment's workspace, or None
    if the experiment has no workspace/source path recorded yet (e.g. still
    at needs_project_path -- nothing to upload into until that's resolved).
    """
    record_path = competition_dir(competition) / "workspace_source.json"
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_path = record.get("source_path")
    return Path(source_path) / "data" if source_path else None


def parse_multipart_files(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Extract (filename, content) pairs from a multipart/form-data body.

    Python 3.13 removed the `cgi` module (the traditional way to parse
    this), so this is a small hand-rolled parser -- deliberately narrow: it
    only pulls out parts that carry a filename (i.e. actual file uploads),
    ignoring plain form fields, since that's all this endpoint needs.
    """
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        return []
    delimiter = b"--" + boundary_match.group(1).encode("utf-8")
    files: list[tuple[str, bytes]] = []
    for raw_part in body.split(delimiter):
        # Each real part is exactly "\r\n<headers>\r\n\r\n<content>\r\n" between
        # delimiters -- strip precisely that one leading/trailing CRLF rather
        # than bytes.strip(b"\r\n"), which would keep eating into content that
        # itself ends in \r or \n (e.g. a CSV's trailing newline).
        part = raw_part
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if not part or part == b"--":
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_text = part[:header_end].decode("utf-8", errors="replace")
        content = part[header_end + 4 :]
        filename_match = re.search(r'filename="([^"]*)"', headers_text)
        if not filename_match or not filename_match.group(1):
            continue
        files.append((filename_match.group(1), content))
    return files


def save_uploaded_files(files: list[tuple[str, bytes]], target_dir: Path) -> dict[str, Any]:
    """Write uploaded files into target_dir; .zip files are extracted in
    place instead of being saved as-is. Every path is resolved against
    target_dir and checked to still be inside it before writing, so a
    crafted filename or zip entry (path traversal / zip-slip) can't escape
    the workspace's data folder.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = target_dir.resolve()
    saved: list[str] = []
    skipped: list[dict[str, str]] = []
    for filename, content in files:
        bare_name = Path(filename).name
        if not bare_name:
            skipped.append({"name": filename, "reason": "invalid_filename"})
            continue
        if bare_name.lower().endswith(".zip"):
            extracted, zip_skipped = _extract_zip_into(content, target_dir, resolved_target)
            saved.extend(extracted)
            skipped.extend(zip_skipped)
            continue
        destination = (target_dir / bare_name).resolve()
        if resolved_target not in destination.parents and destination != resolved_target:
            skipped.append({"name": filename, "reason": "unsafe_path"})
            continue
        destination.write_bytes(content)
        saved.append(str(destination.relative_to(resolved_target)))
    return {"ok": True, "saved": saved, "skipped": skipped}


def _extract_zip_into(content: bytes, target_dir: Path, resolved_target: Path) -> tuple[list[str], list[dict[str, str]]]:
    saved: list[str] = []
    skipped: list[dict[str, str]] = []
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile:
        return saved, [{"name": "<uploaded zip>", "reason": "bad_zip"}]
    for entry in archive.infolist():
        entry_name = entry.filename.replace("\\", "/")
        if entry_name.endswith("/"):
            continue
        destination = (target_dir / entry_name).resolve()
        if resolved_target not in destination.parents:
            skipped.append({"name": entry.filename, "reason": "unsafe_path"})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(entry))
        saved.append(str(destination.relative_to(resolved_target)))
    return saved, skipped


def _settings_for_new_experiment_registration(data: dict[str, list[str]]) -> tuple[dict[str, Any] | None, str | None]:
    description = clean((data.get("description") or [""])[0])
    competition = clean((data.get("competition") or [""])[0])
    if not description and not competition:
        return None, "URL 또는 실험 설명을 입력해주세요."
    base_description = description or competition
    research_direction = clean((data.get("research_direction") or [""])[0])
    source_path = clean((data.get("source_path") or [""])[0]) or None
    settings = _propose_new_experiment_settings(base_description, research_direction, source_path)
    settings = _new_experiment_settings_with_overrides(settings, data)
    if not settings.get("competition"):
        return None, "실험 ID를 해석하지 못했습니다. 분석 결과에서 실험 ID를 직접 입력해주세요."
    return settings, None


def _new_experiment_settings_with_overrides(settings: dict[str, Any], data: dict[str, list[str]]) -> dict[str, Any]:
    updated = dict(settings)
    competition = clean((data.get("competition") or [""])[0])
    if competition:
        updated["competition"] = _normalize_competition_id(competition)
    for key in ["topic", "platform", "metric", "target_column", "id_column", "research_direction", "source_path"]:
        if key not in data:
            continue
        value = clean((data.get(key) or [""])[0])
        if key in {"target_column", "id_column", "research_direction", "source_path"}:
            updated[key] = value or None
        elif value:
            updated[key] = value
    if updated.get("source_path"):
        updated["source_path"] = _normalize_workspace_source_path(str(updated["source_path"]))
    objective = clean((data.get("objective") or [""])[0])
    if objective:
        updated["objective"] = objective if objective in {"maximize", "minimize"} else str(updated.get("objective") or "maximize")
    required_files = [
        item.strip()
        for item in clean((data.get("required_data_files") or [""])[0]).split(",")
        if item.strip()
    ]
    if "required_data_files" in data:
        updated["required_data_files"] = required_files
    create_workspace = clean((data.get("create_workspace") or [""])[0]).lower()
    if create_workspace:
        updated["create_workspace"] = create_workspace in {"1", "true", "yes", "y", "예"}
    elif updated.get("source_path"):
        updated["create_workspace"] = False
    else:
        updated["create_workspace"] = True
    return updated


def current_user_insight(snapshot: dict[str, Any]) -> str | None:
    competition = str(snapshot.get("competition") or "")
    trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial") or "trial_001"
    return _latest_user_insight(competition, str(trial_id)) if competition else None


def render_home(
    payload: dict[str, Any],
    *,
    message: str = "",
    answer: str = "",
    new_experiment_settings: dict[str, Any] | None = None,
) -> str:
    snapshot = payload.get("snapshot") or {}
    experiments = payload.get("experiments") or []
    pending = payload.get("pending_requests") or []
    existing_insight = payload.get("existing_insight")
    competition = str(snapshot.get("competition") or selected_competition())
    persistent_insight = (
        ""
        if message.startswith("인사이트:\n")
        else persistent_insight_notice(existing_insight, snapshot)
    )
    return page(
        title="Research Agent",
        body=f"""
    <main class="dashboard-shell">
      <header class="app-header">
        <div>
          <p class="eyebrow">Research Agent</p>
          <h1>현재 실험 대시보드</h1>
          <p class="subtitle">{escape(snapshot.get("competition") or "-")} | {escape(snapshot.get("topic") or "-")}</p>
          <form method="post" action="/action" class="header-experiment-switch">
            {hidden("action", "select")}
            <label for="dashboard-experiment">실험 바꾸기</label>
            <select id="dashboard-experiment" name="competition">{experiment_options(experiments, snapshot.get("competition"))}</select>
            <button>선택</button>
          </form>
        </div>
        <nav class="dashboard-actions" aria-label="대시보드 작업">
          <a href="/?sync=1">새로고침</a>
          <button type="button" class="secondary" data-open-modal="control-modal">실험 제어</button>
          <button type="button" data-open-modal="new-experiment-modal">새 실험 등록</button>
          <button type="button" data-open-modal="insight-modal">다음 실험 인사이트</button>
          <button type="button" data-open-modal="feedback-modal" {"disabled" if not pending else ""}>피드백 요청 ({len(pending)})</button>
          <button type="button" class="danger" data-open-modal="delete-modal">실험 지우기</button>
        </nav>
      </header>

      {notice(message)}
      {persistent_insight}

      <section class="summary-grid" aria-label="현재 실험 요약">
        {metric_card("상태", snapshot.get("state") or "-")}
        {metric_card("현재 / 다음 trial", f"{escape(snapshot.get('current_trial') or '-')} / {escape(snapshot.get('next_trial') or '-')}")}
        {metric_card("최근 완료", latest_metric(snapshot))}
        {metric_card("베스트", best_metric(snapshot))}
        {metric_card("피드백 요청", len(pending))}
        {metric_card(
            "제출 한도",
            '<span id="dacon-submission-limit-value">-</span> '
            '<button type="button" class="secondary" data-open-modal="dacon-limit-modal" id="dacon-submission-limit-edit-toggle" hidden>설정</button>'
            '<div id="dacon-submission-limit-reset-hint" class="hint"></div>',
        )}
      </section>

      {progress_panel(snapshot)}

      <section class="dashboard-content">
        <article class="panel">
          <div class="panel-head">
            <div>
              <h2>Trial 목록</h2>
              <p>로컬·제출 점수와 개선축을 비교하고 각 trial 산출물을 확인합니다.</p>
            </div>
          </div>
          {trial_table(competition, snapshot=snapshot)}
        </article>

        <article class="panel">
          <details>
            <summary>폴더 / DB 위치</summary>
            {locations_panel(competition, snapshot)}
          </details>
        </article>
      </section>

      <div class="modal-backdrop" id="control-modal" hidden>
        <section class="modal" role="dialog" aria-modal="true" aria-labelledby="control-modal-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow">Experiment Control</p>
              <h2 id="control-modal-title">실험 제어</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <form method="post" action="/action" class="stack" id="start-experiment-form">
            {hidden("action", "start")}
            <label>자동 실험 시작</label>
            <div class="trial-count-control">
              <input name="trial_count" type="number" min="1" value="1" aria-label="실험 횟수" data-trial-count>
            </div>
            <label class="check"><input type="checkbox" name="continuous" value="1" data-continuous-toggle> 중단 요청 전까지 계속 진행</label>
          </form>
          <label class="check" id="dacon-auto-submit-row" hidden>
            <input type="checkbox" id="dacon-auto-submit-checkbox"> 자동 제출 허용 (제출 한도가 있는 대회)
          </label>
          <div class="control-actions divided">
            <form method="post" action="/action">
              {hidden("action", "stop")}
              <button class="danger">현재 실험 중단 요청</button>
            </form>
            <button type="submit" form="start-experiment-form">시작</button>
          </div>
        </section>
      </div>

      <div class="modal-backdrop" id="new-experiment-modal" {"" if new_experiment_settings else "hidden"}>
        <section class="modal modal-large" role="dialog" aria-modal="true" aria-labelledby="new-experiment-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow">New Experiment</p>
              <h2 id="new-experiment-title">새 실험 등록</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <p class="registration-note">등록은 작업공간과 데이터 요구사항을 준비하고 실험을 선택하는 단계입니다. trial_001은 자동으로 실행되지 않습니다.</p>
          {new_experiment_panel(new_experiment_settings)}
        </section>
      </div>

      <div class="modal-backdrop" id="feedback-modal" hidden>
        <section class="modal modal-large" role="dialog" aria-modal="true" aria-labelledby="feedback-modal-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow">Human Review</p>
              <h2 id="feedback-modal-title">피드백 요청</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <p class="interaction-boundary">에이전트가 자율적으로 확정할 수 없는 연구 판단만 요청합니다. 선택 결과는 현재 실행을 다시 돌리지 않고 다음 계획 단계에 반영됩니다.</p>
          {pending_list(pending)}
        </section>
      </div>

      {delete_experiment_modal(snapshot)}

      <div class="modal-backdrop" id="insight-modal" hidden>
        <section class="modal modal-large insight-modal" role="dialog" aria-modal="true" aria-labelledby="insight-modal-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow">Research Insight</p>
              <h2 id="insight-modal-title">다음 실험 인사이트</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <p class="insight-lead">현재 trial은 그대로 두고 다음 계획 단계에 반영합니다.</p>
          <p class="insight-description">코드 작성 전 계획 회차가 있으면 해당 계획을 수정하고, 실행 중이면 다음 trial 계획에 반영합니다.</p>
          <p class="interaction-boundary">명시적으로 저장한 인사이트만 다음 계획 단계의 입력으로 기록됩니다.</p>
          {insight_hint(existing_insight, snapshot) if existing_insight else ""}
          <form method="post" action="/action" class="stack insight-form">
            {hidden("action", "insight")}
            <textarea name="insight" placeholder="다음 trial에 반영할 연구 인사이트를 입력하세요. 기존 내용이 있으면 새 내용으로 교체됩니다."></textarea>
            <button>인사이트 남기기</button>
          </form>
        </section>
      </div>

      <div class="modal-backdrop" id="artifact-modal" hidden>
        <section class="modal modal-large" role="dialog" aria-modal="true" aria-labelledby="artifact-modal-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow" id="artifact-modal-path"></p>
              <h2 id="artifact-modal-title">산출물</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <pre class="document-view" id="artifact-modal-body"></pre>
        </section>
      </div>
      {floating_chat(answer, competition=str(snapshot.get("competition") or ""))}
    </main>
""",
    )


def new_experiment_panel(settings: dict[str, Any] | None = None) -> str:
    if not settings:
        return """
            <form method="post" action="/action" class="stack nested-form">
              <p class="hint">대회 URL을 붙여넣거나 연구 목표를 자연어로 설명하면 에이전트가 실험 ID, 컬럼, 평가 지표, 데이터 파일을 먼저 추론합니다.</p>
              <label>URL 또는 실험을 설명해주세요.</label>
              <textarea name="description" placeholder="예: https://www.kaggle.com/competitions/titanic 또는 고객 이탈 예측 실험을 하고 싶어"></textarea>
              <input name="research_direction" placeholder="추가 연구 방향이나 선호가 있으면 입력하세요. 없으면 비워두세요.">
              <input name="source_path" placeholder="기존 로컬 프로젝트 경로가 있으면 입력하세요. 없으면 비워두세요.">
              <button name="action" value="new_experiment_analyze">분석하기</button>
            </form>
        """
    files = ", ".join(settings.get("required_data_files") or [])
    create_workspace = "1" if settings.get("create_workspace") else "0"
    objective = str(settings.get("objective") or "maximize")
    objective_options = "".join(
        f'<option value="{item}" {"selected" if objective == item else ""}>{item}</option>'
        for item in ["maximize", "minimize"]
    )
    workspace_options = "".join(
        f'<option value="{value}" {"selected" if create_workspace == value else ""}>{label}</option>'
        for value, label in [("1", "새 워크스페이스 생성"), ("0", "기존 경로 사용")]
    )
    platform = str(settings.get("platform") or "kaggle")
    platform_options = "".join(
        f'<option value="{value}" {"selected" if platform == value else ""} {"disabled" if disabled else ""}>{label}</option>'
        for value, label, disabled in [
            ("local_research", "로컬연구 (준비 중)", True),
            ("kaggle", "캐글", False),
            ("dacon", "데이콘", False),
        ]
    )
    return f"""
            <form method="post" action="/action" class="stack nested-form" id="new-experiment-register-form">
              <p class="hint">에이전트 분석 결과를 확인하고 필요한 항목만 수정하세요. 등록 후 이 실험이 현재 선택된 실험으로 바뀝니다.</p>
              <label>URL 또는 실험을 설명해주세요.</label>
              <textarea name="description">{escape(settings.get("description") or "")}</textarea>
              <input name="research_direction" value="{escape(settings.get("research_direction") or "")}" placeholder="추가 연구 방향">
              <input name="source_path" value="{escape(settings.get("source_path") or "")}" placeholder="기존 로컬 프로젝트 경로">

              <div class="analysis-box">
                <strong>에이전트 분석 결과</strong>
                <ul>
                  <li>실험 ID: {escape(settings.get("competition") or "-")}</li>
                  <li>주제명: {escape(settings.get("topic") or "-")}</li>
                  <li>target/id: {escape(settings.get("target_column") or "-")} / {escape(settings.get("id_column") or "-")}</li>
                  <li>평가 지표: {escape(settings.get("metric") or "-")} ({escape(settings.get("objective") or "-")})</li>
                  <li>필수 데이터 파일: {escape(files or "-")}</li>
                </ul>
              </div>

              <div class="two">
                <div class="field">
                  <label for="new-experiment-competition">실험 ID</label>
                  <input id="new-experiment-competition" name="competition" value="{escape(settings.get("competition") or "")}" placeholder="실험 ID">
                </div>
                <div class="field">
                  <label for="new-experiment-topic">주제명</label>
                  <input id="new-experiment-topic" name="topic" value="{escape(settings.get("topic") or "")}" placeholder="주제명">
                </div>
              </div>
              <div class="two">
                <div class="field">
                  <label for="new-experiment-platform">플랫폼</label>
                  <select id="new-experiment-platform" name="platform">{platform_options}</select>
                </div>
                <div class="field">
                  <label for="new-experiment-workspace">워크스페이스</label>
                  <select id="new-experiment-workspace" name="create_workspace">{workspace_options}</select>
                </div>
              </div>
              <div class="two">
                <div class="field">
                  <label for="new-experiment-dacon-team">데콘 팀명 (선택)</label>
                  <input id="new-experiment-dacon-team" name="dacon_team_name" value="{escape(settings.get("dacon_team_name") or "")}" placeholder="비워두면 로컬로만 진행 (제출 없음)">
                </div>
              </div>
              <div class="two">
                <div class="field">
                  <label for="new-experiment-metric">평가지표</label>
                  <input id="new-experiment-metric" name="metric" value="{escape(settings.get("metric") or "")}" placeholder="평가지표">
                </div>
                <div class="field">
                  <label for="new-experiment-objective">목표 방향</label>
                  <select id="new-experiment-objective" name="objective">{objective_options}</select>
                </div>
              </div>
              <div class="two">
                <div class="field">
                  <label for="new-experiment-target-column">타깃 컬럼</label>
                  <input id="new-experiment-target-column" name="target_column" value="{escape(settings.get("target_column") or "")}" placeholder="타깃 컬럼">
                </div>
                <div class="field">
                  <label for="new-experiment-id-column">ID 컬럼</label>
                  <input id="new-experiment-id-column" name="id_column" value="{escape(settings.get("id_column") or "")}" placeholder="ID 컬럼">
                </div>
              </div>
              <div class="field">
                <label for="new-experiment-data-files">필수 데이터 파일</label>
                <input id="new-experiment-data-files" name="required_data_files" value="{escape(files)}" placeholder="필수 데이터 파일, 예: train.csv,test.csv">
              </div>
              <div class="field">
                <label>데이터 업로드 (선택)</label>
                <p class="hint">csv 등 개별 파일은 그대로, zip 파일은 등록 시 자동으로 압축이 풀립니다 (파일이 많은 폴더는 압축해서 올려주세요). 등록 버튼을 눌러야 실제로 저장됩니다.</p>
                <div id="upload-drop-zone" class="upload-drop-zone" tabindex="0">
                  <p>여기로 파일을 드래그하거나 클릭해서 선택하세요.</p>
                  <input type="file" id="upload-file-input" multiple hidden>
                </div>
                <ul id="upload-file-list" class="upload-file-list"></ul>
              </div>
              <div class="button-row two">
                <button class="secondary" name="action" value="new_experiment_analyze">분석 다시 하기</button>
                <button name="action" value="new_experiment">등록</button>
              </div>
              <p id="register-status" class="hint" role="status"></p>
            </form>
        """


def delete_experiment_modal(snapshot: dict[str, Any]) -> str:
    competition = str(snapshot.get("competition") or "")
    topic = str(snapshot.get("topic") or competition or "-")
    expected_text = f"{topic} 지우기"
    return f"""
      <div class="modal-backdrop" id="delete-modal" hidden>
        <section class="modal" role="dialog" aria-modal="true" aria-labelledby="delete-modal-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow">Delete Experiment</p>
              <h2 id="delete-modal-title">실험 지우기</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <div id="delete-step-confirm">
            <p><strong>{escape(topic)}</strong> 실험을 지우시겠습니까?</p>
            <p class="hint">실험에 관련된 모든 파일과 기록(trial, 산출물, 채팅 이력 포함)이 영구적으로 삭제되며 되돌릴 수 없습니다.</p>
            <div class="button-row two">
              <button type="button" class="secondary" data-close-modal>No</button>
              <button type="button" class="danger" id="delete-step-yes">Yes</button>
            </div>
          </div>
          <div id="delete-step-type" hidden>
            <p>정말로 삭제하려면 아래에 <strong>{escape(expected_text)}</strong>를 정확히 입력하세요.</p>
            <form method="post" action="/action" class="stack" id="delete-experiment-form">
              {hidden("action", "delete_experiment")}
              {hidden("competition", competition)}
              <input name="confirm_text" id="delete-confirm-input" placeholder="{escape(expected_text)}" autocomplete="off">
              <div class="button-row two">
                <button type="button" class="secondary" id="delete-step-back">취소</button>
                <button type="submit" class="danger" id="delete-confirm-submit" disabled data-expected="{escape(expected_text)}">지우기</button>
              </div>
            </form>
          </div>
        </section>
      </div>

      <div class="modal-backdrop" id="dacon-limit-modal" hidden>
        <section class="modal" role="dialog" aria-modal="true" aria-labelledby="dacon-limit-modal-title">
          <div class="modal-head">
            <div>
              <p class="eyebrow">DACON</p>
              <h2 id="dacon-limit-modal-title">제출 한도 설정</h2>
            </div>
            <button type="button" class="icon-button" data-close-modal aria-label="닫기">×</button>
          </div>
          <p class="hint">하루 제출 가능 횟수를 직접 입력합니다. 자동으로 확인된 값보다 우선 적용됩니다.</p>
          <div class="stack">
            <input type="number" min="1" id="dacon-submission-limit-input" placeholder="예: 5">
            <div class="button-row two">
              <button type="button" class="secondary" data-close-modal>취소</button>
              <button type="button" id="dacon-submission-limit-save">저장</button>
            </div>
          </div>
        </section>
      </div>
    """


def floating_chat(answer: str = "", *, competition: str | None = None) -> str:
    history = _chat_history_for_render(competition)
    sessions = list(history.get("sessions") or [])
    active = history.get("active_session") or {}
    messages = list(history.get("messages") or [])
    initial_messages = _chat_messages_html(messages)
    if not initial_messages:
        initial_messages = (
            '<div class="chat-message agent">궁금한 점을 물어보세요. 현재 선택된 실험의 '
            "사용자용 산출물과 내부 기록을 기준으로 답변합니다.</div>"
            + answer_block(answer)
        )
    session_options = _chat_session_options(sessions, str(active.get("session_id") or ""))
    chat_status = _chat_status_label()
    return f"""
      <button type="button" class="chat-fab" id="chat-fab" aria-label="에이전트 채팅 열기" aria-controls="chat-widget" aria-expanded="false">AI</button>
      <section class="chat-widget" id="chat-widget" aria-label="에이전트 채팅" hidden>
        <button type="button" class="chat-resize-handle" id="chat-resize-handle" aria-label="채팅창 크기 조절" title="드래그하여 채팅창 크기 조절"></button>
        <header class="chat-widget-head">
          <div>
            <strong>Research Agent</strong>
            <span>{escape(chat_status)}</span>
          </div>
          <button type="button" class="chat-close" id="chat-close" aria-label="채팅 닫기">x</button>
        </header>
        <p class="chat-boundary">읽기 전용 · 대화는 실험 계획, 코드, 점수, 연구 판단을 변경하지 않습니다.</p>
        <div class="chat-history-toolbar">
          <select id="chat-session-select" aria-label="이전 대화 선택" {"disabled" if not sessions else ""}>
            {session_options or '<option value="">저장된 대화 없음</option>'}
          </select>
          <button type="button" class="chat-new-session" id="chat-new-session" aria-label="새 대화" title="새 대화">+</button>
        </div>
        <div class="chat-log" id="chat-log" aria-live="polite">
          {initial_messages}
        </div>
        <form method="post" action="/api/question" class="chat-form" id="question-form">
          <textarea name="question" rows="2" placeholder="메시지를 입력하세요. Enter 전송, Shift+Enter 줄바꿈"></textarea>
          <button type="submit">질문하기</button>
        </form>
      </section>
"""


def _chat_history_for_render(competition: str | None) -> dict[str, Any]:
    if not competition:
        return {"sessions": [], "active_session": None, "messages": []}
    try:
        return chat_history_snapshot(competition)
    except Exception:
        return {"sessions": [], "active_session": None, "messages": []}


def _chat_session_options(sessions: list[dict[str, Any]], active_session_id: str) -> str:
    options = []
    for session in sessions:
        session_id = str(session.get("session_id") or "")
        title = str(session.get("title") or "새 대화")
        selected = " selected" if session_id == active_session_id else ""
        options.append(
            f'<option value="{escape(session_id)}"{selected}>{escape(title)}</option>'
        )
    return "".join(options)


def _chat_messages_html(messages: list[dict[str, Any]]) -> str:
    blocks = []
    for message in messages:
        role = str(message.get("role") or "system")
        if role not in {"user", "assistant", "system"}:
            role = "system"
        css_role = "agent" if role == "assistant" else role
        trial_id = str(message.get("trial_id") or "")
        meta = f'<small class="chat-message-meta">{escape(trial_id)}</small>' if trial_id else ""
        blocks.append(
            f'<div class="chat-message {css_role}">'
            f'<span>{escape(message.get("content") or "")}</span>{meta}</div>'
        )
    return "".join(blocks)


def _chat_status_label() -> str:
    if str(os.environ.get("RESEARCH_AGENT_CHAT_DEMO_MODE") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return "DEMO · API 없이 로컬 근거로 답변"
    if os.environ.get("OPENAI_API_KEY"):
        return "현재 실험 기준 · 저비용 LLM"
    return "현재 실험 기준 · 로컬 근거 모드"


def latest_metric(snapshot: dict[str, Any]) -> str:
    latest = snapshot.get("latest") or {}
    return (
        f"{escape(snapshot.get('last_completed_trial') or latest.get('trial_id') or '-')}<br>"
        f"<span>로컬 {escape(_compact_score(latest.get('local_score')))} / 제출 {escape(_compact_score(latest.get('lb_score')))}</span>"
    )


def best_metric(snapshot: dict[str, Any]) -> str:
    best = snapshot.get("best") or {}
    return (
        f"{escape(best.get('trial_id') or '-')}<br>"
        f"<span>로컬 {escape(_compact_score(best.get('local_score')))} / 제출 {escape(_compact_score(best.get('lb_score')))}</span>"
    )


def progress_panel(snapshot: dict[str, Any]) -> str:
    if not (snapshot.get("loop") or {}):
        progress = ""
    else:
        text = render_status_text(snapshot)
        progress = text.split("\n\n", 1)[1] if "\n\n" in text else ""
    if not progress:
        progress = "현재 표시할 진행 로그가 없습니다."
    return f"""
      <section class="progress-panel">
        <h2>진행 로그</h2>
        <pre>{escape(progress)}</pre>
      </section>
"""


def trial_table(competition: str, *, snapshot: dict[str, Any] | None = None) -> str:
    try:
        rows = _sqlite_trial_rows(competition)
    except Exception as exc:
        return f'<p class="empty">Trial 목록을 불러오지 못했습니다: {escape(exc)}</p>'
    if not rows:
        return '<p class="empty">아직 SQLite DB에 trial 기록이 없습니다.</p>'
    rows = sorted(rows, key=_trial_sort_value, reverse=True)
    insight_trials = user_insight_target_trial_ids(competition)
    body = []
    for row in rows:
        trial_id = str(row.get("trial_id") or "-")
        status = str(row.get("status") or "")
        display_status = _effective_trial_status(trial_id, status, snapshot or {})
        is_planned = status.casefold() in {"planned", "ready"}
        detail = render_sqlite_trial_detail(competition, trial_id)
        axis = str(row.get("change_axis") or "-")
        if trial_id in insight_trials and axis != "-":
            axis = f"insight: {axis}"
        if is_planned:
            axis = "-"
        plan = str(row.get("improvement_plan") or "-")
        source_trial_id = str(row.get("source_trial_id") or "-")
        submit_cell = (
            f'<button type="button" class="secondary" data-submit-trial'
            f' data-competition="{escape(competition)}" data-trial-id="{escape(trial_id)}">제출</button>'
            if not is_planned and row.get("lb_score") is None
            else "-"
        )
        body.append(
            '<tr data-trial-row>'
            f"<td>{escape(trial_id)}</td>"
            f"<td>{trial_status_badge(display_status)}</td>"
            f"<td>{escape(source_trial_id)}</td>"
            f"<td>{escape(_compact_score(row.get('local_score')))}</td>"
            f"<td>{escape(_compact_score(row.get('lb_score')))}</td>"
            f"<td>{_tooltip_cell(axis, 28)}</td>"
            f"<td>{_tooltip_cell(plan, 34)}</td>"
            f"<td>{best_badge(_best_label(row))}</td>"
            f"<td>{submit_cell}</td>"
            f"<td>{trial_artifact_links(competition, trial_id, planned=is_planned)}</td>"
            f"<td><details><summary>보기</summary><pre>{escape(detail)}</pre></details></td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="trial-table" data-page-size="5">'
        "<thead><tr><th>trial</th><th>상태</th><th>기준</th><th>local</th><th>submit</th><th>axis</th><th>plan</th><th>best</th><th>제출</th><th>산출물</th><th>상세</th></tr></thead>"
        f'<tbody>{("").join(body)}</tbody></table></div>'
        '<nav class="pagination" aria-label="Trial 목록 페이지" data-trial-pagination></nav>'
    )


def _tooltip_cell(value: str, limit: int) -> str:
    compact = _compact_cell(value, limit)
    if compact == value:
        return escape(value)
    return (
        f'<span class="cell-tooltip" title="{escape(value)}" '
        f'aria-label="{escape(value)}">{escape(compact)}</span>'
    )


def _trial_sort_value(row: dict[str, Any]) -> tuple[int, str]:
    trial_id = str(row.get("trial_id") or "")
    suffix = trial_id.rpartition("_")[2]
    return (int(suffix) if suffix.isdigit() else -1, trial_id)


def best_badge(label: str) -> str:
    if label == "-":
        return '<span class="badge muted">-</span>'
    return f'<span class="badge">{escape(label)}</span>'


def trial_status_badge(status: str) -> str:
    value = status.casefold()
    label = {
        "discovered": "실행 전",
        "planned": "계획 완료",
        "ready": "계획 완료",
        "completed": "완료",
        "blocked": "중단",
        "recovery_pending": "로컬 완료 · 후처리 대기",
        "running": "진행 중",
    }.get(value, status or "-")
    css = "badge planned" if value in {"discovered", "planned", "ready", "recovery_pending", "running"} else "badge muted"
    return f'<span class="{css}">{escape(label)}</span>'


def _effective_trial_status(trial_id: str, status: str, snapshot: dict[str, Any]) -> str:
    loop = snapshot.get("loop") or {}
    loop_trial = str(loop.get("current_trial") or loop.get("next_trial") or "")
    if loop_trial != trial_id:
        return status
    loop_status = str(loop.get("status") or "").casefold()
    if loop_status in {"starting", "running", "resuming"}:
        return "running"
    if loop_status == "failed" and loop.get("error") == "recoverable_after_metrics_collection":
        return "recovery_pending"
    return status


def trial_artifact_links(competition: str, trial_id: str, *, planned: bool = False) -> str:
    if planned:
        out_dir = project_root() / "experiments" / competition / trial_id
        plan_path = next(
            (
                path
                for path in [
                    out_dir / "user_view" / "01_plan.ko.md",
                    out_dir / "demo_experiment_plan.md",
                    out_dir / "next_experiment.md",
                ]
                if path.is_file()
            ),
            None,
        )
        plan_chip = (
            f'<button type="button" class="artifact-chip" data-open-artifact="{escape(project_relative_path(plan_path))}">계획</button>'
            if plan_path
            else '<span class="artifact-chip disabled">계획</span>'
        )
        return (
            f'{plan_chip} '
            '<span class="artifact-chip disabled">구조</span> '
            '<span class="artifact-chip disabled">점수</span>'
        )
    try:
        with state_db_connection(default_db_path()) as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT artifact_type, path
                    FROM trial_artifacts
                    WHERE competition_id = ?
                      AND trial_id = ?
                      AND is_user_facing = 1
                    ORDER BY artifact_type, path
                    """,
                    [competition, trial_id],
                )
            ]
    except Exception:
        rows = []
    if not rows:
        return '<span class="empty">-</span>'
    labels = {
        "plan_ko": "계획",
        "pipeline_structure_ko": "구조",
        "scores_ko": "점수",
    }
    links = [
        f'<button type="button" class="artifact-chip" data-open-artifact="{escape(project_relative_path(Path(str(row.get("path") or ""))))}">'
        f'{escape(labels.get(str(row.get("artifact_type") or ""), str(row.get("artifact_type") or "문서")))}</button>'
        for row in rows[:4]
        if row.get("path") and _artifact_path_exists(Path(str(row["path"])))
    ]
    return " ".join(links) if links else '<span class="empty">-</span>'


def _artifact_path_exists(path: Path) -> bool:
    candidate = path if path.is_absolute() else project_root() / path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(project_root().resolve())
    except (OSError, ValueError):
        return False
    return resolved.is_file()


def artifact_panel(competition: str, snapshot: dict[str, Any]) -> str:
    locations = safe_artifact_locations(competition, snapshot)
    user_dir = locations.get("user_view")
    if not user_dir:
        return '<p class="empty">사용자용 산출물 위치를 확인하지 못했습니다.</p>'
    files = user_artifact_files(user_dir)
    if not files:
        return f'<p class="empty">아직 사용자용 산출물이 없습니다.</p><p class="path-text">{escape(user_dir)}</p>'
    items = [
        f'<li><button type="button" class="artifact-link" data-open-artifact="{escape(project_relative_path(path))}">{escape(artifact_title(path))}</button><span>{escape(path.name)}</span></li>'
        for path in files
    ]
    return f'<ul class="artifact-list">{"".join(items)}</ul><p class="path-text">{escape(user_dir)}</p>'


def safe_artifact_locations(competition: str, snapshot: dict[str, Any]) -> dict[str, Path]:
    try:
        return _artifact_locations(competition, snapshot)
    except Exception:
        return {}


def user_artifact_files(user_dir: Path) -> list[Path]:
    if not user_dir.exists() or not user_dir.is_dir():
        return []
    preferred = [
        "01_plan.ko.md",
        "02_pipeline_structure.ko.md",
        "03_scores.ko.md",
        "README.ko.md",
    ]
    found = {path.name: path for path in user_dir.iterdir() if path.is_file()}
    ordered = [found[name] for name in preferred if name in found]
    extras = sorted(
        [
            path
            for name, path in found.items()
            if name not in preferred and path.suffix.lower() in {".md", ".txt", ".csv"}
        ],
        key=lambda item: item.name,
    )
    return ordered + extras


def artifact_title(path: Path) -> str:
    labels = {
        "01_plan.ko.md": "실험 계획서",
        "02_pipeline_structure.ko.md": "파이프라인 구조도",
        "03_scores.ko.md": "로컬 / 제출 점수",
        "README.ko.md": "사용자 안내",
    }
    return labels.get(path.name, path.stem)


def artifact_href(path: Path) -> str:
    return "/artifact?" + urlencode({"path": project_relative_path(path)})


def project_relative_path(path: Path) -> str:
    root = project_root().resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def locations_panel(competition: str, snapshot: dict[str, Any]) -> str:
    labels = {
        "user_view": "사용자용 산출물 폴더",
        "workspace": "작업 코드 폴더",
        "experiment_root": "실험 전체 기록 폴더",
        "internal": "현재 trial 내부 기록 폴더",
        "submission": "제출 파일 폴더",
    }
    locations = safe_artifact_locations(competition, snapshot)
    rows = [
        f"<li><strong>{escape(labels[key])}</strong><br><code>{escape(path)}</code></li>"
        for key, path in locations.items()
        if key in labels
    ]
    rows.append(f"<li><strong>SQLite DB</strong><br><code>{escape(default_db_path())}</code></li>")
    return f'<ul class="location-list">{"".join(rows)}</ul>'


def load_artifact_content(relative_path: str) -> dict[str, Any]:
    path = resolve_project_file(relative_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise FileNotFoundError(str(exc)) from exc
    return {
        "title": artifact_title(path),
        "path": project_relative_path(path),
        "content": content,
    }


def render_artifact_page(relative_path: str) -> str:
    path = resolve_project_file(relative_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise FileNotFoundError(str(exc)) from exc
    return page(
        title=path.name,
        body=f"""
    <main class="dashboard-shell narrow">
      <header class="app-header">
        <div>
          <p class="eyebrow">사용자용 산출물</p>
          <h1>{escape(artifact_title(path))}</h1>
          <p class="subtitle">{escape(project_relative_path(path))}</p>
        </div>
        <nav><a href="/">대시보드로 돌아가기</a></nav>
      </header>
      <article class="panel wide">
        <pre class="document-view">{escape(content)}</pre>
      </article>
    </main>
""",
    )


def resolve_project_file(relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("empty path")
    root = project_root().resolve()
    requested = Path(relative_path.replace("\\", "/"))
    candidate = requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes project root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    return candidate


def page(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #16202a;
      --muted: #617084;
      --line: #d9e0e7;
      --soft: #eef2f5;
      --accent: #0f766e;
      --accent-2: #40566f;
      --danger: #b13f3f;
      --terminal-bg: #151a1f;
      --terminal-fg: #edf2f7;
    }}
    * {{ box-sizing: border-box; }}
    [hidden] {{ display: none !important; }}
    body {{ margin: 0; font-family: Arial, "Malgun Gothic", sans-serif; background: var(--bg); color: var(--ink); }}
    a {{ color: var(--accent); text-decoration: none; }}
    .dashboard-shell {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
    .dashboard-shell.narrow {{ max-width: 920px; }}
    .app-header {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 18px; }}
    .app-header > div {{ min-width: 0; }}
    .eyebrow {{ margin: 0 0 4px; color: var(--accent); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }}
    h1 {{ margin: 0; font-size: 28px; line-height: 1.2; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .subtitle, .panel-head p {{ margin: 6px 0 0; color: var(--muted); line-height: 1.45; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    nav a, .mini-link {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; color: var(--accent-2); background: var(--panel); font-size: 13px; }}
    .dashboard-actions {{ justify-content: flex-end; }}
    .dashboard-actions button {{ padding: 8px 11px; }}
    .header-experiment-switch {{ display: grid; grid-template-columns: auto minmax(240px, 390px) auto; gap: 8px; align-items: center; margin-top: 12px; }}
    .header-experiment-switch label {{ font-size: 13px; }}
    .header-experiment-switch select {{ min-width: 0; }}
    .header-experiment-switch button {{ padding: 9px 12px; }}
    .notice {{ margin-bottom: 14px; border: 1px solid #b7dccf; background: #e9f7f2; color: #0d594e; border-radius: 8px; padding: 12px 14px; line-height: 1.65; }}
    .notice-title {{ display: block; margin-bottom: 6px; }}
    .notice-body {{ white-space: pre-wrap; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }}
    .metric {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 14px; min-height: 88px; }}
    .metric .label {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .metric .value {{ font-size: 18px; font-weight: 700; line-height: 1.35; overflow-wrap: anywhere; }}
    .metric .value span {{ display: inline-block; margin-top: 3px; color: var(--muted); font-size: 13px; font-weight: 500; }}
    .progress-panel, .panel {{ min-width: 0; border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 16px; }}
    .progress-panel {{ margin-bottom: 14px; }}
    .progress-panel pre {{ margin: 0; border-radius: 6px; background: var(--soft); padding: 12px; color: var(--ink); white-space: pre-wrap; line-height: 1.55; }}
    .dashboard-content {{ min-width: 0; display: grid; gap: 14px; }}
    .panel.wide {{ grid-column: 1 / -1; }}
    .panel-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
    .table-wrap {{ width: 100%; max-width: 100%; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ color: var(--muted); font-size: 13px; background: #fafbfc; }}
    td details pre {{ margin: 8px 0 0; min-width: 260px; white-space: pre-wrap; }}
    .cell-tooltip {{ cursor: help; text-decoration: underline dotted #9aa7b3; text-underline-offset: 3px; }}
    .badge {{ display: inline-block; border-radius: 999px; background: #dff2eb; color: #0d594e; padding: 3px 8px; font-size: 12px; font-weight: 700; }}
    .badge.planned {{ background: #e8eef5; color: #3f5871; }}
    .badge.muted {{ background: var(--soft); color: var(--muted); }}
    .artifact-chip {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; margin: 0 4px 4px 0; font: inherit; font-size: 12px; background: #fff; color: var(--ink); cursor: pointer; }}
    .artifact-chip.disabled {{ background: var(--soft); color: #9aa4ae; cursor: not-allowed; opacity: .72; }}
    .artifact-list, .location-list {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }}
    .artifact-list li {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; display: grid; gap: 3px; }}
    .artifact-list span, .path-text, .empty, .hint {{ color: var(--muted); }}
    .artifact-link {{ border: none; background: none; padding: 0; margin: 0; font: inherit; color: var(--accent-2); text-decoration: underline; cursor: pointer; text-align: left; }}
    .path-text {{ margin: 12px 0 0; font-size: 12px; overflow-wrap: anywhere; }}
    .location-list li {{ padding: 8px 0; border-bottom: 1px solid var(--line); }}
    .analysis-box {{ border: 1px solid var(--line); border-radius: 8px; background: #fafbfc; padding: 12px; }}
    .analysis-box ul {{ margin-top: 8px; }}
    code {{ color: var(--accent-2); overflow-wrap: anywhere; }}
    label, summary {{ color: var(--ink); font-weight: 700; }}
    summary {{ cursor: pointer; }}
    .stack {{ display: grid; gap: 9px; }}
    .divided {{ margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }}
    .button-row {{ margin-top: 10px; }}
    .inline-control {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; }}
    .inline-control.compact {{ grid-template-columns: 110px auto; justify-content: start; }}
    .trial-count-control {{ width: 110px; }}
    .control-actions {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .control-actions form {{ margin: 0; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .field {{ display: grid; gap: 4px; }}
    .field label {{ font-size: 12px; font-weight: 600; color: var(--muted); }}
    .upload-drop-zone {{ display: grid; place-items: center; text-align: center; border: 2px dashed var(--line); border-radius: 8px; padding: 36px 16px; margin: 12px 0; cursor: pointer; color: var(--muted); transition: border-color .15s, background .15s; }}
    .upload-drop-zone.dragover {{ border-color: var(--accent); background: var(--soft); color: var(--ink); }}
    .upload-file-list {{ list-style: none; margin: 0 0 8px; padding: 0; display: grid; gap: 6px; }}
    .upload-file-list li {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; font-size: 13px; }}
    .upload-file-list button {{ width: auto; padding: 2px 8px; background: transparent; color: var(--muted); border: none; cursor: pointer; }}
    input, select, textarea {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 10px; font: 14px Arial, "Malgun Gothic", sans-serif; color: var(--ink); background: #fff; }}
    input:disabled {{ border-color: #d7dce2; background: #e8ebef; color: #8a949f; cursor: not-allowed; }}
    textarea {{ min-height: 92px; resize: vertical; }}
    button {{ border: 0; border-radius: 6px; background: var(--accent); color: #fff; padding: 10px 13px; font-weight: 700; cursor: pointer; white-space: nowrap; }}
    button.secondary {{ background: var(--accent-2); }}
    button.danger {{ background: var(--danger); }}
    .icon-button {{ width: 36px; height: 36px; padding: 0; border-radius: 50%; background: var(--soft); color: var(--ink); font-size: 22px; font-weight: 400; }}
    .check {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-weight: 500; }}
    .check input {{ width: auto; }}
    .hint {{ margin: 0 0 10px; white-space: pre-wrap; line-height: 1.55; }}
    .registration-note {{ margin: 0 0 16px; padding: 11px 12px; border-left: 3px solid var(--accent); background: var(--soft); color: var(--muted); line-height: 1.5; }}
    .pagination {{ justify-content: center; margin-top: 14px; }}
    .pagination button {{ min-width: 36px; height: 36px; padding: 0; border: 1px solid var(--line); background: #fff; color: var(--accent-2); }}
    .pagination button[aria-current="page"] {{ border-color: var(--accent); background: var(--accent); color: #fff; }}
    .modal-backdrop {{ position: fixed; inset: 0; z-index: 60; display: grid; place-items: center; padding: 20px; background: rgba(22,32,42,.46); }}
    .modal {{ width: min(560px, 100%); max-height: calc(100vh - 40px); overflow-y: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 20px; box-shadow: 0 24px 70px rgba(22,32,42,.3); }}
    .modal.modal-large {{ width: min(780px, 100%); }}
    .modal-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 16px; }}
    .insight-modal {{ width: min(920px, 100%); }}
    .insight-lead {{ margin: 0 0 18px; color: var(--muted); font-size: 18px; line-height: 1.5; }}
    .insight-description {{ margin: 0 0 18px; color: var(--muted); line-height: 1.65; }}
    .insight-form textarea {{ min-height: 150px; }}
    .insight-form button {{ width: 100%; }}
    ul {{ margin: 0; padding-left: 20px; line-height: 1.7; }}
    li form {{ margin-top: 8px; display: grid; gap: 8px; }}
    .feedback-request-card {{ margin-bottom: 14px; border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 7px; padding: 16px; list-style: none; }}
    .feedback-type {{ display: inline-block; margin-bottom: 9px; border-radius: 5px; padding: 4px 7px; background: #e8f5f2; color: #086b60; font-size: 12px; font-weight: 700; }}
    .feedback-problem {{ font-size: 16px; line-height: 1.55; }}
    .feedback-evidence-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin: 12px 0; }}
    .feedback-evidence {{ display: grid; gap: 4px; border-radius: 6px; padding: 10px; background: var(--soft); }}
    .feedback-evidence span, .feedback-evidence small {{ color: var(--muted); }}
    .feedback-evidence strong {{ font-size: 17px; }}
    .feedback-options {{ display: grid; gap: 8px; }}
    .feedback-option {{ display: flex; gap: 9px; align-items: flex-start; border: 1px solid var(--line); border-radius: 6px; padding: 10px; cursor: pointer; }}
    .feedback-option input {{ width: auto; margin-top: 3px; }}
    .feedback-option span {{ display: grid; gap: 3px; }}
    .feedback-option small {{ color: var(--muted); font-weight: 400; line-height: 1.45; }}
    .feedback-default {{ border-radius: 6px; padding: 10px; background: var(--soft); color: var(--muted); }}
    .feedback-execution-note {{ border-radius: 6px; padding: 10px; background: #fff4d6; color: #76520b; }}
    .chat-fab {{ position: fixed; right: 24px; bottom: 24px; z-index: 30; width: 58px; height: 58px; border-radius: 50%; box-shadow: 0 10px 24px rgba(22,32,42,.2); font-size: 16px; }}
    .chat-widget {{ position: fixed; right: 24px; bottom: 94px; z-index: 31; width: 390px; min-width: 390px; max-width: calc(100vw - 48px); height: min(620px, calc(100vh - 126px)); min-height: min(620px, calc(100vh - 126px)); max-height: calc(100vh - 110px); border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: 0 18px 46px rgba(22,32,42,.22); display: flex; flex-direction: column; overflow: hidden; }}
    .chat-resize-handle {{ position: absolute; top: 0; left: 0; z-index: 2; width: 28px; height: 28px; padding: 0; border-radius: 7px 0 7px 0; background: transparent; color: var(--muted); cursor: nwse-resize; touch-action: none; }}
    .chat-resize-handle::before, .chat-resize-handle::after {{ content: ""; position: absolute; left: 7px; top: 7px; width: 9px; height: 1px; background: currentColor; transform: rotate(-45deg); transform-origin: left center; }}
    .chat-resize-handle::after {{ left: 7px; top: 12px; width: 15px; }}
    .chat-resize-handle:hover, .chat-resize-handle:focus-visible {{ background: var(--soft); color: var(--accent); }}
    .chat-widget.is-resizing {{ user-select: none; }}
    .chat-widget-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 14px 12px 36px; border-bottom: 1px solid var(--line); }}
    .chat-widget-head div {{ display: grid; gap: 3px; }}
    .chat-widget-head span {{ color: var(--muted); font-size: 12px; }}
    .chat-boundary {{ margin: 0; padding: 9px 14px; border-bottom: 1px solid var(--line); background: #f5f8fa; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .chat-history-toolbar {{ display: grid; grid-template-columns: minmax(0, 1fr) 36px; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--line); background: #fff; }}
    .chat-history-toolbar select {{ min-width: 0; height: 36px; padding: 5px 30px 5px 9px; font-size: 13px; }}
    .chat-new-session {{ width: 36px; height: 36px; padding: 0; border-radius: 50%; background: var(--soft); color: var(--ink); font-size: 22px; line-height: 1; }}
    .interaction-boundary {{ margin: 0 0 18px; padding: 10px 12px; border-left: 3px solid var(--accent); background: var(--soft); color: var(--muted); line-height: 1.5; }}
    .chat-close {{ width: 34px; height: 34px; border-radius: 50%; padding: 0; background: var(--soft); color: var(--ink); font-size: 18px; }}
    .chat-log {{ display: grid; gap: 10px; margin-bottom: 10px; max-height: 420px; overflow-y: auto; }}
    .chat-widget .chat-log {{ flex: 1; max-height: none; margin: 0; padding: 14px; align-content: start; }}
    .chat-message {{ display: grid; gap: 6px; border-radius: 8px; padding: 10px 12px; line-height: 1.55; }}
    .chat-message > span {{ white-space: pre-wrap; }}
    .chat-message-meta {{ color: var(--muted); font-size: 11px; }}
    .chat-message.user {{ justify-self: end; max-width: 86%; background: #dff2eb; color: #0d594e; }}
    .chat-message.agent {{ justify-self: start; max-width: 100%; background: var(--soft); color: var(--ink); }}
    .chat-message.system {{ justify-self: stretch; background: #fff4d6; color: #76520b; }}
    .chat-loading {{ color: var(--muted); font-size: 14px; }}
    .chat-widget .chat-form {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; padding: 12px; border-top: 1px solid var(--line); background: #fafbfc; }}
    .chat-widget .chat-form textarea {{ min-height: 44px; max-height: 120px; resize: vertical; }}
    .chat-widget .chat-form button {{ align-self: end; }}
    .answer pre, .terminal, .document-view {{ margin: 12px 0 0; padding: 12px; border-radius: 6px; background: var(--terminal-bg); color: var(--terminal-fg); white-space: pre-wrap; line-height: 1.55; overflow-x: auto; }}
    .terminal.small {{ max-height: 360px; }}
    .admin-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }}
    .document-view {{ background: #fff; color: var(--ink); border: 1px solid var(--line); font-family: Consolas, "D2Coding", "Malgun Gothic", monospace; }}
    @media (max-width: 900px) {{
      .dashboard-shell {{ padding: 14px; }}
      .app-header {{ flex-direction: column; }}
      .header-experiment-switch {{ width: 100%; grid-template-columns: minmax(0, 1fr) auto; }}
      .header-experiment-switch label {{ grid-column: 1 / -1; }}
      .summary-grid, .admin-grid {{ grid-template-columns: 1fr; }}
      .panel.wide {{ grid-column: auto; }}
      .inline-control, .inline-control.compact, .two {{ grid-template-columns: 1fr; }}
      .control-actions {{ align-items: stretch; }}
      .chat-fab {{ right: 16px; bottom: 16px; }}
      .chat-widget {{ right: 16px; bottom: 84px; width: calc(100vw - 32px) !important; min-width: 0; max-width: none; height: min(620px, calc(100vh - 104px)) !important; min-height: 0; max-height: none; }}
      .chat-resize-handle {{ display: none; }}
      .chat-widget .chat-form {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  {body}
  <script>
    (function setupDaconSubmissionLimit() {{
      const valueEl = document.getElementById("dacon-submission-limit-value");
      if (!valueEl) return;
      const card = valueEl.closest(".metric");
      const editToggle = document.getElementById("dacon-submission-limit-edit-toggle");
      const resetHintEl = document.getElementById("dacon-submission-limit-reset-hint");
      const editInput = document.getElementById("dacon-submission-limit-input");
      const saveButton = document.getElementById("dacon-submission-limit-save");
      const limitModal = document.getElementById("dacon-limit-modal");
      const autoSubmitRow = document.getElementById("dacon-auto-submit-row");
      const autoSubmitCheckbox = document.getElementById("dacon-auto-submit-checkbox");

      function renderValue(data) {{
        const isManual = data.status === "manual_override";
        if (data.daily_submission_limit == null) {{
          valueEl.textContent = "확인 안됨";
        }} else if (data.remaining == null) {{
          // Limit known, but the remaining count couldn't be computed (e.g.
          // no/expired DACON session token) -- show the limit alone rather
          // than a broken "?/5".
          valueEl.innerHTML = `<strong>${{data.daily_submission_limit}}</strong>회${{isManual ? " (직접입력)" : ""}}`;
        }} else {{
          valueEl.innerHTML = `<strong>${{data.remaining}}</strong> / ${{data.daily_submission_limit}}${{isManual ? " (직접입력)" : ""}}`;
        }}
        valueEl.title = data.message || "";
        if (resetHintEl) {{
          resetHintEl.textContent = data.next_reset_estimate
            ? `예상 다음 초기화: ${{data.next_reset_estimate}}`
            : "";
        }}
        if (autoSubmitCheckbox) autoSubmitCheckbox.checked = !!data.auto_submit;
      }}

      function loadDaconSubmissionLimit() {{
        fetch("/api/dacon-submission-limit")
          .then((response) => response.json())
          .then((data) => {{
            if (!data.applicable) {{
              if (card) card.hidden = true;
              return;
            }}
            renderValue(data);
            if (editToggle) editToggle.hidden = false;
            if (autoSubmitRow) autoSubmitRow.hidden = false;
          }})
          .catch(() => {{
            valueEl.textContent = "확인 안됨";
          }});
      }}

      if (autoSubmitCheckbox) {{
        autoSubmitCheckbox.addEventListener("change", () => {{
          const body = new URLSearchParams({{ enabled: autoSubmitCheckbox.checked ? "1" : "0" }});
          fetch("/api/dacon-auto-submit", {{ method: "POST", body }})
            .then((response) => response.json())
            .then((data) => {{
              if (data.ok === false) {{
                autoSubmitCheckbox.checked = !autoSubmitCheckbox.checked;
                return;
              }}
              renderValue(data);
            }});
        }});
      }}

      if (saveButton && editInput) {{
        saveButton.addEventListener("click", () => {{
          const value = editInput.value;
          if (!value || Number(value) <= 0) return;
          const body = new URLSearchParams({{ value }});
          fetch("/api/dacon-submission-limit", {{ method: "POST", body }})
            .then((response) => response.json())
            .then((data) => {{
              if (data.ok === false) return;
              renderValue(data);
              if (limitModal) setModalOpen(limitModal, false);
              editInput.value = "";
            }});
        }});
      }}

      loadDaconSubmissionLimit();
    }})();

    document.querySelectorAll("[data-submit-trial]").forEach((button) => {{
      button.addEventListener("click", () => {{
        const competition = button.dataset.competition;
        const trialId = button.dataset.trialId;
        const original = button.textContent;
        button.disabled = true;
        button.textContent = "제출 중...";
        const body = new URLSearchParams({{ competition, trial_id: trialId }});
        fetch("/api/submit-trial", {{ method: "POST", body }})
          .then((response) => response.json())
          .then((data) => {{
            if (data.ok === false) {{
              button.disabled = false;
              button.textContent = original;
              window.alert(data.message || "제출에 실패했습니다.");
              return;
            }}
            window.location.reload();
          }})
          .catch(() => {{
            button.disabled = false;
            button.textContent = original;
            window.alert("제출 요청 중 오류가 발생했습니다.");
          }});
      }});
    }});

    const modalOpenButtons = document.querySelectorAll("[data-open-modal]");
    const modalCloseButtons = document.querySelectorAll("[data-close-modal]");
    document.querySelectorAll("[data-continuous-toggle]").forEach((toggle) => {{
      const countInput = toggle.closest("form")?.querySelector("[data-trial-count]");
      const syncTrialCount = () => {{
        if (!countInput) return;
        countInput.disabled = toggle.checked;
        countInput.setAttribute("aria-disabled", toggle.checked ? "true" : "false");
      }};
      toggle.addEventListener("change", syncTrialCount);
      syncTrialCount();
    }});
    function setModalOpen(modal, open) {{
      if (!modal) return;
      modal.hidden = !open;
      document.body.style.overflow = open ? "hidden" : "";
      if (open) setTimeout(() => modal.querySelector("input, select, textarea, button")?.focus(), 0);
    }}
    modalOpenButtons.forEach((button) => {{
      button.addEventListener("click", () => setModalOpen(document.getElementById(button.dataset.openModal), true));
    }});
    modalCloseButtons.forEach((button) => {{
      button.addEventListener("click", () => setModalOpen(button.closest(".modal-backdrop"), false));
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape") {{
        const openModal = [...document.querySelectorAll(".modal-backdrop")].find((modal) => !modal.hidden);
        if (openModal) setModalOpen(openModal, false);
      }}
    }});
    if ([...document.querySelectorAll(".modal-backdrop")].some((modal) => !modal.hidden)) {{
      document.body.style.overflow = "hidden";
    }}

    const deleteModal = document.getElementById("delete-modal");
    const deleteStepConfirm = document.getElementById("delete-step-confirm");
    const deleteStepType = document.getElementById("delete-step-type");
    const deleteYesButton = document.getElementById("delete-step-yes");
    const deleteBackButton = document.getElementById("delete-step-back");
    const deleteConfirmInput = document.getElementById("delete-confirm-input");
    const deleteConfirmSubmit = document.getElementById("delete-confirm-submit");
    function resetDeleteModal() {{
      if (!deleteStepConfirm || !deleteStepType) return;
      deleteStepConfirm.hidden = false;
      deleteStepType.hidden = true;
      if (deleteConfirmInput) deleteConfirmInput.value = "";
      if (deleteConfirmSubmit) deleteConfirmSubmit.disabled = true;
    }}
    document.querySelectorAll('[data-open-modal="delete-modal"]').forEach((button) => {{
      button.addEventListener("click", resetDeleteModal);
    }});
    deleteYesButton?.addEventListener("click", () => {{
      deleteStepConfirm.hidden = true;
      deleteStepType.hidden = false;
      deleteConfirmInput?.focus();
    }});
    deleteBackButton?.addEventListener("click", () => setModalOpen(deleteModal, false));
    deleteConfirmInput?.addEventListener("input", () => {{
      if (!deleteConfirmSubmit) return;
      deleteConfirmSubmit.disabled = deleteConfirmInput.value !== deleteConfirmSubmit.dataset.expected;
    }});

    const uploadDropZone = document.getElementById("upload-drop-zone");
    const uploadFileInput = document.getElementById("upload-file-input");
    const uploadFileList = document.getElementById("upload-file-list");
    let uploadSelectedFiles = [];
    function renderUploadFileList() {{
      if (!uploadFileList) return;
      uploadFileList.innerHTML = "";
      uploadSelectedFiles.forEach((file, index) => {{
        const item = document.createElement("li");
        const sizeKb = (file.size / 1024).toFixed(1);
        const label = document.createElement("span");
        label.textContent = `${{file.name}} (${{sizeKb}} KB)`;
        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.textContent = "×";
        removeButton.addEventListener("click", () => {{
          uploadSelectedFiles.splice(index, 1);
          renderUploadFileList();
        }});
        item.appendChild(label);
        item.appendChild(removeButton);
        uploadFileList.appendChild(item);
      }});
    }}
    function addUploadFiles(fileList) {{
      Array.from(fileList || []).forEach((file) => uploadSelectedFiles.push(file));
      renderUploadFileList();
    }}
    document.querySelectorAll('[data-open-modal="new-experiment-modal"]').forEach((button) => {{
      button.addEventListener("click", () => {{
        uploadSelectedFiles = [];
        renderUploadFileList();
      }});
    }});
    uploadDropZone?.addEventListener("click", () => uploadFileInput?.click());
    uploadDropZone?.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" || event.key === " ") {{
        event.preventDefault();
        uploadFileInput?.click();
      }}
    }});
    uploadFileInput?.addEventListener("change", (event) => {{
      addUploadFiles(event.target.files);
      uploadFileInput.value = "";
    }});
    ["dragover", "dragenter"].forEach((eventName) => {{
      uploadDropZone?.addEventListener(eventName, (event) => {{
        event.preventDefault();
        uploadDropZone.classList.add("dragover");
      }});
    }});
    ["dragleave", "dragend"].forEach((eventName) => {{
      uploadDropZone?.addEventListener(eventName, () => uploadDropZone.classList.remove("dragover"));
    }});
    uploadDropZone?.addEventListener("drop", (event) => {{
      event.preventDefault();
      uploadDropZone.classList.remove("dragover");
      addUploadFiles(event.dataTransfer?.files);
    }});

    const registerForm = document.getElementById("new-experiment-register-form");
    const registerStatus = document.getElementById("register-status");
    registerForm?.addEventListener("submit", async (event) => {{
      const submitter = event.submitter;
      if (!submitter || submitter.value !== "new_experiment") return;
      event.preventDefault();
      submitter.disabled = true;
      if (registerStatus) registerStatus.textContent = "등록 중...";
      try {{
        const formData = new FormData(registerForm);
        formData.set("action", "new_experiment");
        const response = await fetch("/api/register-experiment", {{ method: "POST", body: new URLSearchParams(formData) }});
        const result = await response.json();
        let finalMessage = result.message;
        if (result.ok && uploadSelectedFiles.length) {{
          if (registerStatus) registerStatus.textContent = "데이터 업로드 중...";
          const uploadForm = new FormData();
          uploadSelectedFiles.forEach((file) => uploadForm.append("file", file, file.name));
          const uploadResponse = await fetch("/api/upload-data", {{ method: "POST", body: uploadForm }});
          const uploadResult = await uploadResponse.json();
          finalMessage += uploadResult.ok
            ? ` (데이터 ${{(uploadResult.saved || []).length}}개 파일 업로드됨)`
            : ` (데이터 업로드 실패: ${{uploadResult.message || "알 수 없는 오류"}})`;
        }}
        window.location.href = "/?" + new URLSearchParams({{ message: finalMessage }}).toString();
      }} catch (error) {{
        window.location.href = "/?" + new URLSearchParams({{ message: "등록 중 오류가 발생했습니다: " + error }}).toString();
      }}
    }});

    const artifactModal = document.getElementById("artifact-modal");
    const artifactModalTitle = document.getElementById("artifact-modal-title");
    const artifactModalPath = document.getElementById("artifact-modal-path");
    const artifactModalBody = document.getElementById("artifact-modal-body");
    document.querySelectorAll("[data-open-artifact]").forEach((trigger) => {{
      trigger.addEventListener("click", async () => {{
        const path = trigger.dataset.openArtifact;
        if (!path || !artifactModal) return;
        if (artifactModalTitle) artifactModalTitle.textContent = trigger.textContent.trim() || "산출물";
        if (artifactModalPath) artifactModalPath.textContent = path;
        if (artifactModalBody) artifactModalBody.textContent = "불러오는 중...";
        setModalOpen(artifactModal, true);
        try {{
          const response = await fetch(`/api/artifact?path=${{encodeURIComponent(path)}}`);
          const payload = await response.json();
          if (!response.ok || !payload.ok) {{
            if (artifactModalBody) artifactModalBody.textContent = payload.message || "산출물을 불러오지 못했습니다.";
            return;
          }}
          if (artifactModalTitle) artifactModalTitle.textContent = payload.title || "산출물";
          if (artifactModalBody) artifactModalBody.textContent = payload.content || "";
        }} catch (error) {{
          if (artifactModalBody) artifactModalBody.textContent = `산출물을 불러오는 중 오류가 발생했습니다: ${{error}}`;
        }}
      }});
    }});

    const trialTable = document.querySelector(".trial-table");
    const pagination = document.querySelector("[data-trial-pagination]");
    if (trialTable && pagination) {{
      const rows = [...trialTable.querySelectorAll("[data-trial-row]")];
      const pageSize = Number(trialTable.dataset.pageSize || 5);
      const pageCount = Math.ceil(rows.length / pageSize);
      function showTrialPage(page) {{
        rows.forEach((row, index) => {{
          row.hidden = index < (page - 1) * pageSize || index >= page * pageSize;
        }});
        pagination.querySelectorAll("button").forEach((button) => {{
          const current = Number(button.dataset.page) === page;
          button.setAttribute("aria-current", current ? "page" : "false");
        }});
      }}
      if (pageCount > 1) {{
        for (let page = 1; page <= pageCount; page += 1) {{
          const button = document.createElement("button");
          button.type = "button";
          button.dataset.page = String(page);
          button.textContent = String(page);
          button.setAttribute("aria-label", `Trial 목록 ${{page}}페이지`);
          button.addEventListener("click", () => showTrialPage(page));
          pagination.appendChild(button);
        }}
        showTrialPage(1);
      }} else {{
        pagination.hidden = true;
      }}
    }}

    const questionForm = document.getElementById("question-form");
    const chatLog = document.getElementById("chat-log");
    const chatWidget = document.getElementById("chat-widget");
    const chatFab = document.getElementById("chat-fab");
    const chatClose = document.getElementById("chat-close");
    const chatResizeHandle = document.getElementById("chat-resize-handle");
    const chatSessionSelect = document.getElementById("chat-session-select");
    const chatNewSession = document.getElementById("chat-new-session");
    const chatOpenButtons = document.querySelectorAll("[data-open-chat]");
    const chatSizeStorageKey = "research-agent-chat-size";
    const minimumChatWidth = 390;
    function minimumChatHeight() {{
      return Math.min(620, Math.max(180, window.innerHeight - 126));
    }}
    function clamp(value, minimum, maximum) {{
      return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
    }}
    function applyChatSize(width, height, persist = true) {{
      if (!chatWidget || window.innerWidth <= 900) return;
      const nextWidth = clamp(width, minimumChatWidth, window.innerWidth - 48);
      const nextHeight = clamp(height, minimumChatHeight(), window.innerHeight - 110);
      chatWidget.style.width = `${{Math.round(nextWidth)}}px`;
      chatWidget.style.height = `${{Math.round(nextHeight)}}px`;
      if (persist) {{
        try {{
          localStorage.setItem(chatSizeStorageKey, JSON.stringify({{width: nextWidth, height: nextHeight}}));
        }} catch (error) {{
          // The chat remains resizable when browser storage is unavailable.
        }}
      }}
    }}
    function restoreChatSize() {{
      if (!chatWidget || window.innerWidth <= 900) return;
      try {{
        const saved = JSON.parse(localStorage.getItem(chatSizeStorageKey) || "null");
        if (saved && Number.isFinite(saved.width) && Number.isFinite(saved.height)) {{
          applyChatSize(saved.width, saved.height, false);
        }}
      }} catch (error) {{
        localStorage.removeItem(chatSizeStorageKey);
      }}
    }}
    if (chatResizeHandle && chatWidget) {{
      chatResizeHandle.addEventListener("pointerdown", (event) => {{
        if (window.innerWidth <= 900) return;
        event.preventDefault();
        chatResizeHandle.setPointerCapture(event.pointerId);
        const startX = event.clientX;
        const startY = event.clientY;
        const startWidth = chatWidget.getBoundingClientRect().width;
        const startHeight = chatWidget.getBoundingClientRect().height;
        chatWidget.classList.add("is-resizing");
        const resize = (moveEvent) => {{
          applyChatSize(
            startWidth + startX - moveEvent.clientX,
            startHeight + startY - moveEvent.clientY,
            false,
          );
        }};
        const finish = () => {{
          chatResizeHandle.removeEventListener("pointermove", resize);
          chatResizeHandle.removeEventListener("pointerup", finish);
          chatResizeHandle.removeEventListener("pointercancel", finish);
          chatWidget.classList.remove("is-resizing");
          const bounds = chatWidget.getBoundingClientRect();
          applyChatSize(bounds.width, bounds.height, true);
        }};
        chatResizeHandle.addEventListener("pointermove", resize);
        chatResizeHandle.addEventListener("pointerup", finish);
        chatResizeHandle.addEventListener("pointercancel", finish);
      }});
      chatResizeHandle.addEventListener("keydown", (event) => {{
        if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        const bounds = chatWidget.getBoundingClientRect();
        const step = event.shiftKey ? 48 : 16;
        const width = bounds.width + (event.key === "ArrowLeft" ? step : event.key === "ArrowRight" ? -step : 0);
        const height = bounds.height + (event.key === "ArrowUp" ? step : event.key === "ArrowDown" ? -step : 0);
        applyChatSize(width, height);
      }});
    }}
    restoreChatSize();
    window.addEventListener("resize", () => {{
      if (!chatWidget || window.innerWidth <= 900) return;
      const bounds = chatWidget.getBoundingClientRect();
      applyChatSize(bounds.width, bounds.height, false);
    }});
    function setChatOpen(open) {{
      if (!chatWidget || !chatFab) return;
      chatWidget.hidden = !open;
      chatFab.setAttribute("aria-expanded", open ? "true" : "false");
      chatFab.textContent = open ? "x" : "AI";
      if (open) {{
        const textarea = questionForm?.querySelector("textarea[name='question']");
        setTimeout(() => textarea?.focus(), 0);
        chatLog.scrollTop = chatLog.scrollHeight;
      }}
    }}
    function appendChatMessage(kind, text, meta = "") {{
      if (!chatLog || !text) return;
      const item = document.createElement("div");
      item.className = `chat-message ${{kind}}`;
      const content = document.createElement("span");
      content.textContent = text;
      item.appendChild(content);
      if (meta) {{
        const detail = document.createElement("small");
        detail.className = "chat-message-meta";
        detail.textContent = meta;
        item.appendChild(detail);
      }}
      chatLog.appendChild(item);
      chatLog.scrollTop = chatLog.scrollHeight;
    }}
    function renderChatHistory(payload) {{
      if (!chatLog || !chatSessionSelect || !payload) return;
      const sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
      const activeSessionId = payload.active_session?.session_id || "";
      chatSessionSelect.replaceChildren();
      if (!sessions.length) {{
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "저장된 대화 없음";
        chatSessionSelect.appendChild(empty);
        chatSessionSelect.disabled = true;
      }} else {{
        sessions.forEach((session) => {{
          const option = document.createElement("option");
          option.value = session.session_id || "";
          option.textContent = session.title || "새 대화";
          option.selected = option.value === activeSessionId;
          chatSessionSelect.appendChild(option);
        }});
        chatSessionSelect.disabled = false;
      }}
      chatLog.replaceChildren();
      const messages = Array.isArray(payload.messages) ? payload.messages : [];
      if (!messages.length) {{
        appendChatMessage(
          "agent",
          "궁금한 점을 물어보세요. 현재 선택된 실험의 사용자용 산출물과 내부 기록을 기준으로 답변합니다.",
        );
      }} else {{
        messages.forEach((message) => {{
          const kind = message.role === "assistant" ? "agent" : (message.role || "system");
          appendChatMessage(kind, message.content || "", message.trial_id || "");
        }});
      }}
    }}
    chatFab?.addEventListener("click", () => setChatOpen(chatWidget?.hidden));
    chatClose?.addEventListener("click", () => setChatOpen(false));
    chatOpenButtons.forEach((button) => button.addEventListener("click", () => setChatOpen(true)));
    chatSessionSelect?.addEventListener("change", async () => {{
      const sessionId = chatSessionSelect.value;
      if (!sessionId) return;
      try {{
        const response = await fetch(`/api/chat/history?session_id=${{encodeURIComponent(sessionId)}}`);
        const payload = await response.json();
        if (!response.ok || !payload.ok) {{
          appendChatMessage("system", payload.message || "이전 대화를 불러오지 못했습니다.");
          return;
        }}
        renderChatHistory(payload);
      }} catch (error) {{
        appendChatMessage("system", `이전 대화를 불러오는 중 오류가 발생했습니다: ${{error}}`);
      }}
    }});
    chatNewSession?.addEventListener("click", async () => {{
      try {{
        const response = await fetch("/api/chat/session", {{method: "POST"}});
        const payload = await response.json();
        if (!response.ok || !payload.ok) {{
          appendChatMessage("system", payload.message || "새 대화를 만들지 못했습니다.");
          return;
        }}
        renderChatHistory(payload);
        setChatOpen(true);
      }} catch (error) {{
        appendChatMessage("system", `새 대화를 만드는 중 오류가 발생했습니다: ${{error}}`);
      }}
    }});
    if (questionForm && chatLog) {{
      const textarea = questionForm.querySelector("textarea[name='question']");
      textarea?.addEventListener("keydown", (event) => {{
        if (event.key === "Enter" && !event.shiftKey) {{
          event.preventDefault();
          questionForm.requestSubmit();
        }}
      }});
      questionForm.addEventListener("submit", async (event) => {{
        event.preventDefault();
        const button = questionForm.querySelector("button");
        const question = (textarea?.value || "").trim();
        if (!question) {{
          setChatOpen(true);
          appendChatMessage("system", "질문을 입력해주세요.");
          return;
        }}
        setChatOpen(true);
        appendChatMessage("user", question);
        textarea.value = "";
        if (button) button.disabled = true;
        const loading = document.createElement("div");
        loading.className = "chat-loading";
        loading.textContent = "에이전트가 답변을 작성하는 중입니다...";
        chatLog.appendChild(loading);
        chatLog.scrollTop = chatLog.scrollHeight;
        try {{
          const response = await fetch("/api/question", {{
            method: "POST",
            headers: {{"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}},
            body: new URLSearchParams({{
              question,
              session_id: chatSessionSelect?.value || "",
            }}),
          }});
          const payload = await response.json();
          loading.remove();
          if (!response.ok || !payload.ok) {{
            appendChatMessage("system", payload.message || "답변 생성에 실패했습니다.");
          }} else {{
            renderChatHistory(payload.history);
          }}
        }} catch (error) {{
          loading.remove();
          appendChatMessage("system", `요청 중 오류가 발생했습니다: ${{error}}`);
        }} finally {{
          if (button) button.disabled = false;
          textarea.focus();
        }}
      }});
    }}
  </script>
</body>
</html>"""


def metric_card(label: str, value: Any) -> str:
    return f'<div class="metric"><div class="label">{escape(label)}</div><div class="value">{value}</div></div>'


def experiment_options(experiments: list[dict[str, Any]], selected: Any) -> str:
    selected_text = str(selected or "")
    parts = []
    for item in experiments:
        competition = str(item.get("competition") or "")
        label = f"{competition} | {item.get('topic') or '-'} | {item.get('state') or '-'}"
        chosen = " selected" if competition == selected_text else ""
        parts.append(f'<option value="{escape(competition)}"{chosen}>{escape(label)}</option>')
    return "\n".join(parts)


def pending_list(requests: list[dict[str, Any]]) -> str:
    if not requests:
        return '<p class="empty">피드백 요청이 없습니다.</p>'
    items = []
    for request in requests:
        request_id = str(request.get("request_id") or "")
        title = request.get("title") or request.get("type") or "요청"
        question = request.get("question") or request.get("message") or ""
        evidence = "".join(
            "<div class='feedback-evidence'>"
            f"<span>{escape(item.get('label') or '근거')}</span>"
            f"<strong>{escape(item.get('value') or '-')}</strong>"
            f"<small>{escape(item.get('meaning') or '')}</small>"
            "</div>"
            for item in request.get("evidence_snapshot") or []
            if isinstance(item, dict)
        )
        repeated_evidence = "".join(
            f"<li>{escape(str(item))}</li>"
            for item in request.get("evidence_summary") or []
            if str(item).strip()
        )
        policy = request.get("policy") if isinstance(request.get("policy"), dict) else {}
        policy_score = (
            "<p class='feedback-policy-score'><strong>요청 필요도</strong><br>"
            f"{escape(policy.get('score'))} / {escape(policy.get('threshold'))}"
            "<small>회차 수가 아니라 반복 근거와 사용자 판단 필요성을 기준으로 계산합니다.</small></p>"
            if policy.get("score") is not None
            else ""
        )
        options = "".join(
            "<label class='feedback-option'>"
            f"<input type='radio' name='decision' value='{escape(item.get('value') or item.get('label') or '')}' required>"
            f"<span><strong>{escape(item.get('label') or item.get('value') or '선택')}</strong>"
            f"<small>{escape(item.get('impact') or '')}</small></span>"
            "</label>"
            for item in request.get("options") or []
            if isinstance(item, dict)
        )
        execution_note = (
            "<p class='feedback-execution-note'><strong>실행 안내</strong><br>"
            f"{escape(request.get('execution_note') or '현재 버전에서는 이 답변이 외부 계산 환경을 자동 실행하지 않습니다.')}"
            "</p>"
            if request.get("execution_supported") is False
            else ""
        )
        support_context = (
            (f"<div><strong>반복 근거</strong><ul>{repeated_evidence}</ul></div>" if repeated_evidence else "")
            + policy_score
        )
        items.append(
            "<li class='feedback-request-card'>"
            f"<span class='feedback-type'>{escape(request.get('interaction_label') or '사용자 판단 요청')}</span>"
            f"<h3>{escape(title)}</h3>"
            f"<p class='feedback-problem'><strong>문제</strong><br>{escape(request.get('problem') or request.get('message') or '-')}</p>"
            f"<div class='feedback-evidence-grid'>{evidence}</div>"
            f"{support_context}"
            f"<p><strong>에이전트 해석</strong><br>{escape(request.get('interpretation') or '-')}</p>"
            f"<p><strong>에이전트 추천</strong><br>{escape(request.get('recommendation') or '-')}</p>"
            f"<p><strong>왜 사용자에게 묻나요?</strong><br>{escape(request.get('why_user_needed') or '-')}</p>"
            f"<p><strong>확인할 질문</strong><br>{escape(question)}</p>"
            f'<form method="post" action="/action">{hidden("action", "feedback")}'
            f'{hidden("request_id", request_id)}'
            f"<div class='feedback-options'>{options}</div>"
            '<textarea name="answer" placeholder="선택 이유나 추가 정보를 입력하세요. 선택지가 없다면 여기에 답변을 입력하세요."></textarea>'
            f"<p class='feedback-default'><strong>답변이 없을 때</strong><br>{escape(request.get('default_if_no_response') or '-')}</p>"
            f"{execution_note}"
            "<button>답변 기록</button></form>"
            "</li>"
        )
    return "<ul>" + "\n".join(items) + "</ul>"


def insight_hint(existing: Any, snapshot: dict[str, Any]) -> str:
    trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial") or "trial_001"
    next_trial = snapshot.get("next_trial") or _next_trial_after(str(trial_id)) or "다음"
    # latest_user_insight_record already excludes insights that finished their
    # lifecycle (superseded/completed/exhausted). If it returns nothing, the raw
    # feedback text in `existing` is stale history, not something still pending --
    # showing it here would look like an unresolved duplicate even though it was
    # already applied and closed out.
    record = latest_user_insight_record(str(snapshot.get("competition") or ""), str(trial_id)) if existing else None
    if record:
        text = "\n".join(
            [
                "이미 인사이트가 제공되었습니다",
                f": {existing}",
                "",
                _format_insight_plan_message(str(existing), str(next_trial), include_insight=False),
                f"- 현재 상태: {record.get('status') or 'pending'}",
                f"- 개선축: {record.get('axis') or '해석 대기'}",
            ]
        )
    else:
        text = "코드 작성 전 계획 회차가 있으면 해당 계획을 수정하고, 실행 중이면 다음 trial 계획에 반영합니다."
    return f'<p class="hint">{escape(text)}</p>'


def persistent_insight_notice(existing: Any, snapshot: dict[str, Any]) -> str:
    if not existing:
        return ""
    trial_id = snapshot.get("current_trial") or snapshot.get("last_completed_trial") or "trial_001"
    next_trial = snapshot.get("next_trial") or _next_trial_after(str(trial_id)) or "다음"
    # Same reasoning as insight_hint: only show this banner while the insight is
    # genuinely still open. Once it is superseded/completed/exhausted, stop
    # displaying it instead of rendering placeholder "pending"/"해석 대기" text
    # that makes an already-resolved insight look unresolved.
    record = latest_user_insight_record(str(snapshot.get("competition") or ""), str(trial_id))
    if not record:
        return ""
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    intent = (
        interpretation.get("implementation_intent")
        if isinstance(interpretation.get("implementation_intent"), dict)
        else {}
    )
    target_trial = record.get("target_trial") or next_trial
    message = "\n".join(
        [
            "인사이트:",
            str(existing),
            "",
            f"- 반영 예정: {target_trial}",
            f"- 적용 개선안: {intent.get('change') or '다음 계획 단계에서 구체화'}",
            f"- 상태: {record.get('status') or 'pending'}",
            f"- 개선축: {record.get('axis') or '해석 대기'}",
        ]
    )
    return notice(message)


def notice(message: str) -> str:
    if not message:
        return ""
    if message.startswith("인사이트:\n"):
        body = message.split("\n", 1)[1]
        return (
            '<section class="notice">'
            '<strong class="notice-title">인사이트:</strong>'
            f'<div class="notice-body">{escape(body)}</div>'
            "</section>"
        )
    return f'<section class="notice">{escape(message)}</section>'


def answer_block(answer: str) -> str:
    return f'<div class="chat-message agent">{escape(answer)}</div>' if answer else ""


def hidden(name: str, value: str) -> str:
    return f'<input type="hidden" name="{escape(name)}" value="{escape(value)}">'


def score(value: Any) -> str:
    return "-" if value is None else str(value)


def clean(value: str) -> str:
    return str(value or "").strip()


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _next_trial_after(trial_id: str) -> str | None:
    prefix, _, number = trial_id.rpartition("_")
    try:
        return f"{prefix}_{int(number) + 1:03d}"
    except ValueError:
        return None


def main() -> int:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), ResearchAgentHandler)
    print(f"Research Agent web app listening on port {port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
