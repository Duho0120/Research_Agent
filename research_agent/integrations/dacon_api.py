from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

# DACON's leaderboard page server-renders only the top 100 rows as visible
# <li> HTML, but the page's own "전체보기" (view all) button reveals every
# ranked team from data already embedded in the same response -- inside a
# minified `window.__NUXT__=(function(...){...})(...)` payload that must be
# evaluated as JavaScript (it is not valid JSON: Nuxt's build step dedupes
# repeated strings into positional function arguments). This script does
# that evaluation in an isolated Node vm and prints the full row list as
# JSON, so ranks beyond 100 (like a personal rank of 129) are reachable.
_NUXT_EXTRACT_SCRIPT = Path(__file__).resolve().parents[2] / "vendor" / "dacon_nuxt_extract.js"

# DACON has no CLI (unlike Kaggle's `kaggle` command). Submissions go through
# the `dacon_submit_api` Python package: pip install dacon_submit_api, then
#   from dacon_submit_api import dacon_submit_api
#   dacon_submit_api.post_submission_file(file_path, personal_token, competition_id, team_name, memo)
# The personal token is issued from the user's DACON account settings page
# (계정관리 > Open API 개인 Token 발급) -- never hard-code it; it is read from
# an environment variable so it is never written into experiment artifacts
# or committed to the repo.
TOKEN_ENV_VAR = "DACON_PERSONAL_TOKEN"

SubmitFn = Callable[[str, str, str, str, str], Any]

# dacon_submit_api's own decode_submission_response() maps these DACON API
# message codes to human text; replicated here since post_submission_file
# returns the raw (non-JSON-serializable) requests.Response, not this decoded
# form -- our adapter decodes it so results can be written to trial JSON.
_SUBMISSION_ERROR_MESSAGES = {
    "over_max_count": "Over max submission count of Competition.",
    "day_max_count": "Over max submission count of Daily.",
}
_SUBMISSION_ERROR_CODES = {
    1: "csv file header error.",
    2: "data parsing error.",
    3: "number of row error.",
    4: "other submission value error.",
    21: "data type error.",
    901: "csv file's header doesn't match.",
    902: "error occurs in data parsing.",
    903: "number of row doesn't match.",
    904: "submission value occurs error.",
    9021: "data must not have Character.",
}


def check_package_available(*, import_fn: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Check that the dacon_submit_api package is importable."""
    try:
        if import_fn is not None:
            import_fn()
        else:
            import dacon_submit_api  # noqa: F401
    except ImportError as exc:
        return {
            "action": "check_package",
            "ok": False,
            "status": "package_unavailable",
            "error": (
                f"{exc}. Install with: pip install dacon_submit_api "
                "(DACON distributes this as a wheel file linked from their submission-API forum post)."
            ),
        }
    return {"action": "check_package", "ok": True, "status": "ok"}


def check_auth(*, token: str | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Check that a personal token is available (env var, unless overridden for testing)."""
    resolved = token or (env or {}).get(TOKEN_ENV_VAR)
    if not resolved:
        return {
            "action": "check_auth",
            "ok": False,
            "status": "token_missing",
            "error": f"{TOKEN_ENV_VAR} environment variable is not set. Issue a personal token from DACON account settings.",
        }
    return {"action": "check_auth", "ok": True, "status": "ok"}


def submit_competition(
    *,
    competition_id: str,
    submission_file: str,
    team_name: str,
    message: str = "",
    token: str | None = None,
    env: dict[str, str] | None = None,
    submit_fn: SubmitFn | None = None,
) -> dict[str, Any]:
    """Submit a file to a DACON competition via dacon_submit_api.

    Returns a result shaped like kaggle_cli's _structured_result (ok/status/
    action/error keys) so agents/submission.py can dispatch between platforms
    without caring which one actually ran.
    """
    auth = check_auth(token=token, env=env)
    if not auth["ok"]:
        return {"action": "submit", "ok": False, "status": auth["status"], "error": auth["error"]}
    resolved_token = token or (env or {}).get(TOKEN_ENV_VAR) or ""

    if submit_fn is None:
        package_check = check_package_available()
        if not package_check["ok"]:
            return {
                "action": "submit",
                "ok": False,
                "status": package_check["status"],
                "error": package_check["error"],
            }
        from dacon_submit_api import dacon_submit_api as _dacon_module

        submit_fn = _dacon_module.post_submission_file

    if not Path(submission_file).is_file():
        return {
            "action": "submit",
            "ok": False,
            "status": "submission_file_missing",
            "error": f"Submission file not found: {submission_file}",
        }

    try:
        raw = submit_fn(submission_file, resolved_token, competition_id, team_name, message)
    except Exception as exc:  # noqa: BLE001 - the package's own error surface is undocumented
        return {"action": "submit", "ok": False, "status": "submit_error", "error": str(exc)}

    return _decode_submit_result(raw)


def _decode_submit_result(raw: Any) -> dict[str, Any]:
    """Turn post_submission_file's return value into a JSON-serializable result.

    post_submission_file (from the real dacon_submit_api package) returns the
    raw requests.Response object, not a parsed dict -- storing that directly
    would break anything that later json.dumps()s the submission result. A
    fake submit_fn used in tests may already return a plain dict, so both
    shapes are handled.
    """
    payload: dict[str, Any]
    if hasattr(raw, "json"):
        try:
            payload = raw.json()
        except ValueError:
            return {
                "action": "submit",
                "ok": False,
                "status": "unparseable_response",
                "error": f"DACON response was not JSON (status_code={getattr(raw, 'status_code', '?')}).",
            }
    elif isinstance(raw, dict):
        payload = raw
    else:
        return {"action": "submit", "ok": True, "status": "submitted", "raw": str(raw)}

    message = payload.get("message")
    if message == "ok":
        return {"action": "submit", "ok": True, "status": "submitted", "raw": payload}
    if message in _SUBMISSION_ERROR_MESSAGES:
        return {
            "action": "submit",
            "ok": False,
            "status": "submit_rejected",
            "error": _SUBMISSION_ERROR_MESSAGES[message],
            "raw": payload,
        }
    if message == "wrong":
        code = payload.get("data")
        return {
            "action": "submit",
            "ok": False,
            "status": "submit_rejected",
            "error": _SUBMISSION_ERROR_CODES.get(code, f"Submission rejected (code={code})."),
            "raw": payload,
        }
    # Unrecognized message shape: surface it rather than silently claiming success.
    return {"action": "submit", "ok": False, "status": "unrecognized_response", "error": str(payload), "raw": payload}


# DACON's own submission API (used by submit_competition above) has no
# endpoint for "what have I submitted" -- that only exists behind a
# session-authenticated JSON endpoint the /mysubmission page's own frontend
# calls (confirmed from a real logged-in browser's Network tab):
#   GET https://newapi.dacon.io/competition/submission_list?page=1&data_type=recent&cpt_id={id}
# Auth is a "Token" request header carrying the same JWT DACON issues on
# login -- not a Cookie header and not "Authorization: Bearer", despite the
# env var name below predating that discovery. We deliberately never
# automate the login step (email/password or OAuth) that produces this
# token -- that stays a manual, user-controlled action, exactly like
# DACON_PERSONAL_TOKEN is a manually issued token, never something this
# codebase requests or types in on the user's behalf.
SESSION_COOKIE_ENV_VAR = "DACON_SESSION_COOKIE"


def leaderboard_url(competition_id: str) -> str:
    return f"https://dacon.io/competitions/official/{competition_id}/leaderboard"


def submission_history_url(competition_id: str) -> str:
    return f"https://dacon.io/competitions/official/{competition_id}/mysubmission"


def submission_list_api_url(competition_id: str, *, page: int = 1, data_type: str = "recent") -> str:
    return (
        "https://newapi.dacon.io/competition/submission_list"
        f"?page={page}&data_type={data_type}&cpt_id={competition_id}"
    )


def competition_id_from_link(link: str) -> str | None:
    """Extract the numeric competition id from any dacon.io competition URL,
    e.g. https://dacon.io/competitions/official/236689/overview/description
    or the .../leaderboard variant -- both work since only the id is used.
    """
    match = re.search(r"/competitions/(?:official|open)/(\d+)", link)
    return match.group(1) if match else None


def rules_url(competition_id: str) -> str:
    return f"https://dacon.io/competitions/official/{competition_id}/overview/rules"


def fetch_daily_submission_limit(
    competition_id: str,
    *,
    fetch_html_fn: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Look up a competition's stated daily submission limit from its public
    rules page.

    DACON has no API for this. The number only exists as free text inside
    the rules page's rich-text "4. 유의 사항" section, e.g.
    "1일 최대 제출 횟수: 5회" -- unlike the leaderboard, this text is present
    in the server-rendered HTML directly (confirmed by inspection), so no
    NUXT payload evaluation is needed here.

    Many competitions may phrase this differently or omit it entirely, so a
    miss is reported as status "not_found" rather than an error -- callers
    must treat that as "unknown", never as "no limit". The caller is
    responsible for offering a manual override when this happens.
    """
    if fetch_html_fn is None:
        fetch_html_fn = _fetch_rules_html
    fetched = fetch_html_fn(competition_id)
    if not fetched.get("ok"):
        return {
            "action": "fetch_daily_submission_limit",
            "ok": False,
            "status": fetched.get("status", "fetch_error"),
            "error": fetched.get("error"),
        }
    limit = _parse_daily_submission_limit(fetched["html"])
    if limit is None:
        return {
            "action": "fetch_daily_submission_limit",
            "ok": False,
            "status": "not_found",
            "error": "Could not find a daily submission limit on the rules page.",
        }
    return {
        "action": "fetch_daily_submission_limit",
        "ok": True,
        "status": "found",
        "daily_submission_limit": limit,
    }


def _fetch_rules_html(competition_id: str) -> dict[str, Any]:
    import requests

    url = rules_url(competition_id)
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors vary by cause
        return {"ok": False, "status": "fetch_error", "error": str(exc)}
    return {"ok": True, "html": response.text}


_DAILY_SUBMISSION_LIMIT_PATTERN = re.compile(
    r"1\s*일\s*(?:최대\s*)?제출\s*(?:가능\s*)?횟수\s*[:：]?\s*(\d+)\s*회"
)


def _parse_daily_submission_limit(html: str) -> int | None:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    match = _DAILY_SUBMISSION_LIMIT_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def overview_description_url(competition_id: str) -> str:
    return f"https://dacon.io/competitions/official/{competition_id}/overview/description"


def data_page_url(competition_id: str) -> str:
    return f"https://dacon.io/competitions/official/{competition_id}/data"


def fetch_competition_overview(
    competition_id: str,
    *,
    fetch_html_fn: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scrape a DACON competition's "개요" (overview/description) tab: the
    free-text explanation of the task, evaluation, and schedule.

    This is the piece a Kaggle-only competition_onboarding.py could already
    do via the Kaggle CLI but had no DACON equivalent for, leaving
    overview.md as unhelpful placeholder text after every DACON registration.
    """
    if fetch_html_fn is None:
        fetch_html_fn = _fetch_overview_html
    fetched = fetch_html_fn(competition_id)
    if not fetched.get("ok"):
        return {
            "action": "fetch_competition_overview",
            "ok": False,
            "status": fetched.get("status", "fetch_error"),
            "error": fetched.get("error"),
        }
    text = _extract_first_ql_editor_text(fetched["html"])
    if not text:
        return {
            "action": "fetch_competition_overview",
            "ok": False,
            "status": "not_found",
            "error": "Could not find an overview/description section on the page.",
        }
    return {"action": "fetch_competition_overview", "ok": True, "status": "found", "text": text}


def fetch_competition_data_description(
    competition_id: str,
    *,
    fetch_html_fn: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scrape a DACON competition's "데이터" tab: per-file descriptions the
    organizer wrote (e.g. what each CSV's columns mean), which is real
    domain knowledge no amount of profiling the uploaded files themselves
    can recover.
    """
    if fetch_html_fn is None:
        fetch_html_fn = _fetch_data_page_html
    fetched = fetch_html_fn(competition_id)
    if not fetched.get("ok"):
        return {
            "action": "fetch_competition_data_description",
            "ok": False,
            "status": fetched.get("status", "fetch_error"),
            "error": fetched.get("error"),
        }
    text = _extract_first_ql_editor_text(fetched["html"])
    if not text:
        return {
            "action": "fetch_competition_data_description",
            "ok": False,
            "status": "not_found",
            "error": "Could not find a data description section on the page.",
        }
    return {"action": "fetch_competition_data_description", "ok": True, "status": "found", "text": text}


def _extract_first_ql_editor_text(html: str) -> str | None:
    """DACON renders each free-text section (overview, per-file data
    description) inside a rich-text editor's own container
    (class="ql-editor"). A raw tag-strip over the whole page picks up
    megabytes of unrelated <script>/<style> content first, and the real
    content is duplicated later in the page by the client-hydration
    payload -- so this targets the editor container directly via
    BeautifulSoup and returns only the first non-empty one.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for container in soup.select(".ql-editor"):
        text = container.get_text(" ", strip=True)
        if text:
            return re.sub(r"\s+", " ", text).strip()
    return None


def _fetch_overview_html(competition_id: str) -> dict[str, Any]:
    import requests

    url = overview_description_url(competition_id)
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors vary by cause
        return {"ok": False, "status": "fetch_error", "error": str(exc)}
    return {"ok": True, "html": response.text}


def _fetch_data_page_html(competition_id: str) -> dict[str, Any]:
    import requests

    url = data_page_url(competition_id)
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors vary by cause
        return {"ok": False, "status": "fetch_error", "error": str(exc)}
    return {"ok": True, "html": response.text}


def fetch_leaderboard(
    competition_id: str,
    *,
    team_name: str,
    fetch_html_fn: Callable[[str], dict[str, Any]] | None = None,
    rows_fn: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Look up one team's rank and score on a DACON competition's public
    leaderboard page.

    DACON's submission API has no endpoint for this (unlike Kaggle's CLI),
    but the leaderboard page itself is public, static, server-rendered HTML
    that does not require login -- so this scrapes it instead.
    """
    if fetch_html_fn is None:
        fetch_html_fn = _fetch_leaderboard_html
    if rows_fn is None:
        rows_fn = leaderboard_rows
    fetched = fetch_html_fn(competition_id)
    if not fetched.get("ok"):
        return {
            "action": "fetch_leaderboard",
            "ok": False,
            "status": fetched.get("status", "fetch_error"),
            "error": fetched.get("error"),
        }
    rows = rows_fn(fetched["html"])
    for row in rows:
        if row["team_name"] == team_name:
            return {
                "action": "fetch_leaderboard",
                "ok": True,
                "status": "found",
                "rank": row["rank"],
                "score": row["score"],
                "team_name": team_name,
            }
    return {
        "action": "fetch_leaderboard",
        "ok": False,
        "status": "team_not_found",
        "error": f"Team '{team_name}' was not found among {len(rows)} leaderboard rows.",
    }


def _fetch_leaderboard_html(competition_id: str) -> dict[str, Any]:
    import requests

    url = leaderboard_url(competition_id)
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors vary by cause
        return {"ok": False, "status": "fetch_error", "error": str(exc)}
    return {"ok": True, "html": response.text}


def _parse_leaderboard_rows(html: str) -> list[dict[str, Any]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    for record in soup.select("ul.record"):
        index_el = record.select_one("li.index span")
        team_el = record.select_one("li.team-name span")
        score_el = record.select_one("li.score span")
        if not (index_el and team_el and score_el):
            continue
        try:
            rank = int(index_el.get_text(strip=True))
        except ValueError:
            continue
        score_text = score_el.get_text(strip=True)
        try:
            score = float(score_text)
        except ValueError:
            score = None
        rows.append({"rank": rank, "team_name": team_el.get_text(strip=True), "score": score})
    return rows


def _full_leaderboard_rows_via_node(
    html: str,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, Any]] | None:
    """Extract every ranked row (not just the top 100) via the Node helper.

    Returns None (rather than raising) whenever this path is unavailable --
    Node not installed, the script missing, the payload shape changing, etc.
    -- so callers can fall back to the capped HTML parse instead of failing
    the whole leaderboard lookup over a best-effort enhancement.
    """
    if not _NUXT_EXTRACT_SCRIPT.is_file():
        return None
    try:
        result = run_fn(
            ["node", str(_NUXT_EXTRACT_SCRIPT)],
            input=html,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        raw_rows = json.loads(result.stdout)
    except ValueError:
        return None
    rows: list[dict[str, Any]] = []
    for entry in raw_rows:
        if not isinstance(entry, dict):
            continue
        rank = entry.get("ranking")
        team_name = entry.get("team_name")
        score = entry.get("score")
        if rank is None or team_name is None:
            continue
        rows.append({"rank": rank, "team_name": team_name, "score": score})
    return rows or None


def leaderboard_rows(
    html: str,
    *,
    full_rows_fn: Callable[[str], list[dict[str, Any]] | None] = _full_leaderboard_rows_via_node,
) -> list[dict[str, Any]]:
    """All parsed leaderboard rows, preferring the complete list (every rank)
    over the HTML-only parse (capped at the top 100 DACON renders as <li>).
    """
    full_rows = full_rows_fn(html)
    if full_rows is not None:
        return full_rows
    return _parse_leaderboard_rows(html)


def check_session_auth(*, token: str | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Check that a DACON session token is available (env var, unless overridden for testing).

    This token is never obtained by this codebase -- it must come from the
    user logging into DACON themselves (in their own browser) and copying
    the "Token" request header value out of their browser's devtools, the
    same way DACON_PERSONAL_TOKEN is a manually issued token, never
    something typed in on the user's behalf.
    """
    resolved = token or (env or {}).get(SESSION_COOKIE_ENV_VAR)
    if not resolved:
        return {
            "action": "check_session_auth",
            "ok": False,
            "status": "session_cookie_missing",
            "error": (
                f"{SESSION_COOKIE_ENV_VAR} environment variable is not set. Log into DACON in your own "
                "browser and copy the session token value (never a password) into this env var."
            ),
        }
    return {"action": "check_session_auth", "ok": True, "status": "ok"}


def fetch_my_submissions(
    competition_id: str,
    *,
    token: str | None = None,
    env: dict[str, str] | None = None,
    fetch_json_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """List the logged-in user's own submissions (score, time, etc.) for one
    DACON competition, via their session token.

    Confirmed against a real response (competition_id=236690, 43
    submissions): the endpoint returns
    {"status": 1, "message": <count>, "data": [{"sub_id", "score",
    "private_score", "file_link", "c_time", "is_late", "final_use", "memo",
    ...}, ...]} on success, or {"status": 1, "data": "no data"} when the
    account has no submissions for that competition.
    """
    auth = check_session_auth(token=token, env=env)
    if not auth["ok"]:
        return {"action": "fetch_my_submissions", "ok": False, "status": auth["status"], "error": auth["error"]}
    resolved_token = token or (env or {}).get(SESSION_COOKIE_ENV_VAR) or ""

    if fetch_json_fn is None:
        fetch_json_fn = _fetch_submission_list_json

    fetched = fetch_json_fn(competition_id, resolved_token)
    if not fetched.get("ok"):
        return {
            "action": "fetch_my_submissions",
            "ok": False,
            "status": fetched.get("status", "fetch_error"),
            "error": fetched.get("error"),
        }
    payload = fetched["data"]
    if not isinstance(payload, dict) or payload.get("status") != 1:
        return {
            "action": "fetch_my_submissions",
            "ok": False,
            "status": "unrecognized_response",
            "error": f"Unexpected response payload (token may be invalid/expired): {payload!r}",
        }
    data = payload.get("data")
    if data == "no data":
        return {"action": "fetch_my_submissions", "ok": True, "status": "found", "submissions": []}
    if not isinstance(data, list):
        return {
            "action": "fetch_my_submissions",
            "ok": False,
            "status": "unrecognized_response",
            "error": f"Unexpected 'data' shape in response: {data!r}",
        }
    return {"action": "fetch_my_submissions", "ok": True, "status": "found", "submissions": data}


def _fetch_submission_list_json(competition_id: str, token: str) -> dict[str, Any]:
    import requests

    url = submission_list_api_url(competition_id)
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Token": token,
            },
            timeout=15,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors vary by cause
        return {"ok": False, "status": "fetch_error", "error": str(exc)}
    try:
        payload = response.json()
    except ValueError:
        return {"ok": False, "status": "unparseable_response", "error": "DACON response was not JSON."}
    return {"ok": True, "data": payload}
