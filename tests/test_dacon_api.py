from __future__ import annotations

import json
import unittest

from research_agent.integrations import dacon_api


class DaconApiTest(unittest.TestCase):
    def test_check_auth_fails_without_token(self):
        result = dacon_api.check_auth(env={})
        self.assertFalse(result["ok"])
        self.assertEqual("token_missing", result["status"])

    def test_check_auth_passes_with_env_token(self):
        result = dacon_api.check_auth(env={dacon_api.TOKEN_ENV_VAR: "secret"})
        self.assertTrue(result["ok"])

    def test_check_auth_passes_with_explicit_token(self):
        result = dacon_api.check_auth(token="secret", env={})
        self.assertTrue(result["ok"])

    def test_check_package_available_reports_missing_package(self):
        def fake_import():
            raise ImportError("No module named 'dacon_submit_api'")

        result = dacon_api.check_package_available(import_fn=fake_import)
        self.assertFalse(result["ok"])
        self.assertEqual("package_unavailable", result["status"])
        self.assertIn("pip install dacon_submit_api", result["error"])

    def test_submit_competition_blocks_without_token(self):
        result = dacon_api.submit_competition(
            competition_id="235610",
            submission_file="submission.csv",
            team_name="team",
            env={},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("token_missing", result["status"])

    def test_submit_competition_blocks_when_file_missing(self):
        calls = []

        def fake_submit(*args):
            calls.append(args)
            return {"status": "ok"}

        result = dacon_api.submit_competition(
            competition_id="235610",
            submission_file="/nonexistent/submission.csv",
            team_name="team",
            token="secret",
            submit_fn=fake_submit,
        )
        self.assertFalse(result["ok"])
        self.assertEqual("submission_file_missing", result["status"])
        self.assertEqual([], calls)

    def test_submit_competition_calls_submit_fn_with_expected_argument_order(self):
        # {"message": "ok"} is what DACON's real API returns on success --
        # confirmed from the actual dacon_submit_api-0.0.4 source, which the
        # real post_submission_file returns wrapped in a requests.Response
        # (see test_decode_submit_result_* below for that unwrapping).
        import tempfile
        from pathlib import Path

        calls = []

        def fake_submit(file_path, token, competition_id, team_name, memo):
            calls.append((file_path, token, competition_id, team_name, memo))
            return {"message": "ok"}

        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.csv"
            submission.write_text("id,target\n1,0\n", encoding="utf-8")

            result = dacon_api.submit_competition(
                competition_id="235610",
                submission_file=str(submission),
                team_name="my-team",
                message="trial_005 CV 0.32",
                token="secret-token",
                submit_fn=fake_submit,
            )

        self.assertTrue(result["ok"])
        self.assertEqual("submitted", result["status"])
        self.assertEqual(
            [(str(submission), "secret-token", "235610", "my-team", "trial_005 CV 0.32")],
            calls,
        )

    def test_submit_competition_reports_rejection_message(self):
        import tempfile
        from pathlib import Path

        def fake_submit(*args):
            return {"message": "over_max_count"}

        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.csv"
            submission.write_text("id,target\n1,0\n", encoding="utf-8")

            result = dacon_api.submit_competition(
                competition_id="235610",
                submission_file=str(submission),
                team_name="my-team",
                token="secret-token",
                submit_fn=fake_submit,
            )

        self.assertFalse(result["ok"])
        self.assertEqual("submit_rejected", result["status"])
        self.assertIn("max submission count", result["error"])

    def test_submit_competition_reports_wrong_message_with_error_code(self):
        import tempfile
        from pathlib import Path

        def fake_submit(*args):
            return {"message": "wrong", "data": 901}

        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.csv"
            submission.write_text("id,target\n1,0\n", encoding="utf-8")

            result = dacon_api.submit_competition(
                competition_id="235610",
                submission_file=str(submission),
                team_name="my-team",
                token="secret-token",
                submit_fn=fake_submit,
            )

        self.assertFalse(result["ok"])
        self.assertEqual("submit_rejected", result["status"])
        self.assertIn("header doesn't match", result["error"])

    def test_decode_submit_result_unwraps_a_real_requests_response_object(self):
        # post_submission_file (the real package function) returns the raw
        # requests.Response, not a parsed dict -- this must not be stored
        # as-is (it is not JSON-serializable) and must be decoded the same
        # way the package's own decode_submission_response() would.
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"message": "ok"}

        result = dacon_api._decode_submit_result(FakeResponse())
        self.assertTrue(result["ok"])
        self.assertEqual("submitted", result["status"])
        self.assertEqual({"message": "ok"}, result["raw"])

    def test_decode_submit_result_handles_unparseable_response_body(self):
        class FakeResponse:
            status_code = 502

            def json(self):
                raise ValueError("not JSON")

        result = dacon_api._decode_submit_result(FakeResponse())
        self.assertFalse(result["ok"])
        self.assertEqual("unparseable_response", result["status"])

    def test_submit_competition_reports_submit_fn_exception(self):
        import tempfile
        from pathlib import Path

        def failing_submit(*args):
            raise RuntimeError("connection refused")

        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.csv"
            submission.write_text("id,target\n1,0\n", encoding="utf-8")

            result = dacon_api.submit_competition(
                competition_id="235610",
                submission_file=str(submission),
                team_name="my-team",
                token="secret-token",
                submit_fn=failing_submit,
            )

        self.assertFalse(result["ok"])
        self.assertEqual("submit_error", result["status"])
        self.assertIn("connection refused", result["error"])


# Trimmed to the parts _parse_leaderboard_rows reads, but the markup itself
# (tag names, "record"/"index"/"team-name"/"score" classes and nesting) is
# copied verbatim from a real public DACON leaderboard page fetched during
# development (dacon.io/competitions/official/236689/leaderboard), so this
# is a real regression test against DACON's actual HTML shape, not a shape
# invented to make the parser pass.
_REAL_LEADERBOARD_HTML_FRAGMENT = """
<ul class="record fl winner-group">
  <li class="rank fl tc"><div class="winner-group"></div></li>
  <li class="index fl tc"><span>1</span></li>
  <li class="team-name fl tc ellipsis"><span>핸드크림 바른 쥐</span></li>
  <li class="team-member fl tc"><div class="teams fl">...</div></li>
  <li class="score fl tc"><span>0.81474</span></li>
  <li class="submission fl tc"><span>-</span></li>
  <li class="date fl tc"><span>4달 전</span></li>
</ul>
<ul class="record fl winner-group">
  <li class="rank fl tc"><div class="winner-group"></div></li>
  <li class="index fl tc"><span>2</span></li>
  <li class="team-name fl tc ellipsis"><span>케로로소대</span></li>
  <li class="team-member fl tc"><div class="teams fl">...</div></li>
  <li class="score fl tc"><span>0.81414</span></li>
  <li class="submission fl tc"><span>-</span></li>
  <li class="date fl tc"><span>4달 전</span></li>
</ul>
"""


class DaconCompetitionDocsTest(unittest.TestCase):
    # Real fragment shapes confirmed by inspection: DACON renders each
    # free-text section (overview, per-file data description) inside its own
    # rich-text editor container (class="ql-editor"), and duplicates it later
    # in the page via the client-hydration payload.
    _OVERVIEW_HTML_FRAGMENT = (
        '<script>var junk = "[배경] unrelated script text";</script>'
        '<div class="ql-editor"></div>'
        '<div class="ql-editor">[배경] 모기 비행 궤적 예측 대회 설명입니다. 80ms 뒤 위치를 예측합니다.</div>'
        '<div class="ql-editor">[배경] 모기 비행 궤적 예측 대회 설명입니다. 80ms 뒤 위치를 예측합니다.</div>'
    )
    _DATA_HTML_FRAGMENT = (
        '<div class="ql-editor"></div>'
        '<div class="ql-editor">train/ : timestep_ms, x, y, z 컬럼을 가진 CSV 파일들입니다.</div>'
    )

    def test_fetch_competition_overview_extracts_first_non_empty_editor_block(self):
        result = dacon_api.fetch_competition_overview(
            "236716",
            fetch_html_fn=lambda cid: {"ok": True, "html": self._OVERVIEW_HTML_FRAGMENT},
        )
        self.assertTrue(result["ok"])
        self.assertEqual("found", result["status"])
        self.assertIn("모기 비행 궤적", result["text"])
        self.assertNotIn("unrelated script text", result["text"])

    def test_fetch_competition_overview_reports_not_found_without_raising(self):
        result = dacon_api.fetch_competition_overview(
            "236716",
            fetch_html_fn=lambda cid: {"ok": True, "html": "<div class='ql-editor'></div>"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("not_found", result["status"])

    def test_fetch_competition_overview_reports_fetch_failure(self):
        result = dacon_api.fetch_competition_overview(
            "236716",
            fetch_html_fn=lambda cid: {"ok": False, "status": "fetch_error", "error": "timeout"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("fetch_error", result["status"])

    def test_fetch_competition_data_description_extracts_first_non_empty_editor_block(self):
        result = dacon_api.fetch_competition_data_description(
            "236716",
            fetch_html_fn=lambda cid: {"ok": True, "html": self._DATA_HTML_FRAGMENT},
        )
        self.assertTrue(result["ok"])
        self.assertIn("timestep_ms", result["text"])


class DaconLeaderboardTest(unittest.TestCase):
    def test_competition_id_from_link_extracts_numeric_id(self):
        self.assertEqual(
            "236689",
            dacon_api.competition_id_from_link(
                "https://dacon.io/competitions/official/236689/overview/description"
            ),
        )
        self.assertEqual(
            "236689",
            dacon_api.competition_id_from_link("https://dacon.io/competitions/official/236689/leaderboard"),
        )
        self.assertIsNone(dacon_api.competition_id_from_link("https://dacon.io/competitions/open/"))

    def test_leaderboard_url_uses_the_competition_id(self):
        self.assertEqual(
            "https://dacon.io/competitions/official/236689/leaderboard",
            dacon_api.leaderboard_url("236689"),
        )

    def test_submission_history_url_uses_the_competition_id(self):
        self.assertEqual(
            "https://dacon.io/competitions/official/236689/mysubmission",
            dacon_api.submission_history_url("236689"),
        )

    def test_parse_leaderboard_rows_reads_rank_team_and_score(self):
        rows = dacon_api._parse_leaderboard_rows(_REAL_LEADERBOARD_HTML_FRAGMENT)
        self.assertEqual(
            [
                {"rank": 1, "team_name": "핸드크림 바른 쥐", "score": 0.81474},
                {"rank": 2, "team_name": "케로로소대", "score": 0.81414},
            ],
            rows,
        )

    def test_fetch_leaderboard_finds_the_matching_team(self):
        result = dacon_api.fetch_leaderboard(
            "236689",
            team_name="케로로소대",
            fetch_html_fn=lambda cid: {"ok": True, "html": _REAL_LEADERBOARD_HTML_FRAGMENT},
            rows_fn=dacon_api._parse_leaderboard_rows,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("found", result["status"])
        self.assertEqual(2, result["rank"])
        self.assertEqual(0.81414, result["score"])

    def test_fetch_leaderboard_reports_team_not_found(self):
        result = dacon_api.fetch_leaderboard(
            "236689",
            team_name="없는팀",
            fetch_html_fn=lambda cid: {"ok": True, "html": _REAL_LEADERBOARD_HTML_FRAGMENT},
            rows_fn=dacon_api._parse_leaderboard_rows,
        )
        self.assertFalse(result["ok"])
        self.assertEqual("team_not_found", result["status"])

    def test_fetch_leaderboard_reports_fetch_failure(self):
        result = dacon_api.fetch_leaderboard(
            "236689",
            team_name="any",
            fetch_html_fn=lambda cid: {"ok": False, "status": "fetch_error", "error": "timeout"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("fetch_error", result["status"])


class DaconDailySubmissionLimitTest(unittest.TestCase):
    # Real fragment shape from https://dacon.io/competitions/official/236716/overview/rules
    # ("4. 유의 사항" section), confirmed present in server-rendered HTML with
    # no NUXT/JS evaluation needed, unlike the leaderboard page.
    _REAL_RULES_HTML_FRAGMENT = (
        '<h3>4. 유의 사항</h3><ul><li>1일 최대 제출 횟수: <strong> 5회</strong></li>'
        "<li>사용 가능 언어: Python</li></ul>"
    )

    def test_fetch_daily_submission_limit_parses_the_real_page_shape(self):
        result = dacon_api.fetch_daily_submission_limit(
            "236716",
            fetch_html_fn=lambda cid: {"ok": True, "html": self._REAL_RULES_HTML_FRAGMENT},
        )
        self.assertTrue(result["ok"])
        self.assertEqual("found", result["status"])
        self.assertEqual(5, result["daily_submission_limit"])

    def test_fetch_daily_submission_limit_reports_not_found_without_raising(self):
        # Competitions that phrase this differently or omit it entirely must
        # be reported as "unknown", not silently treated as "no limit" (that
        # distinction matters -- treating unknown as unlimited could burn a
        # real daily cap without warning).
        result = dacon_api.fetch_daily_submission_limit(
            "236716",
            fetch_html_fn=lambda cid: {"ok": True, "html": "<h3>4. 유의 사항</h3><ul><li>어떠한 언급도 없음</li></ul>"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("not_found", result["status"])

    def test_fetch_daily_submission_limit_reports_fetch_failure(self):
        result = dacon_api.fetch_daily_submission_limit(
            "236716",
            fetch_html_fn=lambda cid: {"ok": False, "status": "fetch_error", "error": "timeout"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("fetch_error", result["status"])


class DaconFullLeaderboardTest(unittest.TestCase):
    """Covers the "beyond the top 100" leaderboard path.

    DACON's leaderboard page only server-renders the top 100 rows as <li>
    HTML; a personal rank past that (e.g. 129th) is only reachable through
    the `window.__NUXT__` payload embedded in the same page, which the
    page's own "전체보기" button reveals client-side. `_full_leaderboard_rows_via_node`
    extracts that payload by evaluating it with Node (it isn't valid JSON --
    Nuxt's minifier dedupes strings into positional function arguments).
    """

    def test_full_leaderboard_rows_returns_none_when_node_is_unavailable(self):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("node not found")

        result = dacon_api._full_leaderboard_rows_via_node("<html></html>", run_fn=fake_run)
        self.assertIsNone(result)

    def test_full_leaderboard_rows_returns_none_on_nonzero_exit(self):
        class FakeCompletedProcess:
            returncode = 2
            stdout = ""
            stderr = "nuxt_payload_not_found"

        result = dacon_api._full_leaderboard_rows_via_node(
            "<html></html>", run_fn=lambda *a, **k: FakeCompletedProcess()
        )
        self.assertIsNone(result)

    def test_full_leaderboard_rows_parses_node_stdout(self):
        class FakeCompletedProcess:
            returncode = 0
            stdout = json.dumps(
                [
                    {"ranking": 1, "team_name": "SML", "score": 0.56758},
                    {"ranking": 129, "team_name": "뚜로", "score": 0.59242},
                ]
            )
            stderr = ""

        result = dacon_api._full_leaderboard_rows_via_node(
            "<html></html>", run_fn=lambda *a, **k: FakeCompletedProcess()
        )
        self.assertEqual(
            [
                {"rank": 1, "team_name": "SML", "score": 0.56758},
                {"rank": 129, "team_name": "뚜로", "score": 0.59242},
            ],
            result,
        )

    def test_leaderboard_rows_prefers_full_rows_when_available(self):
        full_rows = [{"rank": 129, "team_name": "뚜로", "score": 0.59242}]
        rows = dacon_api.leaderboard_rows("<html></html>", full_rows_fn=lambda html: full_rows)
        self.assertEqual(full_rows, rows)

    def test_leaderboard_rows_falls_back_to_html_parse_when_node_unavailable(self):
        rows = dacon_api.leaderboard_rows(
            _REAL_LEADERBOARD_HTML_FRAGMENT, full_rows_fn=lambda html: None
        )
        self.assertEqual(
            [
                {"rank": 1, "team_name": "핸드크림 바른 쥐", "score": 0.81474},
                {"rank": 2, "team_name": "케로로소대", "score": 0.81414},
            ],
            rows,
        )

    def test_fetch_leaderboard_finds_a_rank_beyond_the_top_100_via_full_rows(self):
        full_rows = [{"rank": 129, "team_name": "뚜로", "score": 0.59242}]
        result = dacon_api.fetch_leaderboard(
            "236690",
            team_name="뚜로",
            fetch_html_fn=lambda cid: {"ok": True, "html": "<html></html>"},
            rows_fn=lambda html: full_rows,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(129, result["rank"])
        self.assertEqual(0.59242, result["score"])


class DaconMySubmissionsTest(unittest.TestCase):
    """Covers the personal submission-history path.

    This never logs in on the user's behalf -- it only ever reads a session
    token (the "Token" request header value, a JWT) the user obtained by
    logging into DACON themselves and copying out of their own browser's
    devtools, the same trust boundary as DACON_PERSONAL_TOKEN for the submit
    path. Endpoint and response shape confirmed against a real logged-in
    session (competition_id=236690, team "뚜로", 43 real submissions).
    """

    def test_submission_list_api_url_uses_the_competition_id(self):
        self.assertEqual(
            "https://newapi.dacon.io/competition/submission_list?page=1&data_type=recent&cpt_id=236690",
            dacon_api.submission_list_api_url("236690"),
        )

    def test_check_session_auth_fails_without_token(self):
        result = dacon_api.check_session_auth(env={})
        self.assertFalse(result["ok"])
        self.assertEqual("session_cookie_missing", result["status"])

    def test_check_session_auth_passes_with_env_token(self):
        result = dacon_api.check_session_auth(env={dacon_api.SESSION_COOKIE_ENV_VAR: "eyJhbGciOi..."})
        self.assertTrue(result["ok"])

    def test_fetch_my_submissions_blocks_without_token(self):
        result = dacon_api.fetch_my_submissions("236690", env={})
        self.assertFalse(result["ok"])
        self.assertEqual("session_cookie_missing", result["status"])

    def test_fetch_my_submissions_passes_token_to_fetch_fn(self):
        calls = []

        def fake_fetch(competition_id, token):
            calls.append((competition_id, token))
            return {
                "ok": True,
                "data": {
                    "status": 1,
                    "message": 1,
                    "data": [
                        {
                            "sub_id": 1468640,
                            "cpt_id": 236690,
                            "team_name": "뚜로",
                            "score": 0.5911560701,
                            "private_score": 0,
                            "file_link": "3MLmodel_Attention-LSTM_Ensemble_Calibrated.csv",
                            "c_time": "2026-06-26 08:10:10",
                            "is_late": 0,
                            "final_use": 1,
                            "memo": "",
                        }
                    ],
                },
            }

        result = dacon_api.fetch_my_submissions(
            "236690",
            token="eyJhbGciOi...",
            fetch_json_fn=fake_fetch,
        )
        self.assertEqual([("236690", "eyJhbGciOi...")], calls)
        self.assertTrue(result["ok"])
        self.assertEqual("found", result["status"])
        self.assertEqual(1, len(result["submissions"]))
        self.assertEqual(0.5911560701, result["submissions"][0]["score"])
        self.assertEqual("2026-06-26 08:10:10", result["submissions"][0]["c_time"])

    def test_fetch_my_submissions_returns_empty_list_when_no_submissions_yet(self):
        # DACON's real response for a competition the account hasn't
        # submitted to: {"message": <code>, "data": "no data", "status": 1}.
        result = dacon_api.fetch_my_submissions(
            "236689",
            token="eyJhbGciOi...",
            fetch_json_fn=lambda cid, token: {"ok": True, "data": {"status": 1, "message": 7, "data": "no data"}},
        )
        self.assertTrue(result["ok"])
        self.assertEqual("found", result["status"])
        self.assertEqual([], result["submissions"])

    def test_fetch_my_submissions_reports_fetch_failure(self):
        result = dacon_api.fetch_my_submissions(
            "236690",
            token="eyJhbGciOi...",
            fetch_json_fn=lambda cid, token: {"ok": False, "status": "fetch_error", "error": "timeout"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("fetch_error", result["status"])

    def test_fetch_my_submissions_reports_invalid_token_response(self):
        result = dacon_api.fetch_my_submissions(
            "236690",
            token="expired-token",
            fetch_json_fn=lambda cid, token: {"ok": True, "data": {"status": 0}},
        )
        self.assertFalse(result["ok"])
        self.assertEqual("unrecognized_response", result["status"])


if __name__ == "__main__":
    unittest.main()
