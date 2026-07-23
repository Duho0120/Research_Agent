from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, Tk, Toplevel, filedialog, messagebox
from tkinter import Button, Frame, Label, StringVar
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from .execution_profile import load_execution_profile
from .cli_app import experiment_snapshot, loop_state_path
from .interface_contract import (
    get_experiment,
    get_pending_request,
    get_trial,
    list_experiments,
    list_pending_requests,
    preview_next_trial,
    respond_to_request,
    submit_human_insight,
    sync_state,
)
from .paths import project_root


APP_TITLE = "Autonomous ML Research Agent"

BG = "#191919"
PANEL = "#242424"
PANEL_SOFT = "#2b2b2b"
PANEL_DEEP = "#151515"
FG = "#f5f5f5"
FG_MUTED = "#b8b8b8"
FG_DIM = "#8f8f8f"
BLUE = "#2f6f95"
BLUE_HOVER = "#397fa8"
PURPLE = "#6b43a1"
PURPLE_HOVER = "#7851b0"
GOLD = "#c79309"
GOLD_HOVER = "#d2a112"
RED = "#873131"
RED_HOVER = "#9b3a3a"
GREEN = "#3b7a57"
BORDER = "#343434"

FONT = ("Malgun Gothic", 10)
FONT_BOLD = ("Malgun Gothic", 10, "bold")
FONT_TITLE = ("Malgun Gothic", 24, "bold")
FONT_SECTION = ("Malgun Gothic", 13, "bold")
FONT_MONO = ("Consolas", 10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent-app")
    parser.add_argument("--no-sync", action="store_true", help="Skip file-to-SQLite synchronization on startup.")
    args = parser.parse_args(argv)
    app = ResearchAgentLocalApp(sync_on_start=not args.no_sync)
    app.mainloop()
    return 0


class ResearchAgentLocalApp(Tk):
    def __init__(self, *, sync_on_start: bool = True) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x820")
        self.minsize(1080, 680)
        self.configure(bg=BG)

        self.sync_on_start = sync_on_start
        self.selected_competition: str | None = None
        self.selected_trial: str | None = None
        self.selected_artifact_path: str | None = None
        self.current_trial: dict[str, Any] = {}
        self.current_artifacts: list[dict[str, Any]] = []
        self.pending_requests: list[dict[str, Any]] = []
        self.active_tab = "experiment"
        self.auto_loop_process: subprocess.Popen[str] | None = None

        self._configure_style()
        self._build_layout()
        self.refresh_all(sync=sync_on_start)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Dark.Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=FG,
            borderwidth=0,
            rowheight=34,
            font=FONT,
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=PANEL_SOFT,
            foreground=FG,
            borderwidth=0,
            relief="flat",
            font=FONT_BOLD,
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", BLUE)],
            foreground=[("selected", "#ffffff")],
        )
        style.configure("Dark.Vertical.TScrollbar", background=PANEL_SOFT, troughcolor=BG, bordercolor=BG, arrowcolor=FG)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0, minsize=390)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        header = Frame(self, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=34, pady=(28, 22))
        header.columnconfigure(0, weight=1)
        Label(header, text=APP_TITLE, bg=BG, fg=FG, font=FONT_TITLE).grid(row=0, column=0, sticky="w")
        _button(header, "상태 새로고침", self.refresh_all_sync, BLUE).grid(row=0, column=1, padx=(10, 0))
        _button(header, "앱 종료", self.destroy, RED).grid(row=0, column=2, padx=(10, 0))

        sidebar = Frame(self, bg=BG)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(34, 14), pady=(0, 28))
        sidebar.rowconfigure(1, weight=1)
        sidebar.rowconfigure(4, weight=1)
        sidebar.columnconfigure(0, weight=1)

        Label(sidebar, text="실험 목록", bg=BG, fg=FG, font=FONT_SECTION).grid(row=0, column=0, sticky="w", pady=(0, 10))
        exp_card = _card(sidebar)
        exp_card.grid(row=1, column=0, sticky="nsew")
        exp_card.rowconfigure(0, weight=1)
        exp_card.columnconfigure(0, weight=1)
        self.experiment_tree = ttk.Treeview(
            exp_card,
            columns=("state", "best", "tokens"),
            show="tree headings",
            height=10,
            selectmode="browse",
            style="Dark.Treeview",
        )
        self.experiment_tree.heading("#0", text="competition")
        self.experiment_tree.heading("state", text="status")
        self.experiment_tree.heading("best", text="best")
        self.experiment_tree.heading("tokens", text="tokens")
        self.experiment_tree.column("#0", width=150, stretch=True)
        self.experiment_tree.column("state", width=115, stretch=False)
        self.experiment_tree.column("best", width=82, stretch=False)
        self.experiment_tree.column("tokens", width=72, stretch=False, anchor="e")
        self.experiment_tree.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.experiment_tree.bind("<<TreeviewSelect>>", self._on_experiment_selected)

        request_header = Frame(sidebar, bg=BG)
        request_header.grid(row=2, column=0, sticky="ew", pady=(22, 10))
        request_header.columnconfigure(0, weight=1)
        Label(request_header, text="사용자 요청", bg=BG, fg=FG, font=FONT_SECTION).grid(row=0, column=0, sticky="w")
        _button(request_header, "새로고침", self.refresh_pending_requests, GOLD, width=10).grid(row=0, column=1)

        pending_card = _card(sidebar)
        pending_card.grid(row=4, column=0, sticky="nsew")
        pending_card.rowconfigure(0, weight=1)
        pending_card.columnconfigure(0, weight=1)
        self.pending_tree = ttk.Treeview(
            pending_card,
            columns=("trial", "priority"),
            show="tree headings",
            height=7,
            selectmode="browse",
            style="Dark.Treeview",
        )
        self.pending_tree.heading("#0", text="request")
        self.pending_tree.heading("trial", text="trial")
        self.pending_tree.heading("priority", text="prio")
        self.pending_tree.column("#0", width=210, stretch=True)
        self.pending_tree.column("trial", width=84, stretch=False)
        self.pending_tree.column("priority", width=48, stretch=False, anchor="e")
        self.pending_tree.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.pending_tree.bind("<<TreeviewSelect>>", self._on_pending_selected)

        self.main_panel = _card(self)
        self.main_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 34), pady=(0, 28))
        self.main_panel.columnconfigure(0, weight=1)
        self.main_panel.rowconfigure(2, weight=1)

        self._build_tab_bar()
        self._build_content_frames()
        self._show_tab("experiment")

    def _build_tab_bar(self) -> None:
        self.tab_bar = Frame(self.main_panel, bg=PANEL)
        self.tab_bar.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 8))
        self.tab_buttons: dict[str, Button] = {}
        for key, label in [
            ("experiment", "실험"),
            ("trial", "Trial"),
            ("request", "요청/피드백"),
        ]:
            button = _button(self.tab_bar, label, lambda name=key: self._show_tab(name), PANEL_SOFT, width=13)
            button.pack(side=LEFT, padx=(0, 8))
            self.tab_buttons[key] = button

    def _build_content_frames(self) -> None:
        self.content_area = Frame(self.main_panel, bg=PANEL)
        self.content_area.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=22, pady=(0, 22))
        self.content_area.columnconfigure(0, weight=1)
        self.content_area.rowconfigure(0, weight=1)

        self.experiment_frame = Frame(self.content_area, bg=PANEL)
        self.trial_frame = Frame(self.content_area, bg=PANEL)
        self.request_frame = Frame(self.content_area, bg=PANEL)
        for frame in [self.experiment_frame, self.trial_frame, self.request_frame]:
            frame.grid(row=0, column=0, sticky="nsew")

        self._build_experiment_tab()
        self._build_trial_tab()
        self._build_request_tab()

    def _build_experiment_tab(self) -> None:
        frame = self.experiment_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        self.experiment_meta = Label(
            frame,
            text="실험을 선택하세요.",
            bg=PANEL,
            fg=FG_MUTED,
            font=FONT,
            anchor="w",
            justify=LEFT,
            wraplength=820,
        )
        self.experiment_meta.grid(row=0, column=0, sticky="ew", pady=(4, 14))

        actions = Frame(frame, bg=PANEL)
        actions.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        _button(actions, "다음 Trial 미리보기", self.preview_next, GOLD).pack(side=LEFT)
        _button(actions, "자동 실험 시작", self.start_auto_loop, BLUE).pack(side=LEFT, padx=(10, 0))
        _button(actions, "중단 요청", self.request_auto_loop_pause, RED).pack(side=LEFT, padx=(10, 0))
        _button(actions, "재개", self.resume_auto_loop, GREEN).pack(side=LEFT, padx=(10, 0))
        _button(actions, "사용자 인사이트 추가", self.open_insight_dialog, PURPLE).pack(side=LEFT, padx=(10, 0))
        self.loop_status = Label(frame, text=self._loop_status_text(), bg=PANEL, fg=FG_MUTED, font=FONT, anchor="w", justify=LEFT)
        self.loop_status.grid(row=4, column=0, sticky="ew", pady=(12, 0))

        trial_card = _card(frame, background=PANEL_DEEP)
        trial_card.grid(row=2, column=0, sticky="nsew")
        trial_card.columnconfigure(0, weight=1)
        trial_card.rowconfigure(0, weight=1)
        self.trial_tree = ttk.Treeview(
            trial_card,
            columns=("best", "status", "score", "axis", "decision"),
            show="tree headings",
            selectmode="browse",
            style="Dark.Treeview",
        )
        self.trial_tree.heading("#0", text="trial")
        self.trial_tree.heading("best", text="best")
        self.trial_tree.heading("status", text="status")
        self.trial_tree.heading("score", text="score")
        self.trial_tree.heading("axis", text="axis")
        self.trial_tree.heading("decision", text="decision")
        self.trial_tree.column("#0", width=110, stretch=False)
        self.trial_tree.column("best", width=72, stretch=False, anchor="center")
        self.trial_tree.column("status", width=105, stretch=False)
        self.trial_tree.column("score", width=95, stretch=False, anchor="e")
        self.trial_tree.column("axis", width=260, stretch=True)
        self.trial_tree.column("decision", width=150, stretch=True)
        self.trial_tree.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.trial_tree.bind("<<TreeviewSelect>>", self._on_trial_selected)

        self.preview_text = _text_box(frame, height=7)
        self.preview_text.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self._set_text(self.preview_text, "다음 trial 미리보기 결과가 여기에 표시됩니다.")

    def _build_trial_tab(self) -> None:
        frame = self.trial_frame
        frame.columnconfigure(0, weight=0, minsize=300)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

        self.trial_meta = Label(
            frame,
            text="Trial을 선택하세요.",
            bg=PANEL,
            fg=FG_MUTED,
            font=FONT,
            anchor="w",
            justify=LEFT,
            wraplength=840,
        )
        self.trial_meta.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(4, 14))

        Label(frame, text="사용자용 산출물", bg=PANEL, fg=FG, font=FONT_SECTION).grid(row=1, column=0, sticky="w", pady=(0, 8))
        Label(frame, text="읽기 화면", bg=PANEL, fg=FG, font=FONT_SECTION).grid(row=1, column=1, sticky="w", pady=(0, 8))

        artifact_card = _card(frame, background=PANEL_DEEP)
        artifact_card.grid(row=2, column=0, sticky="nsew", padx=(0, 14))
        artifact_card.columnconfigure(0, weight=1)
        artifact_card.rowconfigure(0, weight=1)
        self.artifact_list = ttk.Treeview(artifact_card, columns=("path",), show="tree", selectmode="browse", style="Dark.Treeview")
        self.artifact_list.column("#0", width=280, stretch=True)
        self.artifact_list.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.artifact_list.bind("<<TreeviewSelect>>", self._on_artifact_selected)

        self.artifact_text = _text_box(frame)
        self.artifact_text.grid(row=2, column=1, sticky="nsew")

        artifact_actions = Frame(frame, bg=PANEL)
        artifact_actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        _button(artifact_actions, "파일 열기", self.open_selected_artifact, GOLD).pack(side=LEFT)
        _button(artifact_actions, "폴더 선택", self.open_folder_dialog, BLUE).pack(side=LEFT, padx=(10, 0))

    def _build_request_tab(self) -> None:
        frame = self.request_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        self.request_meta = Label(
            frame,
            text="대기 요청을 선택하세요.",
            bg=PANEL,
            fg=FG_MUTED,
            font=FONT,
            anchor="w",
            justify=LEFT,
            wraplength=840,
        )
        self.request_meta.grid(row=0, column=0, sticky="ew", pady=(4, 12))

        self.request_text = _text_box(frame)
        self.request_text.grid(row=1, column=0, sticky="nsew", pady=(0, 14))

        form = _card(frame, background=PANEL_DEEP)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        Label(form, text="답변", bg=PANEL_DEEP, fg=FG, font=FONT_BOLD).grid(row=0, column=0, sticky="nw", padx=(14, 10), pady=14)
        self.response_text = _text_box(form, height=5)
        self.response_text.grid(row=0, column=1, sticky="ew", pady=14)
        _button(form, "응답 기록", self.submit_request_response, PURPLE).grid(row=0, column=2, sticky="n", padx=14, pady=14)

    def _show_tab(self, tab: str) -> None:
        self.active_tab = tab
        frames = {
            "experiment": self.experiment_frame,
            "trial": self.trial_frame,
            "request": self.request_frame,
        }
        frames[tab].tkraise()
        for key, button in self.tab_buttons.items():
            button.configure(bg=BLUE if key == tab else PANEL_SOFT, activebackground=BLUE_HOVER if key == tab else BORDER)

    def refresh_all_sync(self) -> None:
        self.refresh_all(sync=True)

    def refresh_all(self, *, sync: bool) -> None:
        if sync:
            self._safe_call(lambda: sync_state())
        result = self._safe_call(lambda: list_experiments(sync=False))
        if not result:
            return
        self._load_experiment_tree(result.get("data", {}).get("experiments", []))
        self.refresh_pending_requests()

    def refresh_pending_requests(self) -> None:
        result = self._safe_call(lambda: list_pending_requests(sync=False))
        if not result:
            return
        self.pending_requests = list(result.get("data", {}).get("requests", []))
        for item in self.pending_tree.get_children():
            self.pending_tree.delete(item)
        for request in self.pending_requests:
            label = request.get("title") or request.get("type") or request.get("request_id")
            self.pending_tree.insert(
                "",
                END,
                iid=str(request.get("request_id")),
                text=str(label),
                values=(request.get("trial_id") or "-", request.get("priority") or 0),
            )

    def _load_experiment_tree(self, experiments: list[dict[str, Any]]) -> None:
        for item in self.experiment_tree.get_children():
            self.experiment_tree.delete(item)
        for experiment in experiments:
            best = experiment.get("best_trial") or {}
            best_label = best.get("trial_id") or "-"
            self.experiment_tree.insert(
                "",
                END,
                iid=str(experiment.get("competition")),
                text=str(experiment.get("competition")),
                values=(experiment.get("state") or "-", best_label, experiment.get("total_tokens") or 0),
            )

    def _on_experiment_selected(self, _event: Any) -> None:
        selected = self.experiment_tree.selection()
        if not selected:
            return
        self.selected_competition = str(selected[0])
        result = self._safe_call(lambda: get_experiment(self.selected_competition, sync=False))
        if not result or not result.get("ok"):
            return
        data = result.get("data", {})
        experiment = data.get("experiment") or {}
        axis = experiment.get("active_axis") or {}
        best = experiment.get("best_trial") or {}
        self.experiment_meta.configure(
            text=(
                f"{experiment.get('competition')}  |  {experiment.get('topic') or '-'}\n"
                f"state={experiment.get('state') or '-'}  |  metric={experiment.get('metric') or '-'}  |  "
                f"best={best.get('trial_id') or '-'} / {_score_text(best.get('score'))}\n"
                f"active_axis={axis.get('name') or '-'} ({axis.get('attempt_count') or 0}/{axis.get('attempt_limit') or 0})"
            )
        )
        self._load_trial_tree(data.get("trials", []))
        self._show_tab("experiment")

    def _load_trial_tree(self, trials: list[dict[str, Any]]) -> None:
        for item in self.trial_tree.get_children():
            self.trial_tree.delete(item)
        for trial in trials:
            trial_id = str(trial.get("trial_id"))
            self.trial_tree.insert(
                "",
                END,
                iid=trial_id,
                text=trial_id,
                values=(
                    _best_badge(trial),
                    trial.get("status") or "-",
                    _score_text(trial.get("local_score")),
                    trial.get("change_axis") or "-",
                    trial.get("decision") or "-",
                ),
            )

    def _on_trial_selected(self, _event: Any) -> None:
        selected = self.trial_tree.selection()
        if not selected or not self.selected_competition:
            return
        self.selected_trial = str(selected[0])
        result = self._safe_call(lambda: get_trial(self.selected_competition, self.selected_trial, sync=False))
        if not result or not result.get("ok"):
            return
        trial = result.get("data", {}).get("trial") or {}
        self.current_trial = trial
        self.trial_meta.configure(
            text=(
                f"{trial.get('competition')} / {trial.get('trial_id')}\n"
                f"status={trial.get('status')}  |  score={_score_text(trial.get('local_score'))}  |  "
                f"tokens={trial.get('token_total') or 0}  |  decision={trial.get('decision') or '-'}"
            )
        )
        self._load_artifacts(trial.get("user_artifacts", []))
        self._show_trial_overview()
        self._show_tab("trial")

    def _load_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        for item in self.artifact_list.get_children():
            self.artifact_list.delete(item)
        self.selected_artifact_path = None
        self.current_artifacts = _visible_artifacts(list(artifacts))
        self.artifact_list.insert("", END, iid="__overview__", text="0. 핵심 요약", values=("",))
        for index, artifact in enumerate(_ordered_artifacts(self.current_artifacts), start=1):
            item_id = f"artifact_{index}"
            label = _artifact_list_label(index, artifact)
            self.artifact_list.insert("", END, iid=item_id, text=str(label), values=(artifact.get("path") or "",))

    def _on_artifact_selected(self, _event: Any) -> None:
        selected = self.artifact_list.selection()
        if not selected:
            return
        if str(selected[0]) == "__overview__":
            self.selected_artifact_path = None
            self._show_trial_overview()
            return
        values = self.artifact_list.item(selected[0], "values")
        self.selected_artifact_path = str(values[0]) if values else None
        self._preview_artifact(self.selected_artifact_path)

    def _show_trial_overview(self) -> None:
        self._set_text(self.artifact_text, _render_trial_overview(self.current_trial, self.current_artifacts))

    def _preview_artifact(self, raw_path: str | None) -> None:
        path = _resolve_path(raw_path)
        if not path:
            self._set_text(self.artifact_text, "파일 경로가 없습니다.")
            return
        if path.is_dir():
            files = sorted(item.name for item in path.iterdir())
            self._set_text(self.artifact_text, _render_directory_preview(path, files))
            return
        if not path.exists():
            self._set_text(self.artifact_text, f"파일을 찾을 수 없습니다.\n{path}")
            return
        if path.suffix.lower() not in {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".py", ".csv"}:
            self._set_text(self.artifact_text, f"미리보기를 지원하지 않는 파일입니다.\n{path}")
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 30000:
            text = text[:30000] + "\n\n... truncated ..."
        if path.suffix.lower() in {".md", ".txt"}:
            text = _format_markdown_for_app(text)
        self._set_text(self.artifact_text, text)

    def preview_next(self) -> None:
        if not self.selected_competition:
            messagebox.showinfo(APP_TITLE, "먼저 실험을 선택하세요.")
            return
        result = self._safe_call(lambda: preview_next_trial(self.selected_competition, sync=False))
        if result:
            self._set_text(self.preview_text, _pretty_result(result))

    def start_auto_loop(self) -> None:
        if self.auto_loop_process and self.auto_loop_process.poll() is None:
            messagebox.showinfo(APP_TITLE, "자동 실험이 이미 실행 중입니다.")
            return
        state = self._read_loop_state()
        if state.get("next_trial") and state.get("status") in {"paused", "completed", "failed", "running"}:
            self._launch_auto_loop(["--resume"])
            return
        self._launch_auto_loop([])

    def resume_auto_loop(self) -> None:
        if self.auto_loop_process and self.auto_loop_process.poll() is None:
            messagebox.showinfo(APP_TITLE, "자동 실험이 이미 실행 중입니다.")
            return
        self._launch_auto_loop(["--resume"])

    def request_auto_loop_pause(self) -> None:
        command = self._auto_loop_command(["--request-pause"])
        try:
            completed = subprocess.run(command, cwd=project_root(), text=True, capture_output=True, timeout=30)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        if completed.returncode != 0:
            messagebox.showwarning(APP_TITLE, completed.stderr or completed.stdout or "중단 요청 실패")
            return
        messagebox.showinfo(APP_TITLE, "현재 trial 완료와 다음 trial 계획 생성 후 자동 실험이 멈춥니다.")
        self._refresh_loop_status_label()

    def _launch_auto_loop(self, args: list[str]) -> None:
        command = self._auto_loop_command(args)
        try:
            self.auto_loop_process = subprocess.Popen(command, cwd=project_root(), text=True)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._refresh_loop_status_label()
        self.after(2000, self._poll_auto_loop)

    def _poll_auto_loop(self) -> None:
        self._refresh_loop_status_label()
        if self.auto_loop_process and self.auto_loop_process.poll() is None:
            self.after(3000, self._poll_auto_loop)
            return
        self.auto_loop_process = None
        self.refresh_all(sync=True)

    def _auto_loop_command(self, args: list[str]) -> list[str]:
        competition = self.selected_competition or "titanic"
        command = [
            sys.executable,
            str(project_root() / "scripts" / "generic_workspace_auto_loop.py"),
            "--competition",
            competition,
        ]
        if "--request-pause" in args:
            return [*command, "--request-pause"]

        snapshot = experiment_snapshot(competition, sync=False)
        start_trial = snapshot.get("next_trial") or "trial_001"
        command.extend(
            [
                "--start-trial",
                str(start_trial),
                "--max-trials",
                "999",
                "--code-writer",
                "--allow-api",
            ]
        )
        try:
            profile = load_execution_profile(competition)
        except (FileNotFoundError, ValueError):
            profile = {}
        if str(profile.get("platform") or "").casefold() == "kaggle":
            command.extend(["--submit", "--kaggle-slug", competition])
        return command

    def _refresh_loop_status_label(self) -> None:
        if hasattr(self, "loop_status"):
            self.loop_status.configure(text=self._loop_status_text())

    def _loop_status_text(self) -> str:
        state = self._read_loop_state()
        running = self.auto_loop_process is not None and self.auto_loop_process.poll() is None
        if not state:
            return "자동 실험 상태: 기록 없음" + (" (프로세스 실행 중)" if running else "")
        parts = [
            f"자동 실험 상태: {state.get('status') or '-'}",
            f"현재 trial={state.get('current_trial') or '-'}",
            f"마지막 완료={state.get('last_completed_trial') or '-'}",
            f"다음={state.get('next_trial') or '-'}",
        ]
        if state.get("pause_requested"):
            parts.append("중단 요청됨")
        if running:
            parts.append("프로세스 실행 중")
        return "  |  ".join(parts)

    def _read_loop_state(self) -> dict[str, Any]:
        state_path = loop_state_path()
        if not state_path.exists():
            return {}
        try:
            value = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "상태 파일을 읽을 수 없음"}
        return value if isinstance(value, dict) else {}

    def _on_pending_selected(self, _event: Any) -> None:
        selected = self.pending_tree.selection()
        if not selected:
            return
        request_id = str(selected[0])
        result = self._safe_call(lambda: get_pending_request(request_id))
        if not result or not result.get("ok"):
            return
        request = result.get("data", {}).get("request") or {}
        self.request_meta.configure(
            text=f"{request.get('competition')} / {request.get('trial_id') or '-'}  |  {request.get('title') or request_id}"
        )
        self._set_text(self.request_text, _format_pending_request(request))
        self.response_text.delete("1.0", END)
        self._show_tab("request")

    def submit_request_response(self) -> None:
        selected = self.pending_tree.selection()
        if not selected:
            messagebox.showinfo(APP_TITLE, "응답할 요청을 선택하세요.")
            return
        free_text = self.response_text.get("1.0", END).strip()
        if not free_text:
            messagebox.showinfo(APP_TITLE, "응답 내용을 입력하세요.")
            return
        result = self._safe_call(lambda: respond_to_request(str(selected[0]), free_text=free_text))
        if result and result.get("ok"):
            messagebox.showinfo(APP_TITLE, "응답이 기록되었습니다.")
            self.refresh_pending_requests()

    def open_insight_dialog(self) -> None:
        if not self.selected_competition or not self.selected_trial:
            messagebox.showinfo(APP_TITLE, "먼저 실험과 trial을 선택하세요.")
            return
        dialog = InsightDialog(self, self.selected_competition, self.selected_trial)
        self.wait_window(dialog.window)
        if dialog.submitted:
            self.refresh_all(sync=True)

    def open_selected_artifact(self) -> None:
        path = _resolve_path(self.selected_artifact_path)
        if not path or not path.exists():
            messagebox.showinfo(APP_TITLE, "열 수 있는 파일이 없습니다.")
            return
        os.startfile(path)  # type: ignore[attr-defined]

    def open_folder_dialog(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(project_root()))
        if folder:
            os.startfile(folder)  # type: ignore[attr-defined]

    def _safe_call(self, callback: Any) -> dict[str, Any] | None:
        try:
            result = callback()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return None
        if isinstance(result, dict) and not result.get("ok", True):
            message = result.get("message") or "Operation failed."
            errors = result.get("errors") or []
            if errors:
                message += "\n\n" + "\n".join(str(item.get("message") or item) for item in errors)
            messagebox.showwarning(APP_TITLE, message)
        return result

    @staticmethod
    def _set_text(widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")


class InsightDialog(ttk.Frame):
    def __init__(self, parent: ResearchAgentLocalApp, competition: str, trial_id: str) -> None:
        self.window = Toplevel(parent)
        self.window.title("사용자 인사이트 추가")
        self.window.geometry("640x360")
        self.window.configure(bg=BG)
        self.window.transient(parent)
        self.window.grab_set()
        super().__init__(self.window, padding=18)
        self.configure(style="Dialog.TFrame")
        self.pack(fill=BOTH, expand=True)
        self.parent_app = parent
        self.competition = competition
        self.trial_id = trial_id
        self.submitted = False
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        Label(self, text=f"{self.competition} / {self.trial_id}", bg=BG, fg=FG_MUTED, font=FONT).grid(row=0, column=0, sticky="ew")
        Label(self, text="다음 실험에서 참고할 의견을 입력하세요.", bg=BG, fg=FG, font=FONT_SECTION).grid(
            row=1, column=0, sticky="w", pady=(10, 8)
        )
        options = Frame(self, bg=BG)
        options.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        Label(options, text="적용 범위", bg=BG, fg=FG_MUTED, font=FONT).pack(side=LEFT)
        self.scope_var = StringVar(value="next_trial")
        ttk.Combobox(
            options,
            textvariable=self.scope_var,
            values=("next_trial", "competition"),
            state="readonly",
            width=16,
        ).pack(side=LEFT, padx=(8, 18))
        Label(options, text="중요도", bg=BG, fg=FG_MUTED, font=FONT).pack(side=LEFT)
        self.priority_var = StringVar(value="normal")
        ttk.Combobox(
            options,
            textvariable=self.priority_var,
            values=("normal", "high"),
            state="readonly",
            width=12,
        ).pack(side=LEFT, padx=(8, 0))
        Label(
            self,
            text="개선축은 에이전트가 다음 계획 단계에서 해석합니다.",
            bg=BG,
            fg=FG_MUTED,
            font=FONT,
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))
        self.text = _text_box(self, height=10)
        self.text.grid(row=4, column=0, sticky="nsew")
        buttons = Frame(self, bg=BG)
        buttons.grid(row=5, column=0, sticky="e", pady=(14, 0))
        _button(buttons, "취소", self.window.destroy, RED).pack(side=RIGHT)
        _button(buttons, "저장", self._submit, BLUE).pack(side=RIGHT, padx=(0, 10))

    def _submit(self) -> None:
        insight = self.text.get("1.0", END).strip()
        if not insight:
            messagebox.showinfo(APP_TITLE, "인사이트 내용을 입력하세요.")
            return
        result = self.parent_app._safe_call(
            lambda: submit_human_insight(
                self.competition,
                self.trial_id,
                insight=insight,
                scope=self.scope_var.get(),
                priority=self.priority_var.get(),
            )
        )
        if result and result.get("ok"):
            self.submitted = True
            self.window.destroy()


def _button(parent: Any, text: str, command: Any, color: str, *, width: int = 16) -> Button:
    hover = {
        BLUE: BLUE_HOVER,
        PURPLE: PURPLE_HOVER,
        GOLD: GOLD_HOVER,
        RED: RED_HOVER,
        PANEL_SOFT: BORDER,
    }.get(color, color)
    button = Button(
        parent,
        text=text,
        command=command,
        bg=color,
        fg=FG,
        activebackground=hover,
        activeforeground=FG,
        relief="flat",
        borderwidth=0,
        highlightthickness=0,
        padx=14,
        pady=9,
        width=width,
        cursor="hand2",
        font=FONT_BOLD,
    )
    button.bind("<Enter>", lambda _event: button.configure(bg=hover))
    button.bind("<Leave>", lambda _event: button.configure(bg=color))
    return button


def _card(parent: Any, *, background: str = PANEL) -> Frame:
    return Frame(parent, bg=background, highlightbackground=BORDER, highlightthickness=1, bd=0)


def _text_box(parent: Any, *, height: int | None = None) -> ScrolledText:
    kwargs: dict[str, Any] = {
        "wrap": "word",
        "background": PANEL_DEEP,
        "foreground": FG,
        "insertbackground": FG,
        "selectbackground": BLUE,
        "selectforeground": FG,
        "relief": "flat",
        "borderwidth": 0,
        "font": FONT_MONO,
        "padx": 14,
        "pady": 12,
    }
    if height is not None:
        kwargs["height"] = height
    return ScrolledText(parent, **kwargs)


ARTIFACT_ORDER = {
    "summary_card_ko": 1,
    "plan_ko": 2,
    "pipeline_structure_ko": 3,
    "result_ko": 4,
    "decision_ko": 5,
    "submission_ko": 6,
    "code_pipeline_ko": 7,
    "user_code": 8,
    "readme_ko": 9,
    "paths_ko": 10,
}


ARTIFACT_PURPOSE = {
    "summary_card_ko": "전체 상황을 30초 안에 확인",
    "plan_ko": "이번 trial에서 무엇을 검증했는지",
    "pipeline_structure_ko": "데이터 로드부터 추론까지 구조",
    "result_ko": "로컬 실행 결과와 점수",
    "decision_ko": "다음 trial 판단 근거",
    "submission_ko": "제출 파일과 LB 기록",
    "code_pipeline_ko": "작성/수정된 코드 구성",
    "user_code": "실제 코드 파일",
    "readme_ko": "사용자 안내",
    "paths_ko": "파일 경로 안내",
}


ARTIFACT_ORDER.clear()
ARTIFACT_ORDER.update(
    {
        "plan_ko": 1,
        "pipeline_structure_ko": 2,
        "scores_ko": 3,
        "result_ko": 3,
        "submission_ko": 3,
    }
)
ARTIFACT_PURPOSE.clear()
ARTIFACT_PURPOSE.update(
    {
        "plan_ko": "실험 계획",
        "pipeline_structure_ko": "파이프라인 구조",
        "scores_ko": "로컬/제출 점수",
        "result_ko": "로컬/제출 점수",
        "submission_ko": "로컬/제출 점수",
    }
)


def _best_badge(trial: dict[str, Any]) -> str:
    local_best = bool(trial.get("is_best_local"))
    lb_best = bool(trial.get("is_best_lb"))
    if local_best and lb_best:
        return "BEST"
    if lb_best:
        return "LB"
    if local_best:
        return "LOCAL"
    return ""


def _visible_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"plan_ko", "pipeline_structure_ko", "scores_ko", "result_ko", "submission_ko"}
    seen_score = False
    visible: list[dict[str, Any]] = []
    for artifact in _ordered_artifacts(artifacts):
        artifact_type = str(artifact.get("type") or "")
        if artifact_type not in allowed:
            continue
        if artifact_type in {"scores_ko", "result_ko", "submission_ko"}:
            if seen_score:
                continue
            seen_score = True
        visible.append(artifact)
    return visible


def _ordered_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        artifacts,
        key=lambda item: (ARTIFACT_ORDER.get(str(item.get("type")), 99), str(item.get("label") or item.get("type"))),
    )


def _artifact_list_label(index: int, artifact: dict[str, Any]) -> str:
    artifact_type = str(artifact.get("type") or "")
    label = str(artifact.get("label") or artifact_type or "artifact")
    purpose = ARTIFACT_PURPOSE.get(artifact_type)
    if purpose:
        return f"{index}. {label} - {purpose}"
    return f"{index}. {label}"


def _render_trial_overview(trial: dict[str, Any], artifacts: list[dict[str, Any]]) -> str:
    if not trial:
        return "Trial을 선택하면 핵심 요약이 표시됩니다."
    axis = trial.get("change_axis") or trial.get("active_axis") or "-"
    best_flags = []
    if trial.get("is_best_local"):
        best_flags.append("local best")
    if trial.get("is_best_lb"):
        best_flags.append("LB best")
    lines = [
        "핵심 요약",
        "=" * 48,
        "",
        f"실험: {trial.get('competition')} / {trial.get('trial_id')}",
        f"상태: {trial.get('status') or '-'}",
        f"점수: {_score_text(trial.get('local_score'))} ({trial.get('metric') or '-'})",
        f"개선축: {axis}",
        f"판단: {trial.get('decision') or '-'}",
        f"참고 기준 trial: {trial.get('recommended_base_trial') or '-'}",
        f"토큰 사용량: {trial.get('token_total') or 0}",
    ]
    if best_flags:
        lines.append(f"Best 표시: {', '.join(best_flags)}")
    lines.extend(
        [
            "",
            "추천 읽기 순서",
            "-" * 48,
            "1. 실험 계획서: 이번 trial의 목적과 바꾼 개선축을 확인",
            "2. 파이프라인 구조도: 실제로 데이터/전처리/모델/추론이 어떻게 흐르는지 확인",
            "3. 실행 결과: 로컬 점수와 실행 상태 확인",
            "4. 판단 카드: 다음 trial에서 유지/기각/보류할 내용 확인",
            "",
            "산출물 목록",
            "-" * 48,
        ]
    )
    for index, artifact in enumerate(_ordered_artifacts(artifacts), start=1):
        label = artifact.get("label") or artifact.get("type")
        purpose = ARTIFACT_PURPOSE.get(str(artifact.get("type")), "추가 참고 자료")
        lines.append(f"{index}. {label}: {purpose}")
    lines.extend(
        [
            "",
            "왼쪽 산출물을 누르면 Markdown 문서를 앱에서 읽기 쉬운 형태로 정리해서 보여줍니다.",
            "원문 그대로 확인하고 싶으면 아래의 '파일 열기' 버튼을 사용하세요.",
        ]
    )
    return "\n".join(lines)


def _render_trial_overview(trial: dict[str, Any], artifacts: list[dict[str, Any]]) -> str:
    if not trial:
        return "Trial을 선택하면 핵심 요약이 표시됩니다."
    axis = trial.get("change_axis") or trial.get("active_axis") or "-"
    best = _best_badge(trial) or "-"
    lines = [
        "핵심 요약",
        "=" * 48,
        "",
        f"실험: {trial.get('competition')} / {trial.get('trial_id')}",
        f"상태: {trial.get('status') or '-'}",
        f"Best: {best}",
        f"로컬 점수: {_score_text(trial.get('local_score'))} ({trial.get('metric') or '-'})",
        f"제출 점수: {_score_text(trial.get('lb_score'))}",
        f"개선축: {axis}",
        "",
        "사용자용 산출물",
        "-" * 48,
    ]
    for index, artifact in enumerate(_ordered_artifacts(artifacts), start=1):
        label = artifact.get("label") or artifact.get("type")
        purpose = ARTIFACT_PURPOSE.get(str(artifact.get("type")), "")
        suffix = f" - {purpose}" if purpose else ""
        lines.append(f"{index}. {label}{suffix}")
    if not artifacts:
        lines.append("- 표시할 사용자용 산출물이 없습니다.")
    return "\n".join(lines)


def _render_directory_preview(path: Path, files: list[str]) -> str:
    lines = [
        "코드 폴더",
        "=" * 48,
        "",
        f"경로: {path}",
        "",
        "파일 목록",
        "-" * 48,
    ]
    if not files:
        lines.append("빈 폴더입니다.")
    else:
        lines.extend(f"- {item}" for item in files)
    lines.extend(["", "코드 내용을 보려면 '파일 열기' 버튼으로 폴더를 열어 확인하세요."])
    return "\n".join(lines)


def _format_markdown_for_app(text: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in text.replace("\ufeff", "").splitlines():
        line = raw_line.strip()
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        if _is_markdown_table_separator(line):
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            lines.extend([title, "=" * min(72, max(24, len(title) * 2)), ""])
            previous_blank = True
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            lines.extend(["", title, "-" * min(72, max(20, len(title) * 2))])
            continue
        if line.startswith("### "):
            lines.extend(["", f"[{line[4:].strip()}]"])
            continue
        if line.startswith("#### "):
            lines.append(f"- {line[5:].strip()}")
            continue
        if line.startswith("|") and line.endswith("|"):
            converted = _format_markdown_table_row(line)
            if converted:
                lines.append(converted)
            continue
        if line.startswith("- "):
            lines.append("• " + line[2:].strip())
            continue
        if line.startswith("* "):
            lines.append("• " + line[2:].strip())
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _is_markdown_table_separator(line: str) -> bool:
    if not line.startswith("|") or not line.endswith("|"):
        return False
    body = line.strip("|").strip()
    return bool(body) and all(char in {"|", "-", ":", " "} for char in body)


def _format_markdown_table_row(line: str) -> str:
    cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if not cells:
        return ""
    if len(cells) == 1:
        return f"• {cells[0]}"
    return f"• {cells[0]}: {' / '.join(cells[1:])}"


def _resolve_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root() / path
    return path


def _score_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _pretty_result(result: dict[str, Any]) -> str:
    data = result.get("data") or {}
    lines = [
        f"action: {result.get('action')}",
        f"status: {result.get('status')}",
        f"message: {result.get('message')}",
        "",
        "data:",
    ]
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    next_actions = result.get("next_actions") or []
    if next_actions:
        lines.extend(["", "next_actions:"])
        lines.extend(f"- {item.get('label') or item.get('action')}" for item in next_actions)
    return "\n".join(lines)


def _format_pending_request(request: dict[str, Any]) -> str:
    lines = [
        f"# {request.get('title') or request.get('request_id')}",
        "",
        f"- competition: {request.get('competition')}",
        f"- trial_id: {request.get('trial_id') or '-'}",
        f"- type: {request.get('type')}",
        f"- priority: {request.get('priority')}",
        "",
        "## Message",
        "",
        str(request.get("message") or "-"),
        "",
        "## Question",
        "",
        str(request.get("question") or "-"),
    ]
    context_files = request.get("context_files") or []
    if context_files:
        lines.extend(["", "## Context Files", ""])
        for item in context_files:
            lines.append(f"- {item.get('label') or 'file'}: {item.get('path')}")
    questions = request.get("questions") or []
    if questions:
        lines.extend(["", "## Structured Questions", ""])
        for item in questions:
            lines.append(f"- {item.get('id')}: {item.get('label')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
