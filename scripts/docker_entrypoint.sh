#!/bin/sh
set -eu

STORAGE_DIR="${RESEARCH_AGENT_STORAGE_DIR:-/mnt/research-agent}"

mkdir -p "$STORAGE_DIR/demo_workspaces" \
  "$STORAGE_DIR/competitions" \
  "$STORAGE_DIR/experiments" \
  "$STORAGE_DIR/memory" \
  "$STORAGE_DIR/submissions" \
  "$STORAGE_DIR/jobs" \
  "$STORAGE_DIR/runs" \
  "$STORAGE_DIR/runtime"

link_runtime_dir() {
  name="$1"
  target="$STORAGE_DIR/$name"
  link="/app/$name"
  if [ -L "$link" ]; then
    return
  fi
  if [ -e "$link" ]; then
    backup="/app/${name}.image"
    if [ ! -e "$backup" ]; then
      mv "$link" "$backup"
    else
      rm -rf "$link"
    fi
  fi
  ln -s "$target" "$link"
}

link_runtime_dir demo_workspaces
link_runtime_dir competitions
link_runtime_dir experiments
link_runtime_dir memory
link_runtime_dir submissions
link_runtime_dir jobs
link_runtime_dir runs

if [ "${RESEARCH_AGENT_LOAD_DEMO_SEED:-1}" != "0" ] && [ -d "/app/demo_seed" ]; then
  python -B /app/scripts/initialize_demo_seed.py \
    --seed-root /app/demo_seed \
    --storage-dir "$STORAGE_DIR"
fi

exec "$@"
