#!/usr/bin/env bash
# Static checks + optional smoke steps for the wav2vec unsupervised pipeline.
# Run from repo root: bash scripts/validate_pipeline.sh
#
# Environment:
#   RUN_SMOKE_TB=1       Run scripts/smoke_tb_w2vu.sh (needs GPU, prepared data/, venv)
#   RUN_SMOKE_TRAIN=1    Run scripts/smoke_w2vu_train.sh (same)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Repo: $ROOT"

fail=0
err() { echo "FAIL: $*" >&2; fail=1; }

# --- Shell syntax ---
for s in \
  utils.sh \
  wav2vec_functions.sh \
  gans_functions.sh \
  eval_functions.sh \
  run_wav2vec.sh \
  run_gans.sh \
  run_eval.sh \
  run_setup.sh \
  scripts/smoke_tb_w2vu.sh \
  scripts/smoke_w2vu_train.sh
do
  if [[ -f "$ROOT/$s" ]]; then
    bash -n "$ROOT/$s" || err "bash -n $s"
  else
    err "missing $s"
  fi
done

# --- Source utils (must resolve DIR_PATH to this repo) ---
# shellcheck source=/dev/null
source "$ROOT/utils.sh"
[[ "$DIR_PATH" == "$ROOT" ]] || err "DIR_PATH ($DIR_PATH) != ROOT ($ROOT) — fix utils.sh / sourcing"

# --- Layout ---
[[ -d "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" ]] || err "missing Fairseq unsupervised tree: $FAIRSEQ_ROOT"

TB_DIR="${DIR_PATH}/tb"
mkdir -p "$TB_DIR"

PRE="$CLUSTERING_DIR/$W2VU_PRECOMPUTE_SUBDIR"
if [[ -f "$PRE/train.npy" && -f "$PRE/valid.npy" ]]; then
  echo "OK: precompute features present ($PRE)"
else
  echo "WARN: precompute not found under $PRE (run ./run_wav2vec.sh after data prep)."
fi

TP="$TEXT_OUTPUT/phones"
if [[ -f "$TP/lm.phones.filtered.04.bin" && -f "$TP/train.idx" && -f "$TP/dict.txt" ]]; then
  echo "OK: phone text + KenLM present ($TP)"
else
  echo "WARN: prepare_text outputs missing under $TP"
fi

# --- TensorBoard event files (recurse: Hydra may nest runs) ---
mapfile -t ev < <(find "$TB_DIR" -name 'events.out.tfevents.*' -type f 2>/dev/null | head -50)
if [[ ${#ev[@]} -eq 0 ]]; then
  echo "INFO: no TensorBoard event files under $TB_DIR yet (expected before training)."
else
  echo "OK: found ${#ev[@]} TensorBoard event file(s) under $TB_DIR (showing newest 3):"
  while IFS= read -r line; do echo "  $line"; done < <(
    find "$TB_DIR" -name 'events.out.tfevents.*' -type f -printf '%T+\t%p\n' 2>/dev/null | sort -r | head -3
  )
fi

echo ""
echo "TensorBoard (reloads runs periodically; use absolute logdir):"
echo "  source venv/bin/activate"
echo "  tensorboard --logdir=\"$TB_DIR\" --bind_all --reload_interval=5"
echo ""

if command -v tensorboard >/dev/null 2>&1; then
  if tensorboard --help 2>&1 | grep -q reload_multifile; then
    echo "Tip: add --reload_multifile=true if new runs do not appear without restarting TensorBoard."
  fi
fi

# --- Optional smoke (GPU + data) ---
if [[ "${RUN_SMOKE_TB:-0}" == "1" ]]; then
  echo "== RUN_SMOKE_TB=1: running scripts/smoke_tb_w2vu.sh"
  bash "$ROOT/scripts/smoke_tb_w2vu.sh"
fi
if [[ "${RUN_SMOKE_TRAIN:-0}" == "1" ]]; then
  echo "== RUN_SMOKE_TRAIN=1: running scripts/smoke_w2vu_train.sh"
  bash "$ROOT/scripts/smoke_w2vu_train.sh"
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "validate_pipeline.sh: all checks passed."
