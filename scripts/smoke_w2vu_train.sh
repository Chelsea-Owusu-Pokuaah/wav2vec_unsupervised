#!/usr/bin/env bash
# Quick local sanity check for Wav2Vec-U GAN training (no Modal, no Hydra multirun).
# Uses your existing prepared ``data/`` paths from ``utils.sh`` but stops after a few updates.
# Writes TensorBoard events under ``<repo>/tb`` (absolute path, like ``gans_functions.sh``) and
# runs ``scripts/plot_training_curves.py`` so the loss-curve visualization is exercised.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/utils.sh"

activate_venv 2>/dev/null || true
setup_path
create_dirs

export PYTHONPATH="${FAIRSEQ_ROOT}:${PYTHONPATH:-}"
export PREFIX="${PREFIX:-w2v_unsup_smoke}"

MAX_UPDATE="${SMOKE_MAX_UPDATE:-5}"
# Default 1 so short runs still emit TensorBoard scalars (w2vu default log_interval=100 would skip them).
LOG_INTERVAL="${SMOKE_LOG_INTERVAL:-1}"
TB_DIR="${SMOKE_TENSORBOARD_DIR:-$ROOT/tb}"
mkdir -p "$TB_DIR"

if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "Smoke train: max_update=${MAX_UPDATE} log_interval=${LOG_INTERVAL} tensorboard=${TB_DIR}"
echo "  (override: SMOKE_MAX_UPDATE, SMOKE_LOG_INTERVAL, SMOKE_TENSORBOARD_DIR)"

fairseq-hydra-train \
  --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
  --config-name w2vu \
  task.data="$CLUSTERING_DIR/$W2VU_PRECOMPUTE_SUBDIR" \
  task.text_data="$TEXT_OUTPUT/phones/" \
  task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
  common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  common.tensorboard_logdir="$TB_DIR" \
  common.log_interval="$LOG_INTERVAL" \
  model.input_dim="$W2VU_INPUT_DIM" \
  model.code_penalty=6 \
  model.gradient_penalty=0.5 \
  model.smoothness_weight=1.5 \
  common.seed=0 \
  optimization.max_update="$MAX_UPDATE" \
  dataset.batch_size=32 \
  dataset.num_workers="${SMOKE_NUM_WORKERS:-2}" \
  checkpoint.save_interval_updates=999999 \
  dataset.validate_interval_updates=999999 \
  +optimizer.groups.generator.optimizer.lr="[0.00004]" \
  +optimizer.groups.discriminator.optimizer.lr="[0.00002]" \
  '~optimizer.groups.generator.optimizer.amsgrad' \
  '~optimizer.groups.discriminator.optimizer.amsgrad'

echo "Smoke train finished OK."

if [[ -f "$ROOT/scripts/plot_training_curves.py" ]]; then
  SMOKE_PLOT="${SMOKE_PLOT:-$ROOT/smoke_training_curves.png}"
  echo "Smoke plot: writing ${SMOKE_PLOT}"
  "$PYTHON" "$ROOT/scripts/plot_training_curves.py" \
    --tensorboard-dir "$TB_DIR" \
    -o "$SMOKE_PLOT"
  echo "Smoke visualization finished OK (${SMOKE_PLOT})."
else
  echo "Optional: add scripts/plot_training_curves.py to plot curves from TensorBoard logs."
fi
