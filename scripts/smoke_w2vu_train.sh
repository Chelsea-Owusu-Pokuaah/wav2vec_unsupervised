#!/usr/bin/env bash
# Quick local sanity check for Wav2Vec-U GAN training (no Modal, no Hydra multirun).
# Uses your existing prepared ``data/`` paths from ``utils.sh`` but stops after a few updates.
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
echo "Smoke train: max_update=${MAX_UPDATE} (set SMOKE_MAX_UPDATE to override)"

fairseq-hydra-train \
  --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
  --config-name w2vu \
  task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  task.text_data="$TEXT_OUTPUT/phones/" \
  task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
  common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
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
