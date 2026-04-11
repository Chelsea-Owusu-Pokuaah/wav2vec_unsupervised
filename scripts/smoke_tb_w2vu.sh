#!/usr/bin/env bash
# Fast check that TensorBoard logging survives Hydra + validation (same failure mode as Modal).
# Run from repo root: ./scripts/smoke_tb_w2vu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/utils.sh"

activate_venv 2>/dev/null || true
setup_path
create_dirs

tb_dir="${DIR_PATH}/tb"
mkdir -p "$tb_dir"

export PYTHONPATH="${FAIRSEQ_ROOT}:${PYTHONPATH:-}"
export PREFIX="${PREFIX:-w2v_tb_smoke}"

echo "TB smoke: writing under ${tb_dir} (absolute)"
rm -rf "${tb_dir:?}/"* 2>/dev/null || true

fairseq-hydra-train \
  --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
  --config-name w2vu \
  task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  task.text_data="$TEXT_OUTPUT/phones/" \
  task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
  common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  common.tensorboard_logdir="$tb_dir" \
  model.code_penalty=6 \
  model.gradient_penalty=0.5 \
  model.smoothness_weight=1.5 \
  common.seed=0 \
  optimization.max_update=8 \
  dataset.batch_size=32 \
  dataset.num_workers="${SMOKE_NUM_WORKERS:-2}" \
  dataset.validate_interval_updates=1 \
  dataset.validate_after_updates=0 \
  checkpoint.save_interval_updates=999999 \
  +optimizer.groups.generator.optimizer.lr="[0.00004]" \
  +optimizer.groups.discriminator.optimizer.lr="[0.00002]" \
  '~optimizer.groups.generator.optimizer.amsgrad' \
  '~optimizer.groups.discriminator.optimizer.amsgrad'

echo "OK: TB smoke finished. Event files:"
find "$tb_dir" -name 'events.out.*' 2>/dev/null | head -20 || echo "(none — check tb/ subdirs)"
