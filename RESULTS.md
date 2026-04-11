# Experiment results log

Append a new section per **training** or **eval** run (or one section that covers both for the same checkpoint). Keep Hydra snapshots so configurations stay reproducible.

## Where configs are saved

| Kind | Location |
|------|----------|
| Training (Hydra) | `outputs/<date>/<time>/.hydra/config.yaml`, `overrides.yaml`, `hydra.yaml` |
| Eval (`w2vu_generate`) | Run directory with `w2vu_generate.log` and often `.hydra/` if Hydra creates one |

**Tip:** After a run, copy the path to `config.yaml` below, or paste the diff of `overrides.yaml`.

### Phone references (PER)

When hypotheses and references are space-separated **phones**, the `w2vu_generate` line reports **`WER:`** but it is **phone error rate (PER)** for that setup.

1. Ensure the task uses `labels: phn` and a phone `dict.txt` (see training/eval configs).
2. Build `{split}.phn` next to the precomputed features (one line per `{split}.lengths` row, same order as `{split}.tsv`):

   ```bash
   pip install -r requirements-pipeline.txt   # datasets, g2p-en, …
   python scripts/build_split_phn_references.py \
     --precompute-dir data/clustering/librispeech/precompute_pca512_cls128_mean_pooled \
     --split test \
     --dict data/text/phones/dict.txt
   ```

   For `train_*.wav`, pass `--train-split` to match `scripts/download_librispeech.py` (e.g. `train.100`, `train.360`).

3. Re-run evaluation; the log line shows **`WER:`** as PER.

---

## Template (copy below this line for each experiment)

```markdown
### YYYY-MM-DD — <short label, e.g. libri GAN v1>

#### Identity
- **Notes:**
- **Git commit:** `git rev-parse --short HEAD`
- **Hardware / env:** local GPU / Modal A10G / …

#### Training
- **Command:** e.g. `bash run_gans.sh` or `modal run scripts/run_modal.py`
- **Hydra output dir:** `outputs/…` or `multirun/…`
- **Checkpoint:** `outputs/…/checkpoint_best.pt` (and `checkpoint_last.pt` if relevant)
- **Best metric (from fairseq logs):** e.g. `weighted_lm_ppl=…` @ update …
- **Stopped at:** update … / reason (converged, max_update, early stop, …)
- **TensorBoard:** path if used

**Training hyperparameters** *(from `fairseq_/examples/wav2vec/unsupervised/config/gan/w2vu.yaml` + CLI overrides)*

| Group | Key | Value |
|-------|-----|-------|
| optimization | max_update | |
| dataset | batch_size | |
| dataset | validate_interval_updates | |
| task | data | |
| task | text_data | |
| task | kenlm_path | |
| model | code_penalty | |
| model | gradient_penalty | |
| model | smoothness_weight | |
| model | temp | |
| optimizer | generator lr | |
| optimizer | discriminator lr | |
| checkpoint | best_checkpoint_metric | |

*Add any extra overrides (e.g. `+optimizer…`) as extra rows.*

#### Evaluation
- **Command:** e.g. `bash run_eval.sh <checkpoint_relpath>`
- **Checkpoint:** same as above
- **Subset:** e.g. `test`
- **WER:** *(needs `targets` / references; else `None`)*
- **LM PPL:** *(with `fairseq.task.kenlm_path` / phone LM .bin)*
- **UER to viterbi:** *(only if `viterbi_transcript` set; else not meaningful)*
- **Transcriptions:** `data/transcription_phones/test.txt`, `test_units.txt`
- **Log snippet:** paste final `Generate …` line

**Eval hyperparameters** *(from `config/generate/viterbi.yaml` + CLI)*

| Key | Value |
|-----|-------|
| fairseq.task.data | |
| fairseq.task.text_data | |
| fairseq.task.kenlm_path | |
| fairseq.dataset.gen_subset | |
| post_process | |
| w2l_decoder | |
| lm_weight | |
| targets | *(if used for WER/PER)* |

#### Config artifacts
- Training: `outputs/<run>/.hydra/config.yaml`
- Eval: *(path to log or .hydra if present)*
```

---

## Logged experiments

### 2026-04-11 — LibriSpeech test decode (example)

#### Identity
- **Notes:** Example row; replace with your own runs.
- **Git commit:** *(fill in)*
- **Hardware / env:** local `venv`, GPU

#### Training
- **Command:** *(fill in; e.g. `bash run_gans.sh`)*
- **Hydra output dir:** `outputs/2026-04-10/12-59-18/` *(example)*
- **Checkpoint:** `outputs/2026-04-10/12-59-18/checkpoint_best.pt`
- **Best metric (from fairseq logs):** *(fill in)*
- **Stopped at:** *(fill in)*

**Training hyperparameters**

| Group | Key | Value |
|-------|-----|-------|
| optimization | max_update | *(from run `config.yaml`)* |
| dataset | batch_size | *(from run)* |
| task | kenlm_path | `data/text/phones/lm.phones.filtered.04.bin` *(typical)* |
| model | code_penalty | *(from run)* |
| model | gradient_penalty | *(from run)* |
| model | smoothness_weight | *(from run)* |

#### Evaluation
- **Command:** `bash run_eval.sh outputs/2026-04-10/12-59-18/checkpoint_best.pt`
- **Checkpoint:** `outputs/2026-04-10/12-59-18/checkpoint_best.pt`
- **Subset:** `test`
- **WER / PER:** `None` without `{split}.phn`; after `scripts/build_split_phn_references.py`, **`WER:` in the log is PER** (phones).
- **LM PPL:** `208.02` *(phone KenLM scoring of hypotheses)*
- **UER to viterbi:** `0` *(not computed without `viterbi_transcript`)*
- **Transcriptions:** `data/transcription_phones/test.txt`, `test_units.txt`

**Log snippet**

```text
| Generate test with beam=5, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: None, LM_PPL: 208.02451534855075, num feats: 44519, length: 31477, UER to viterbi: 0, score: None
```

**Eval hyperparameters**

| Key | Value |
|-----|-------|
| fairseq.task.data | `data/clustering/librispeech/precompute_pca512_cls128_mean_pooled` |
| fairseq.task.text_data | `data/text/phones/` |
| fairseq.task.kenlm_path | `data/text/phones/lm.phones.filtered.04.bin` |
| fairseq.dataset.gen_subset | `test` |
| post_process | `silence` |
| w2l_decoder | `VITERBI` |

#### Config artifacts
- Training: `outputs/2026-04-10/12-59-18/.hydra/config.yaml` *(if present)*
- Eval: `w2vu_generate.log` in cwd when run locally

---

*(Add new `### YYYY-MM-DD — …` sections above this line, or below the example, newest-first or oldest-first—stay consistent.)*
