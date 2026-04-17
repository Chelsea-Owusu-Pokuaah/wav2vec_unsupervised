"""
Run GAN training (`run_gans.sh` → `fairseq-hydra-train`) on Modal with a GPU.

Layout matches `utils.sh`: project lives at ``$HOME/NLP/wav2vec_unsupervised`` (we set
``HOME=/root``). The ``data/`` directory is a Modal Volume so you can sync prepared
artifacts from your machine.

Setup
-----
See ``MODAL.md`` for the full runbook. Short version:

1. ``bash scripts/modal_preflight.sh`` locally.
2. ``pip install modal`` and ``modal setup`` (or set Modal token env vars).
3. Upload your local ``data/`` tree (outputs from ``prepare_audio`` + ``prepare_text``,
   etc.)::

       modal volume put wav2vec-unsupervised-data ./data /

   Run from the repo root so ``./data`` is your project data directory.

3. Optional: upload large models if they are not already under ``data/``::

       modal volume put wav2vec-unsupervised-data ./pre-trained /pre-trained
       modal volume put wav2vec-unsupervised-data ./lid_model /lid_model

   If you keep them only on disk locally, add them to the image by removing them from
   ``IMAGE_IGNORE`` below (image rebuilds get slower).

Run training (CLI)::

    modal run scripts/run_modal.py

Deploy HTTPS endpoints (POST returns immediately; GPU job keeps running)::

    modal deploy scripts/run_modal.py

    # Start training (optional JSON body)
    curl -X POST "https://<web_start_gan_train-url>" \\
      -H "Content-Type: application/json" \\
      -d '{"eval_after_train": true, "force_train_gans": "1"}'

    # Eval only (checkpoint relative to repo root)
    curl -X POST "https://<web_start_w2vu_eval-url>" \\
      -H "Content-Type: application/json" \\
      -d '{"checkpoint_relpath": "data/modal_gan_outputs/outputs/.../checkpoint_best.pt"}'

Train **and** run eval in the **same** Modal run (uses ``outputs/.../checkpoint_best.pt``
while the container is still up; no need to wait for volume copy)::

    MODAL_EVAL_AFTER_TRAIN=1 modal run scripts/run_modal.py

Or via deploy: POST body ``{"eval_after_train": true}`` to ``web_start_gan_train``.

Run Viterbi eval / generation (``run_eval.sh``) with a checkpoint path **relative to the
repo root** (same as local ``bash run_eval.sh <path>``)::

    MODAL_EVAL_CHECKPOINT=outputs/2026-04-10/12-57-48/checkpoint_best.pt \\
      modal run scripts/run_modal.py

After training, ``outputs/``, ``multirun/``, and ``tb/`` are copied into
``data/modal_gan_outputs/`` on the volume (see ``_sync_training_artifacts_to_data_volume``)
so ``checkpoint_best.pt`` paths look like
``data/modal_gan_outputs/outputs/<date>/<time>/checkpoint_best.pt`` after you
``modal volume get`` the volume back to your machine.

For eval on Modal, pass that path relative to the repo root, e.g.
``MODAL_EVAL_CHECKPOINT=data/modal_gan_outputs/outputs/.../checkpoint_best.pt``.

Override GPU type (Modal-supported name; applies at **deploy** time)::

    MODAL_GPU=A100 modal deploy scripts/run_modal.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import modal
try:
    from fastapi import Request
except ImportError:
    from typing import Any as Request  # type: ignore[assignment]

APP_NAME = "wav2vec-unsupervised"
VOLUME_NAME = "wav2vec-unsupervised-data"

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_REMOTE_ROOT = Path("/root/NLP/wav2vec_unsupervised")
_REMOTE_DATA = _REMOTE_ROOT / "data"

def _modal_dataset_name() -> str:
    return (os.environ.get("MODAL_DATASET_NAME") or "librispeech").strip() or "librispeech"


def _modal_w2vu_precompute_subdir() -> str:
    return (
        os.environ.get("MODAL_W2VU_PRECOMPUTE_SUBDIR", "").strip()
        or os.environ.get("W2VU_PRECOMPUTE_SUBDIR", "").strip()
        or "precompute_pca512_cls128_mean_pooled"
    )


def _modal_w2vu_input_dim() -> str:
    return (
        os.environ.get("MODAL_W2VU_INPUT_DIM", "").strip()
        or os.environ.get("W2VU_INPUT_DIM", "").strip()
        or "512"
    )


def _resolve_w2vu_train_config(
    w2vu_precompute_subdir: str | None,
    w2vu_input_dim: str | None,
    dataset_name: str | None,
) -> tuple[str, str, str]:
    sub = (w2vu_precompute_subdir or _modal_w2vu_precompute_subdir()).strip()
    dim = (w2vu_input_dim or _modal_w2vu_input_dim()).strip()
    ds = (dataset_name or _modal_dataset_name()).strip()
    return sub, dim, ds


# Must match ``CLUSTERING_DIR/$W2VU_PRECOMPUTE_SUBDIR`` in ``gans_functions.sh`` / ``w2vu.yaml``.
def _assert_remote_training_data(precompute_subdir: str, dataset_name: str) -> None:
    root = _REMOTE_DATA
    pre = root / "clustering" / dataset_name / precompute_subdir
    need = [
        pre / "train.npy",
        pre / "valid.npy",
        root / "text/phones/lm.phones.filtered.04.bin",
        root / "text/phones/dict.txt",
        root / "text/phones/train.idx",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        msg = (
            "Missing training inputs on the Modal volume (under /data). "
            "From the repo root: `modal volume put wav2vec-unsupervised-data ./data /` "
            "after `bash scripts/modal_preflight.sh` passes locally.\n"
            f"Expected clustering features under: clustering/{dataset_name}/{precompute_subdir}/\n"
            "Override with MODAL_DATASET_NAME, MODAL_W2VU_PRECOMPUTE_SUBDIR (and MODAL_W2VU_INPUT_DIM) "
            "if your prep used different PCA/cluster dirs.\n"
            "Missing:\n  "
            + "\n  ".join(str(p) for p in missing)
        )
        raise FileNotFoundError(msg)


def _sync_training_artifacts_to_data_volume() -> None:
    """
    Hydra/Fairseq write under ``outputs/``, ``multirun/``, ``tb/`` on the container
    disk. Only ``data/`` is on the Modal volume, so copy those trees into
    ``data/modal_gan_outputs/`` before ``commit()`` so checkpoints survive the run.
    """
    dest_parent = _REMOTE_DATA / "modal_gan_outputs"
    dest_parent.mkdir(parents=True, exist_ok=True)
    for name in ("outputs", "multirun", "tb"):
        src = _REMOTE_ROOT / name
        if not src.is_dir():
            continue
        dst = dest_parent / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _find_latest_checkpoint_best(repo: Path) -> Path | None:
    """Newest ``checkpoint_best.pt`` under ``outputs/`` or ``multirun/`` (by mtime)."""
    candidates: list[Path] = []
    for pattern in ("outputs/**/checkpoint_best.pt", "multirun/**/checkpoint_best.pt"):
        candidates.extend(repo.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_cuda121 = "https://download.pytorch.org/whl/cu121"

_CODE_IGNORE = [
    "fairseq_", "data", "pre-trained", "lid_model",
    "kenlm", "venv", ".venv", ".git",
    "outputs", "multirun", "tb", "logs", "sequence",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "*.pyc", "*.log", "*.bak",
]

# ── Image ──────────────────────────────────────────────────────────────────────
# Only fairseq_ is baked in (needs pip install -e at build time).  All other
# repo files come from the code mount, so shell-script edits never trigger a
# rebuild.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "curl", "build-essential", "cmake", "pkg-config",
        "ffmpeg", "libsndfile1", "libsndfile1-dev", "zsh",
        # KenLM native extensions
        "python3-dev", "libeigen3-dev", "libboost-all-dev",
        "zlib1g-dev", "libbz2-dev", "liblzma-dev",
    )
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        extra_index_url=_cuda121,
    )
    # fairseq_ in its own cached layer — exclude .git to avoid modification errors
    .add_local_dir(
        str(_REPO_ROOT / "fairseq_"),
        remote_path=str(_REMOTE_ROOT / "fairseq_"),
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", "*.egg-info"],
    )
    .run_commands(
        # fairseq 0.12.2 needs pip<24.1 due to omegaconf metadata
        "python -m pip install 'pip<24.1'",
        f"pip install -e {_REMOTE_ROOT}/fairseq_",
        # torch 2.1.2 requires NumPy 1.x ABI
        "pip install 'numpy<2'",
        "pip install https://github.com/kpu/kenlm/archive/master.zip",
        "pip install editdistance",
        "pip install tensorboard",
        "pip install 'fastapi[standard]'",
        "python -c \"import numpy, torch, kenlm, editdistance; print('modal image imports OK')\"",
    )
    # Code files: incremental sync per-run (copy=False).
    .add_local_dir(
        str(_REPO_ROOT),
        remote_path=str(_REMOTE_ROOT),
        copy=False,
        ignore=_CODE_IGNORE,
    )
)


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A10G"),
    volumes={str(_REMOTE_DATA): data_volume},
    timeout=86400,
)
def train_gans_cloud(
    eval_after_train: bool | None = None,
    force_train_gans: str | None = None,
    w2vu_precompute_subdir: str | None = None,
    w2vu_input_dim: str | None = None,
    dataset_name: str | None = None,
) -> int:
    """
    Runs ``run_gans.sh`` in the container. Requires ``data/`` on the volume to match
    what ``gans_functions.sh`` expects (clustering features, ``text/phones`` LM, etc.).

    ``eval_after_train`` / ``force_train_gans`` override ``MODAL_EVAL_AFTER_TRAIN`` /
    ``FORCE_TRAIN_GANS`` when not ``None`` (for ``modal deploy`` + JSON body).

    ``w2vu_precompute_subdir`` / ``w2vu_input_dim`` / ``dataset_name`` override
    ``MODAL_W2VU_*`` / ``MODAL_DATASET_NAME`` (directory under ``data/clustering/<dataset>/``).
    """
    sub, dim, ds = _resolve_w2vu_train_config(
        w2vu_precompute_subdir, w2vu_input_dim, dataset_name
    )
    _assert_remote_training_data(sub, ds)
    env = os.environ.copy()
    env["HOME"] = "/root"
    # Skip venv activation when no ``venv`` exists in the image (see ``utils.sh``).
    env["VENV_PATH"] = ""
    env["DATASET_NAME"] = ds
    env["W2VU_PRECOMPUTE_SUBDIR"] = sub
    env["W2VU_INPUT_DIM"] = dim
    # Volume often includes ``data/checkpoints/.../progress.checkpoint`` with
    # ``train_gans:COMPLETED`` from a prior local run; still run training on Modal.
    if force_train_gans is not None:
        env["FORCE_TRAIN_GANS"] = force_train_gans
    else:
        env["FORCE_TRAIN_GANS"] = os.environ.get("FORCE_TRAIN_GANS", "1")

    if eval_after_train is None:
        run_eval_after = os.environ.get("MODAL_EVAL_AFTER_TRAIN") == "1"
    else:
        run_eval_after = eval_after_train

    repo = str(_REMOTE_ROOT)
    rc = subprocess.run(
        ["bash", str(_REMOTE_ROOT / "run_gans.sh")],
        cwd=repo,
        env=env,
        check=False,
    ).returncode

    eval_rc = 0
    # Same container still has ``outputs/`` / ``multirun/`` — eval now without a second job.
    if rc == 0 and run_eval_after:
        ckpt = _find_latest_checkpoint_best(_REMOTE_ROOT)
        if ckpt is None:
            print(
                "warning: eval_after_train but no checkpoint_best.pt found under outputs/ or multirun/"
            )
        else:
            rel = ckpt.relative_to(_REMOTE_ROOT)
            print(f"eval_after_train: running eval on {rel} (same container)...")
            eval_rc = subprocess.run(
                ["bash", str(_REMOTE_ROOT / "run_eval.sh"), str(rel)],
                cwd=repo,
                env=env,
                check=False,
            ).returncode

    try:
        _sync_training_artifacts_to_data_volume()
    except OSError as e:
        print(f"warning: could not copy training outputs to volume: {e}")

    data_volume.commit()
    return eval_rc if rc == 0 and eval_rc != 0 else rc


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_EVAL_GPU", os.environ.get("MODAL_GPU", "A10G")),
    volumes={str(_REMOTE_DATA): data_volume},
    timeout=86400,
)
def eval_w2vu_cloud(checkpoint_relpath: str) -> int:
    """
    Runs ``run_eval.sh <checkpoint_relpath>`` (Viterbi ``w2vu_generate``).
    ``checkpoint_relpath`` is relative to the repo root (``DIR_PATH`` in ``utils.sh``).
    """
    env = os.environ.copy()
    env["HOME"] = "/root"
    env["VENV_PATH"] = ""

    repo = str(_REMOTE_ROOT)
    rc = subprocess.run(
        ["bash", str(_REMOTE_ROOT / "run_eval.sh"), checkpoint_relpath],
        cwd=repo,
        env=env,
        check=False,
    ).returncode

    data_volume.commit()
    return rc


def _spawn_kwargs_train_body(body: dict[str, Any]) -> dict[str, Any]:
    """Map JSON POST body keys to ``train_gans_cloud`` keyword arguments."""
    out: dict[str, Any] = {}
    if "eval_after_train" in body and body["eval_after_train"] is not None:
        out["eval_after_train"] = bool(body["eval_after_train"])
    if "force_train_gans" in body and body["force_train_gans"] is not None:
        out["force_train_gans"] = str(body["force_train_gans"])
    if "w2vu_precompute_subdir" in body and body["w2vu_precompute_subdir"] is not None:
        out["w2vu_precompute_subdir"] = str(body["w2vu_precompute_subdir"])
    if "w2vu_input_dim" in body and body["w2vu_input_dim"] is not None:
        out["w2vu_input_dim"] = str(body["w2vu_input_dim"])
    if "dataset_name" in body and body["dataset_name"] is not None:
        out["dataset_name"] = str(body["dataset_name"])
    return out


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def web_start_gan_train(request: Request):
    """
    Fire-and-forget: spawns ``train_gans_cloud`` so the HTTP request returns immediately.
    Use after ``modal deploy scripts/run_modal.py``.

    Optional JSON body keys: ``eval_after_train`` (bool), ``force_train_gans`` (string),
    ``w2vu_precompute_subdir`` (e.g. ``precompute_pca128_cls64_mean_pooled``),
    ``w2vu_input_dim`` (e.g. ``128``), ``dataset_name`` (e.g. ``librispeech``).
    Empty body uses the same defaults as ``modal run`` (``MODAL_*`` env when set).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    kwargs = _spawn_kwargs_train_body(body)
    train_gans_cloud.spawn(**kwargs)
    return {
        "status": "spawned",
        "function": "train_gans_cloud",
        "volume": VOLUME_NAME,
        "kwargs": kwargs,
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def web_start_w2vu_eval(request: Request):
    """
    Fire-and-forget: spawns ``eval_w2vu_cloud``. POST JSON must include
    ``checkpoint_relpath`` (relative to repo root).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    ckpt = (body.get("checkpoint_relpath") or body.get("checkpoint") or "").strip()
    if not ckpt:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="checkpoint_relpath is required (relative to repo root)",
        )
    eval_w2vu_cloud.spawn(ckpt)
    return {
        "status": "spawned",
        "function": "eval_w2vu_cloud",
        "volume": VOLUME_NAME,
        "checkpoint_relpath": ckpt,
    }


def _local_train_kwargs_from_env() -> dict[str, Any]:
    """Build ``train_gans_cloud`` kwargs from the shell that invoked ``modal run``.

    Modal workers do not inherit your laptop's environment; pass these explicitly via
    ``.remote(**kwargs)`` so ``MODAL_W2VU_*`` / ``MODAL_EVAL_AFTER_TRAIN`` / etc. work.
    """
    kwargs: dict[str, Any] = {}
    eat = os.environ.get("MODAL_EVAL_AFTER_TRAIN", "").strip()
    if eat == "1":
        kwargs["eval_after_train"] = True
    elif eat == "0":
        kwargs["eval_after_train"] = False
    if "FORCE_TRAIN_GANS" in os.environ:
        kwargs["force_train_gans"] = os.environ["FORCE_TRAIN_GANS"]
    sub = (
        os.environ.get("MODAL_W2VU_PRECOMPUTE_SUBDIR", "").strip()
        or os.environ.get("W2VU_PRECOMPUTE_SUBDIR", "").strip()
    )
    if sub:
        kwargs["w2vu_precompute_subdir"] = sub
    dim = (
        os.environ.get("MODAL_W2VU_INPUT_DIM", "").strip()
        or os.environ.get("W2VU_INPUT_DIM", "").strip()
    )
    if dim:
        kwargs["w2vu_input_dim"] = dim
    ds = (
        os.environ.get("MODAL_DATASET_NAME", "").strip()
        or os.environ.get("DATASET_NAME", "").strip()
    )
    if ds:
        kwargs["dataset_name"] = ds
    return kwargs


@app.local_entrypoint()
def main() -> None:
    ckpt = os.environ.get("MODAL_EVAL_CHECKPOINT", "").strip()
    if ckpt:
        print(
            f"Starting remote eval (checkpoint {ckpt!r}, volume {VOLUME_NAME} → {_REMOTE_DATA})..."
        )
        exit_code = eval_w2vu_cloud.remote(ckpt)
        if exit_code != 0:
            raise SystemExit(exit_code)
        print("Finished eval; volume changes committed.")
        return

    train_kwargs = _local_train_kwargs_from_env()
    print(f"Starting remote GAN training (volume {VOLUME_NAME} → {_REMOTE_DATA})...")
    if train_kwargs:
        print(f"train_gans_cloud kwargs from local env: {train_kwargs!r}")
    exit_code = train_gans_cloud.remote(**train_kwargs)
    if exit_code != 0:
        raise SystemExit(exit_code)
    print("Finished training; volume changes committed.")
