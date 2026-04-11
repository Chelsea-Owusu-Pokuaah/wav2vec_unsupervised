"""
Run GAN training (`run_gans.sh` → `fairseq-hydra-train`) on Modal with a GPU.

Layout matches `utils.sh`: project lives at ``$HOME/NLP/wav2vec_unsupervised`` (we set
``HOME=/root``). The ``data/`` directory is a Modal Volume so you can sync prepared
artifacts from your machine.

Setup
-----
1. ``pip install modal`` and ``modal setup`` (or set Modal token env vars).
2. Upload your local ``data/`` tree (outputs from ``prepare_audio`` + ``prepare_text``,
   checkpoints, etc.)::

       modal volume put wav2vec-unsupervised-data ./data /

   Run from the repo root so ``./data`` is your project data directory.

3. Optional: upload large models if they are not already under ``data/``::

       modal volume put wav2vec-unsupervised-data ./pre-trained /pre-trained
       modal volume put wav2vec-unsupervised-data ./lid_model /lid_model

   If you keep them only on disk locally, add them to the image by removing them from
   ``IMAGE_IGNORE`` below (image rebuilds get slower).

Run training::

    modal run scripts/run_modal.py

Run Viterbi eval / generation (``run_eval.sh``) with a checkpoint path **relative to the
repo root** (same as local ``bash run_eval.sh <path>``)::

    MODAL_EVAL_CHECKPOINT=outputs/2026-04-10/12-57-48/checkpoint_best.pt \\
      modal run scripts/run_modal.py

Checkpoints under ``outputs/`` or ``multirun/`` from a **previous** Modal train are only
available if that path still exists in the rebuilt image **or** you put the ``.pt`` on
the volume (e.g. ``data/checkpoints/...``) and pass that relative path.

Override GPU type (Modal-supported name)::

    MODAL_GPU=A100 modal run scripts/run_modal.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "wav2vec-unsupervised"
VOLUME_NAME = "wav2vec-unsupervised-data"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REMOTE_ROOT = Path("/root/NLP/wav2vec_unsupervised")
_REMOTE_DATA = _REMOTE_ROOT / "data"

# Keep image builds small; sync ``data/`` via the volume instead.
IMAGE_IGNORE = [
    ".git",
    "venv",
    ".venv",
    "data",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.log",
    ".mypy_cache",
]

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_cuda121 = "https://download.pytorch.org/whl/cu121"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "cmake",
        "pkg-config",
        "ffmpeg",
        "libsndfile1",
        "libsndfile1-dev",
        "zsh",
        # Native extensions (KenLM): Python headers + full Boost stack (KenLM is picky on Debian).
        "python3-dev",
        "libeigen3-dev",
        "libboost-all-dev",
        "zlib1g-dev",
        "libbz2-dev",
        "liblzma-dev",
    )
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        extra_index_url=_cuda121,
    )
    .add_local_dir(
        str(_REPO_ROOT),
        remote_path=str(_REMOTE_ROOT),
        copy=True,
        ignore=IMAGE_IGNORE,
    )
    .run_commands(
        # fairseq 0.12.2 depends on omegaconf metadata rejected by pip>=24.1.
        "python -m pip install 'pip<24.1'",
        f"pip install -e {_REMOTE_ROOT}/fairseq_",
        # fairseq deps may pull NumPy 2.x; torch 2.1.2 expects NumPy 1.x ABI (see training logs).
        "pip install 'numpy<2'",
        # ``unpaired_audio_text`` does ``import kenlm`` for ``task.kenlm_path`` (same as local setup).
        "pip install https://github.com/kpu/kenlm/archive/master.zip",
        # ``unpaired_audio_text.valid_step`` does ``import editdistance`` (not a fairseq dependency).
        "pip install editdistance",
        # ``w2vu.yaml`` sets ``tensorboard_logdir``; fairseq uses ``torch.utils.tensorboard`` when available.
        "pip install tensorboard",
        # Fail the image build if critical imports break (cheap sanity check).
        "python -c \"import numpy, torch, kenlm, editdistance; print('modal image imports OK')\"",
    )
)


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A10G"),
    volumes={str(_REMOTE_DATA): data_volume},
    timeout=86400,
)
def train_gans_cloud() -> int:
    """
    Runs ``run_gans.sh`` in the container. Requires ``data/`` on the volume to match
    what ``gans_functions.sh`` expects (clustering features, ``text/phones`` LM, etc.).
    """
    env = os.environ.copy()
    env["HOME"] = "/root"
    # Skip venv activation when no ``venv`` exists in the image (see ``utils.sh``).
    env["VENV_PATH"] = ""

    repo = str(_REMOTE_ROOT)
    rc = subprocess.run(
        ["bash", str(_REMOTE_ROOT / "run_gans.sh")],
        cwd=repo,
        env=env,
        check=False,
    ).returncode

    data_volume.commit()
    return rc


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

    print(f"Starting remote GAN training (volume {VOLUME_NAME} → {_REMOTE_DATA})...")
    exit_code = train_gans_cloud.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
    print("Finished training; volume changes committed.")
