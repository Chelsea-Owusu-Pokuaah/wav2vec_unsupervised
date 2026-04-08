#!/usr/bin/env python3
"""
Download LibriSpeech (Hugging Face) and lay it out for wav2vec_unsupervised.

Produces:
  <output-dir>/train_wav/*.wav   — training utterances (16 kHz mono PCM WAV)
  <output-dir>/val_wav/*.wav     — LibriSpeech dev-clean (validation split)
  <output-dir>/test_wav/*.wav    — LibriSpeech test-clean
  <output-dir>/unlabelled_text.txt — one sentence per line (from training transcripts)

Requires:
  pip install datasets soundfile tqdm

Example (after activation of project venv):
  python scripts/download_librispeech.py

Then run the pipeline (paths match README Step 3):
  ./run_wav2vec.sh \\
    "$HOME/NLP/wav2vec_unsupervised/data/librispeech/train_wav" \\
    "$HOME/NLP/wav2vec_unsupervised/data/librispeech/val_wav" \\
    "$HOME/NLP/wav2vec_unsupervised/data/librispeech/test_wav" \\
    "$HOME/NLP/wav2vec_unsupervised/data/librispeech/unlabelled_text.txt"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TARGET_SR = 16_000


def _huggingface_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()


def _find_extracted_dir_containing(hf_home: Path, basename: str) -> Path:
    """
    Hub Parquet often stores a bogus absolute `file` path (another user's cache).
    Real .flac files live under datasets/downloads/extracted/<hash>/ on this machine.
    """
    extracted = hf_home / "datasets" / "downloads" / "extracted"
    if not extracted.is_dir():
        raise FileNotFoundError(
            f"HF extract directory missing: {extracted}. Run load_dataset once with network."
        )
    for sub in sorted(extracted.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / basename
        if cand.is_file():
            return sub
    for sub in extracted.iterdir():
        if not sub.is_dir():
            continue
        for hit in sub.rglob(basename):
            if hit.is_file():
                return hit.parent
    raise FileNotFoundError(
        f"Could not find {basename!r} under {extracted}. "
        "Clear ~/.cache/huggingface/datasets and re-download LibriSpeech."
    )


def _no_torchcodec_audio(ds):
    """
    HF datasets may decode Audio via torchcodec, which breaks on PyTorch 2.3
    (ImportError: register_fake). Use file-backed audio and read with soundfile.
    """
    from datasets import Audio

    return ds.cast_column("audio", Audio(decode=False))


def _dataset_for_wav_reading(ds):
    """
    Use Audio(decode=False) so torchcodec is not loaded (PyTorch 2.3 incompatible).
    We resolve paths ourselves: Hub `file` column may point at another machine's cache.
    """
    return _no_torchcodec_audio(ds.select_columns(["audio"]))


def _make_waveform_reader():
    """Build a reader that caches the local HF extract dir after first lookup."""
    import io

    import numpy as np
    import soundfile as sf

    hf_home = _huggingface_home()
    cached_root: list[Path | None] = [None]

    def read_waveform(ex: dict):
        # Optional `file` column: only trust paths that actually exist here.
        fp = ex.get("file")
        if fp is not None and str(fp).strip() != "":
            p = Path(str(fp).strip()).expanduser()
            if p.is_file():
                arr, sr = sf.read(str(p))
                return arr.astype(np.float32), int(sr)

        audio = ex["audio"]
        if not isinstance(audio, dict):
            raise TypeError(f"Expected audio dict, got {type(audio)}")

        if audio.get("array") is not None:
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            return arr, sr

        raw = audio.get("bytes")
        if raw is not None:
            arr, sr = sf.read(io.BytesIO(raw))
            return arr.astype(np.float32), int(sr)

        rel = audio.get("path")
        if not rel:
            raise ValueError("audio has no path/bytes/array")

        name = Path(str(rel)).name
        if cached_root[0] is not None:
            cand = cached_root[0] / name
            if cand.is_file():
                arr, sr = sf.read(str(cand))
                return arr.astype(np.float32), int(sr)
        cached_root[0] = _find_extracted_dir_containing(hf_home, name)
        cand = cached_root[0] / name
        if not cand.is_file():
            raise FileNotFoundError(f"Expected under HF cache: {cand}")
        arr, sr = sf.read(str(cand))
        return arr.astype(np.float32), int(sr)

    return read_waveform


def _require_deps():
    try:
        import datasets  # noqa: F401
        import soundfile as sf  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError as e:
        print(
            "Missing dependency. In your project venv run:\n"
            "  pip install datasets soundfile tqdm\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _resample_if_needed(audio: "object", sr: int):
    """Return (numpy array, sample_rate) at TARGET_SR if possible."""
    import numpy as np

    arr = audio
    if hasattr(audio, "numpy"):
        arr = audio.numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if sr == TARGET_SR:
        return arr, sr
    try:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        return arr, TARGET_SR
    except ImportError:
        try:
            from scipy import signal

            num = int(len(arr) * TARGET_SR / sr)
            arr = signal.resample(arr, num)
            return arr.astype(np.float32), TARGET_SR
        except ImportError:
            raise RuntimeError(
                f"Audio is {sr} Hz but project expects {TARGET_SR} Hz. "
                "Install librosa or scipy: pip install librosa"
            ) from None


def _write_wavs(
    split_ds,
    out_dir: Path,
    name_prefix: str,
    max_samples: int | None,
    read_waveform,
) -> int:
    import soundfile as sf
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(split_ds)
    if max_samples is not None:
        n = min(n, max_samples)

    for i in tqdm(range(n), desc=f"Writing {out_dir.name}"):
        ex = split_ds[i]
        arr, sr = read_waveform(ex)
        arr, sr = _resample_if_needed(arr, sr)
        path = out_dir / f"{name_prefix}_{i:08d}.wav"
        sf.write(str(path), arr, sr, subtype="PCM_16")
    return n


def _write_text_from_train(split_ds, text_path: Path, n_lines: int) -> None:
    text_ds = split_ds.select_columns(["text"])
    with text_path.open("w", encoding="utf-8") as f:
        for i in range(n_lines):
            line = text_ds[i]["text"]
            f.write(line.strip() + "\n")


def main() -> None:
    _require_deps()
    from datasets import load_dataset

    project_root = Path(__file__).resolve().parent.parent
    default_out = project_root / "data" / "librispeech"

    parser = argparse.ArgumentParser(
        description="Export LibriSpeech to flat WAV dirs + unlabelled text for wav2vec_unsupervised."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Root output directory (default: {default_out})",
    )
    parser.add_argument(
        "--config",
        default="clean",
        choices=("clean", "other"),
        help='Hugging Face config: "clean" (default) or "other".',
    )
    parser.add_argument(
        "--train-split",
        default="train.100",
        help='Training split name, e.g. train.100, train.360 (config "clean").',
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Cap number of training utterances (for quick tests).",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Cap validation utterances.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Cap test utterances.",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Do not write unlabelled_text.txt (audio only).",
    )
    args = parser.parse_args()

    out = args.output_dir.resolve()
    train_wav = out / "train_wav"
    val_wav = out / "val_wav"
    test_wav = out / "test_wav"
    text_file = out / "unlabelled_text.txt"

    print(f"Loading LibriSpeech ({args.config}) from Hugging Face Hub (may download on first run)...")
    train_ds = load_dataset("librispeech_asr", args.config, split=args.train_split)
    val_ds = load_dataset("librispeech_asr", args.config, split="validation")
    test_ds = load_dataset("librispeech_asr", args.config, split="test")

    train_audio = _dataset_for_wav_reading(train_ds)
    val_audio = _dataset_for_wav_reading(val_ds)
    test_audio = _dataset_for_wav_reading(test_ds)

    print(f"Train split '{args.train_split}': {len(train_ds)} utterances")
    print(f"Validation: {len(val_ds)} utterances")
    print(f"Test: {len(test_ds)} utterances")
    print(f"Using HF cache under: {_huggingface_home()}")

    read_wav = _make_waveform_reader()

    n_train = _write_wavs(train_audio, train_wav, "train", args.max_train, read_wav)
    _write_wavs(val_audio, val_wav, "val", args.max_val, read_wav)
    _write_wavs(test_audio, test_wav, "test", args.max_test, read_wav)

    if not args.skip_text:
        _write_text_from_train(train_ds, text_file, n_train)
        print(f"Wrote text corpus: {text_file} ({n_train} lines)")

    print("\nDone. Use these paths with ./run_wav2vec.sh:")
    print(f'  TRAIN_AUDIO="{train_wav}"')
    print(f'  VAL_AUDIO="{val_wav}"')
    print(f'  TEST_AUDIO="{test_wav}"')
    print(f'  TEXT="{text_file}"')
    print("\nExample:")
    print(
        f'  ./run_wav2vec.sh "{train_wav}" "{val_wav}" "{test_wav}" "{text_file}"'
    )


if __name__ == "__main__":
    main()
