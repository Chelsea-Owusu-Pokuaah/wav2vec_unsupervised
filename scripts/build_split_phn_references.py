#!/usr/bin/env python3
"""
Build `{split}.phn` reference phone sequences for PER (reported as WER on phones).

`ExtractedFeaturesDataset` loads `path/{split}.phn` when `labels: phn` is set: one
space-separated phone line per row in `{split}.lengths`, in the same order as
`{split}.tsv` data rows (first TSV line is the audio root; row i matches
`.lengths` line i). Lines must stay aligned even if some utterances are filtered
by min/max length (read one label line per `.lengths` line).

WAV names must follow `scripts/download_librispeech.py`: `{train|val|test}_NNNNNNNN.wav`
where the 8-digit index is the position in the matching Hugging Face
`librispeech_asr` split (`train_*` → `--train-split`, e.g. `train.100`;
`val_*` → `validation`; `test_*` → `test`).

Requires:
  pip install datasets g2p-en
  (see requirements-pipeline.txt for compatible pins)

Example:
  python scripts/build_split_phn_references.py \\
    --precompute-dir data/clustering/librispeech/precompute_pca512_cls128_mean_pooled \\
    --split test \\
    --dict data/text/phones/dict.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WAV_RE = re.compile(r"^(train|val|test)_([0-9]{8})\.wav$", re.IGNORECASE)


def load_phone_vocab(dict_path: Path) -> set[str]:
    vocab: set[str] = set()
    with dict_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sym = line.split()[0]
            vocab.add(sym)
    return vocab


def read_tsv_rel_paths(tsv_path: Path) -> list[str]:
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Expected header + data rows in {tsv_path}")
    out: list[str] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        out.append(line.split("\t", 1)[0].strip())
    return out


def read_lengths(lengths_path: Path) -> list[int]:
    out: list[int] = []
    with lengths_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(int(line))
    return out


def parse_wav_index(rel_path: str) -> tuple[str, int]:
    base = Path(rel_path).name
    m = WAV_RE.match(base)
    if not m:
        raise ValueError(
            f"Filename {base!r} does not match {WAV_RE.pattern} "
            "(expected download_librispeech.py naming)."
        )
    prefix = m.group(1).lower()
    idx = int(m.group(2), 10)
    return prefix, idx


def hf_split_name(prefix: str, train_split: str) -> str:
    if prefix == "train":
        return train_split
    if prefix == "val":
        return "validation"
    if prefix == "test":
        return "test"
    raise ValueError(f"Unknown prefix {prefix!r}")


def phones_for_text(text: str, g2p, vocab: set[str]) -> list[str]:
    """Map LibriSpeech transcript to space-separated dict symbols (ARPAbet, no stress)."""
    if not text or not str(text).strip():
        return ["<SIL>"]
    raw = g2p(str(text))
    out: list[str] = []
    for tok in raw:
        if tok is None:
            continue
        if isinstance(tok, str) and (tok.isspace() or tok == ""):
            continue
        s = str(tok).strip()
        if len(s) == 1 and not s.isalpha():
            continue
        stripped = re.sub(r"\d", "", s.upper())
        if stripped and stripped in vocab:
            out.append(stripped)
    if not out:
        return ["<SIL>"]
    return out


def load_hf_dataset(config: str, split: str):
    from datasets import load_dataset

    # Drop audio columns so HF does not decode audio (torchcodec can break on some PyTorch).
    ds = load_dataset("librispeech_asr", config, split=split)
    if "text" not in ds.column_names:
        raise RuntimeError(f"No 'text' column in librispeech_asr split {split!r}")
    return ds.select_columns(["text"])


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    default_dict = root / "data" / "text" / "phones" / "dict.txt"

    parser = argparse.ArgumentParser(
        description="Write {split}.phn next to precomputed features for PER as WER."
    )
    parser.add_argument(
        "--precompute-dir",
        type=Path,
        required=True,
        help="Directory containing {split}.tsv, {split}.lengths, etc.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "valid", "train"),
        required=True,
        help="File stem (test / valid / train), matching fairseq gen_subset names.",
    )
    parser.add_argument(
        "--dict",
        type=Path,
        default=default_dict,
        help=f"Phone dictionary (default: {default_dict})",
    )
    parser.add_argument(
        "--hf-config",
        default="clean",
        choices=("clean", "other"),
        help='Hugging Face librispeech_asr config (default: "clean").',
    )
    parser.add_argument(
        "--train-split",
        default="train.100",
        help="HF split for train_*.wav (must match WAVs from download_librispeech.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: <precompute-dir>/{split}.phn).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first 3 phone lines and exit without writing.",
    )
    args = parser.parse_args()

    pre = args.precompute_dir.resolve()
    stem = args.split
    tsv_path = pre / f"{stem}.tsv"
    lengths_path = pre / f"{stem}.lengths"
    out_path = args.output if args.output else pre / f"{stem}.phn"

    if not tsv_path.is_file():
        print(f"error: missing {tsv_path}", file=sys.stderr)
        return 1
    if not lengths_path.is_file():
        print(f"error: missing {lengths_path}", file=sys.stderr)
        return 1
    if not args.dict.is_file():
        print(f"error: missing dict {args.dict}", file=sys.stderr)
        return 1

    rel_paths = read_tsv_rel_paths(tsv_path)
    lengths = read_lengths(lengths_path)
    if len(rel_paths) != len(lengths):
        print(
            f"error: {tsv_path} has {len(rel_paths)} data rows but "
            f"{lengths_path} has {len(lengths)} lines (must match).",
            file=sys.stderr,
        )
        return 1

    vocab = load_phone_vocab(args.dict)

    try:
        from g2p_en import G2p
    except ImportError:
        print("error: install g2p-en (pip install g2p-en)", file=sys.stderr)
        return 1

    g2p = G2p()

    # Load each HF split we need (at most train / validation / test for this run).
    cache: dict[str, object] = {}

    def get_ds(prefix: str):
        split_name = hf_split_name(prefix, args.train_split)
        if split_name not in cache:
            cache[split_name] = load_hf_dataset(args.hf_config, split_name)
        return cache[split_name]

    lines_out: list[str] = []
    for rel, _nframes in zip(rel_paths, lengths):
        prefix, idx = parse_wav_index(rel)
        ds = get_ds(prefix)
        n = len(ds)
        if idx < 0 or idx >= n:
            print(
                f"error: index {idx} out of range for HF split "
                f"'{hf_split_name(prefix, args.train_split)}' (len={n}). "
                f"Check --train-split / --hf-config for file {rel!r}.",
                file=sys.stderr,
            )
            return 1
        text = ds[idx]["text"]
        phones = phones_for_text(text, g2p, vocab)
        lines_out.append(" ".join(phones))

    if args.dry_run:
        for i, line in enumerate(lines_out[:3]):
            print(f"[{i}] {line}")
        print(f"(dry-run: would write {len(lines_out)} lines to {out_path})")
        return 0

    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines_out)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
