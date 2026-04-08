#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Download LibriLight unlabeled audio subsets (official libri-light data release).

Source: https://dl.fbaipublicfiles.com/librilight/data/ (see facebookresearch/libri-light data_preparation README).

Usage:
  ./download_librilight.sh [--out-dir PATH] [--subset LIST] [--keep-archives] [--verify-md5]

Options:
  --out-dir PATH      Destination directory (default: ./data/librilight)
  --subset LIST       Comma-separated: small,medium,large,all (default: small)
  --keep-archives     Keep downloaded .tar files after extraction
  --verify-md5        Verify each archive against published checksums after download (slow on large.tar)
  -h, --help          Show this help message

Examples:
  ./download_librilight.sh --subset small
  ./download_librilight.sh --subset small,medium --out-dir ./data/librilight
  ./download_librilight.sh --subset all

Note: "all" fetches small (~35 GiB) + medium (~321 GiB) + large (~3 TiB) before extraction; reserve enough disk space.
EOF
}

OUT_DIR="./data/librilight"
SUBSET_LIST="small"
KEEP_ARCHIVES=0
VERIFY_MD5=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --subset)
      SUBSET_LIST="${2:-}"
      shift 2
      ;;
    --keep-archives)
      KEEP_ARCHIVES=1
      shift
      ;;
    --verify-md5)
      VERIFY_MD5=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${OUT_DIR}" || -z "${SUBSET_LIST}" ]]; then
  echo "Error: --out-dir and --subset require values."
  exit 1
fi

BASE_URL="https://dl.fbaipublicfiles.com/librilight/data"

# Published MD5 sums (libri-light data_preparation README).
declare -A SUBSET_MD5=(
  [small]=c49207eb86a8e8ac895561c37232041e
  [medium]=c75e7ac62471bfbf2db77528d62a9b74
  [large]=4dfbac018f50b99797ece101fc9f0c30
)

normalize_subsets() {
  local raw="$1"
  local cleaned="${raw// /}"
  if [[ "$cleaned" == "all" ]]; then
    echo "small medium large"
    return 0
  fi

  local result=()
  IFS=',' read -r -a parts <<< "$cleaned"
  for part in "${parts[@]}"; do
    case "$part" in
      small|medium|large)
        result+=("$part")
        ;;
      *)
        echo "Invalid subset: '$part'. Allowed values: small, medium, large, all." >&2
        return 1
        ;;
    esac
  done

  # Deduplicate while preserving order.
  local unique=()
  local seen=""
  for s in "${result[@]}"; do
    if [[ " $seen " != *" $s "* ]]; then
      unique+=("$s")
      seen="$seen $s"
    fi
  done

  echo "${unique[*]}"
}

download_file() {
  local url="$1"
  local out_file="$2"
  echo "Downloading: $url"

  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=0 --read-timeout=60 --progress=bar:force:noscroll "$url" -O "$out_file"
  elif command -v curl >/dev/null 2>&1; then
    curl -fSL --connect-timeout 60 --retry 5 --retry-delay 10 -C - "$url" -o "$out_file"
  else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
  fi
}

verify_md5() {
  local file_path="$1"
  local expected="$2"
  local actual=""

  if command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "$file_path" | awk '{print $1}')"
  elif command -v md5 >/dev/null 2>&1; then
    actual="$(md5 -q "$file_path")"
  else
    echo "Error: --verify-md5 requires md5sum or md5." >&2
    exit 1
  fi

  if [[ "$actual" != "$expected" ]]; then
    echo "MD5 mismatch for $file_path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
  echo "MD5 OK: $file_path"
}

extract_archive() {
  local archive="$1"
  local target_dir="$2"
  echo "Extracting: $archive"
  tar -xf "$archive" -C "$target_dir"
}

if ! SUBSETS="$(normalize_subsets "$SUBSET_LIST")"; then
  exit 1
fi

mkdir -p "$OUT_DIR/archives"

echo "Output directory: $OUT_DIR"
echo "Subsets: $SUBSETS"

for subset in $SUBSETS; do
  archive_name="${subset}.tar"
  url="${BASE_URL}/${archive_name}"
  archive_path="$OUT_DIR/archives/$archive_name"

  download_file "$url" "$archive_path"

  if [[ "$VERIFY_MD5" -eq 1 ]]; then
    verify_md5 "$archive_path" "${SUBSET_MD5[$subset]}"
  fi

  extract_archive "$archive_path" "$OUT_DIR"

  if [[ "$KEEP_ARCHIVES" -eq 0 ]]; then
    rm -f "$archive_path"
  fi
done

echo "Done. LibriLight data is available under: $OUT_DIR"
