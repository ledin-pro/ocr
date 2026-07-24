#!/usr/bin/env bash
# probe.sh — OCR triage for a single file
# Usage: bash probe.sh <FILE>
# Output: single-line JSON to stdout
# Exit codes: 0=ok, 1=missing arg, 2=file not found, 4=missing binary

set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "$1" >&2; exit "${2:-1}"; }

require_binary() {
  command -v "$1" >/dev/null 2>&1 || die "Required binary '$1' not found. Install: $2" 4
}

json_bool() { [[ "$1" == "true" ]] && echo "true" || echo "false"; }

# Simple JSON string escape (handles backslash and double-quote)
json_str() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  echo "\"$s\""
}

# ── arg check ────────────────────────────────────────────────────────────────

[[ $# -ge 1 ]] || die "Usage: bash probe.sh <FILE>" 1
FILE="$1"
[[ -f "$FILE" ]] || die "File not found: $FILE" 2

# ── binary check ─────────────────────────────────────────────────────────────

require_binary pdfinfo  "brew install poppler  OR  sudo apt install poppler-utils"
require_binary pdftotext "brew install poppler OR  sudo apt install poppler-utils"
require_binary pdffonts  "brew install poppler OR  sudo apt install poppler-utils"
require_binary pdfimages "brew install poppler OR  sudo apt install poppler-utils"

# ── extension classification ──────────────────────────────────────────────────

EXT="${FILE##*.}"
EXT="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"

IMAGE_EXTS="png jpg jpeg tiff tif heic webp bmp gif"
is_image=false
for e in $IMAGE_EXTS; do
  [[ "$EXT" == "$e" ]] && is_image=true && break
done

if [[ "$is_image" == "true" ]]; then
  printf '{"input_type":"image","pages":1,"encrypted":false,"per_page_chars":[0],"median_chars":0,"has_text_layer":false,"suspected_rasterized_text":false,"needs_ocr":true,"reason":"image file — OCR directly"}\n'
  exit 0
fi

if [[ "$EXT" != "pdf" ]]; then
  printf '{"input_type":"unsupported","pages":0,"encrypted":false,"per_page_chars":[],"median_chars":0,"has_text_layer":false,"suspected_rasterized_text":false,"needs_ocr":false,"reason":"unsupported file type: %s"}\n' "$EXT"
  exit 3
fi

# ── PDF metadata ──────────────────────────────────────────────────────────────

PDFINFO_OUT=$(pdfinfo "$FILE" 2>/dev/null) || PDFINFO_OUT=""

PAGES=$(echo "$PDFINFO_OUT" | grep -i "^Pages:" | awk '{print $2}')
PAGES="${PAGES:-0}"

ENCRYPTED_RAW=$(echo "$PDFINFO_OUT" | grep -i "^Encrypted:" | awk '{print $2}')
ENCRYPTED=false
ENCRYPTED_RAW_LC="$(echo "$ENCRYPTED_RAW" | tr '[:upper:]' '[:lower:]')"
[[ "$ENCRYPTED_RAW_LC" == "yes" ]] && ENCRYPTED=true

if [[ "$ENCRYPTED" == "true" ]]; then
  printf '{"input_type":"pdf","pages":%s,"encrypted":true,"per_page_chars":[],"median_chars":0,"has_text_layer":false,"suspected_rasterized_text":false,"needs_ocr":false,"reason":"PDF is encrypted — decrypt first (qpdf --decrypt)"}\n' "$PAGES"
  exit 0
fi

# ── per-page character count ──────────────────────────────────────────────────

CHAR_COUNTS=()
for (( p=1; p<=PAGES; p++ )); do
  # Extract text for page p, strip all whitespace, count remaining bytes
  COUNT=$(pdftotext -layout -f "$p" -l "$p" "$FILE" - 2>/dev/null \
          | tr -d '[:space:]' | wc -c | tr -d ' ')
  CHAR_COUNTS+=("${COUNT:-0}")
done

# Median of CHAR_COUNTS (sort numerically, pick middle element)
SORTED=( $(printf '%s\n' "${CHAR_COUNTS[@]}" | sort -n) )
MID_IDX=$(( ${#SORTED[@]} / 2 ))
MEDIAN="${SORTED[$MID_IDX]:-0}"

# Build JSON array string
CHAR_ARRAY="["
for i in "${!CHAR_COUNTS[@]}"; do
  [[ $i -gt 0 ]] && CHAR_ARRAY+=","
  CHAR_ARRAY+="${CHAR_COUNTS[$i]}"
done
CHAR_ARRAY+="]"

# ── font analysis ─────────────────────────────────────────────────────────────

FONT_OUT=$(pdffonts "$FILE" 2>/dev/null) || FONT_OUT=""

# Count fonts with uni=no (columns: name type encoding emb sub uni object)
# pdffonts outputs 'yes'/'no' in column 6 (uni) after a header line
NONUNI_COUNT=$(echo "$FONT_OUT" | tail -n +3 | awk '{print $6}' | grep -c "^no$" || true)
TOTAL_FONTS=$(echo "$FONT_OUT" | tail -n +3 | grep -c "." || true)

SUSPECTED_RASTERIZED=false
RASTER_REASON=""
if [[ "$TOTAL_FONTS" -gt 0 && "$NONUNI_COUNT" -gt 0 ]]; then
  SUSPECTED_RASTERIZED=true
  RASTER_REASON="$NONUNI_COUNT/$TOTAL_FONTS fonts non-unicode (uni=no)"
fi
# Also flag if zero fonts found — fully image-based PDF
if [[ "$TOTAL_FONTS" -eq 0 ]]; then
  SUSPECTED_RASTERIZED=true
  RASTER_REASON="no fonts found — likely fully image-based PDF"
fi

# ── image coverage analysis ───────────────────────────────────────────────────

IMAGE_OUT=$(pdfimages -list "$FILE" 2>/dev/null | tail -n +3) || IMAGE_OUT=""

# Count smask entries (soft masks = rasterized text/transparency on image layers)
SMASK_COUNT=$(echo "$IMAGE_OUT" | awk '{print $3}' | grep -c "^smask$" || true)
TOTAL_IMAGES=$(echo "$IMAGE_OUT" | grep -c "." || true)

HIGH_IMAGE_COVERAGE=false
IMAGE_REASON=""
if [[ "$SMASK_COUNT" -ge 3 ]]; then
  HIGH_IMAGE_COVERAGE=true
  IMAGE_REASON="$SMASK_COUNT full-page image masks (smask) detected"
fi

# ── decision ──────────────────────────────────────────────────────────────────

NEEDS_OCR=false
HAS_TEXT_LAYER=false
REASON_PARTS=()

# Primary signal: character yield per page.
# >= 30 chars AND either no rasterized-font signal OR high image masks → text layer.
# < 30 chars AND (rasterized fonts OR high image masks) → needs OCR.
if [[ "$MEDIAN" -ge 30 && "$HIGH_IMAGE_COVERAGE" == "false" ]]; then
  HAS_TEXT_LAYER=true
  REASON_PARTS+=("median ${MEDIAN} chars/page — real text layer present")
elif [[ "$MEDIAN" -ge 30 && "$SUSPECTED_RASTERIZED" == "false" ]]; then
  HAS_TEXT_LAYER=true
  REASON_PARTS+=("median ${MEDIAN} chars/page — real text layer present")
else
  NEEDS_OCR=true
  REASON_PARTS+=("median ${MEDIAN} non-space chars/page (threshold: 30)")
fi

# Additional signals only push needs_ocr=true if text layer not confirmed
if [[ "$HAS_TEXT_LAYER" == "false" && "$SUSPECTED_RASTERIZED" == "true" ]]; then
  NEEDS_OCR=true
  REASON_PARTS+=("$RASTER_REASON")
fi

if [[ "$HAS_TEXT_LAYER" == "false" && "$HIGH_IMAGE_COVERAGE" == "true" ]]; then
  NEEDS_OCR=true
  REASON_PARTS+=("$IMAGE_REASON")
fi

[[ ${#REASON_PARTS[@]} -eq 0 ]] && REASON_PARTS+=("no issues detected")

# Join reason parts with "; "
REASON="${REASON_PARTS[0]}"
for (( i=1; i<${#REASON_PARTS[@]}; i++ )); do
  REASON+="; ${REASON_PARTS[$i]}"
done

# ── emit JSON ────────────────────────────────────────────────────────────────

printf '{"input_type":"pdf","pages":%s,"encrypted":%s,"per_page_chars":%s,"median_chars":%s,"has_text_layer":%s,"suspected_rasterized_text":%s,"needs_ocr":%s,"reason":%s}\n' \
  "$PAGES" \
  "$(json_bool "$ENCRYPTED")" \
  "$CHAR_ARRAY" \
  "$MEDIAN" \
  "$(json_bool "$HAS_TEXT_LAYER")" \
  "$(json_bool "$SUSPECTED_RASTERIZED")" \
  "$(json_bool "$NEEDS_OCR")" \
  "$(json_str "$REASON")"
