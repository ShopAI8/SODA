#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults}"
ANALYZER="${2:-/home/sunyahui/ljk/FilterVector/FilterVectorCode/ACORN/build_local/demos/analyze_acorn_rabitq_side}"
DATASETS=(Amazon BookReviews Genome Music Reviews Tiktok VariousImg)

if [[ ! -x "$ANALYZER" ]]; then
  echo "[ERROR] analyzer not found or not executable: $ANALYZER"
  echo "Build it first: cmake --build ACORN/build_local --target analyze_acorn_rabitq_side -j"
  exit 1
fi

meta_get() {
  local meta_file="$1"
  local key="$2"
  awk -v k="$key" '
    {
      line=$0
      sub(/\r$/, "", line)
      pos=index(line, ":")
      if (pos == 0) pos=index(line, "=")
      if (pos > 0) {
        lhs=substr(line, 1, pos - 1)
        rhs=substr(line, pos + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", lhs)
        gsub(/^[ \t]+|[ \t]+$/, "", rhs)
        if (lhs == k) { print rhs; exit }
      }
    }
  ' "$meta_file" 2>/dev/null || true
}

append_row() {
  local variant="$1" meta="$2" csv="$3"
  [[ -f "$meta" ]] || return 0
  local build_time_s total_size_bytes index_only_size_bytes total_logical_memory_bytes index_only_logical_memory_bytes
  build_time_s="$(meta_get "$meta" "total_build_time_s")"
  [[ -n "$build_time_s" ]] || build_time_s="$(meta_get "$meta" "build_time_s")"
  total_size_bytes="$(meta_get "$meta" "total_size_bytes")"
  index_only_size_bytes="$(meta_get "$meta" "index_only_size_bytes")"
  total_logical_memory_bytes="$(meta_get "$meta" "total_logical_memory_bytes")"
  index_only_logical_memory_bytes="$(meta_get "$meta" "index_only_logical_memory_bytes")"
  echo "${variant},${build_time_s},${total_size_bytes},${index_only_size_bytes},${total_logical_memory_bytes},${index_only_logical_memory_bytes}" >> "$csv"
}

refresh_acorn_csv() {
  local acorn_output_dir="$1"
  local idx_dir others csv
  idx_dir="$(dirname "$acorn_output_dir")"
  others="$idx_dir/others"
  mkdir -p "$others"
  csv="$others/acorn_build_stats.csv"
  echo "variant,build_time_s,total_size_bytes,index_only_size_bytes,total_logical_memory_bytes,index_only_logical_memory_bytes" > "$csv"
  append_row "acorn_gamma_exact" "$acorn_output_dir/acorn.index.meta" "$csv"
  append_row "acorn_1_exact" "$acorn_output_dir/acorn1.index.meta" "$csv"
  append_row "acorn_gamma_rabitq" "$acorn_output_dir/acorn_rabitq.index.meta" "$csv"
  append_row "acorn_1_rabitq" "$acorn_output_dir/acorn1_rabitq.index.meta" "$csv"
}

updated=0
skipped=0
failed=0
declare -A touched_dirs=()

for ds in "${DATASETS[@]}"; do
  base="$ROOT/$ds"
  [[ -d "$base" ]] || { echo "[WARN] dataset dir missing: $base"; continue; }

  while IFS= read -r side; do
    acorn_output_dir="$(dirname "$side")"
    fname="$(basename "$side")"
    stem="${fname%.rabitq_side.bin}"
    meta="$acorn_output_dir/${stem}.meta"
    if [[ ! -f "$meta" ]]; then
      echo "[SKIP] meta missing for side file: $side"
      skipped=$((skipped + 1))
      continue
    fi
    if "$ANALYZER" --side_file "$side" --meta_file "$meta" --update_meta true >/dev/null; then
      echo "[OK] updated $meta"
      updated=$((updated + 1))
      touched_dirs["$acorn_output_dir"]=1
    else
      echo "[FAIL] $side"
      failed=$((failed + 1))
    fi
  done < <(find "$base" -type f -name "*.rabitq_side.bin" -path "*/acorn_output/*" 2>/dev/null | sort)
done

for d in "${!touched_dirs[@]}"; do
  refresh_acorn_csv "$d"
done

echo "done: updated=$updated skipped=$skipped failed=$failed refreshed_csv_dirs=${#touched_dirs[@]}"

