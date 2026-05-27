#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults}"
ANALYZER="${2:-/home/sunyahui/ljk/FilterVector/FilterVectorCode/UNG/build/apps/analyze_rabitq_side}"

DATASETS=(Amazon BookReviews Genome Music Reviews Tiktok VariousImg)

if [[ ! -x "$ANALYZER" ]]; then
  echo "[ERROR] analyzer not found or not executable: $ANALYZER"
  echo "Build it first: cmake --build UNG/build --target analyze_rabitq_side -j"
  exit 1
fi

updated=0
skipped=0
failed=0

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

refresh_ung_csvs() {
  local meta="$1"
  local meta_dir idx_dir others
  meta_dir="$(dirname "$meta")"
  idx_dir="$(dirname "$meta_dir")"
  others="$idx_dir/others"
  mkdir -p "$others"

  local rabitq_build_requested rabitq_enabled rabitq_total_bits rabitq_build_time_ms rabitq_side_size_bytes
  local index_time_without_rabitq_ms index_time_with_rabitq_ms
  local index_size_without_rabitq_mb index_size_with_rabitq_mb

  rabitq_build_requested="$(meta_get "$meta" "rabitq_build_requested")"
  rabitq_enabled="$(meta_get "$meta" "rabitq_enabled")"
  rabitq_total_bits="$(meta_get "$meta" "rabitq_total_bits")"
  rabitq_build_time_ms="$(meta_get "$meta" "rabitq_build_time(ms)")"
  rabitq_side_size_bytes="$(meta_get "$meta" "rabitq_side_size_bytes")"
  index_time_without_rabitq_ms="$(meta_get "$meta" "index_time_without_rabitq(ms)")"
  index_time_with_rabitq_ms="$(meta_get "$meta" "index_time_with_rabitq(ms)")"
  index_size_without_rabitq_mb="$(meta_get "$meta" "index_size_without_rabitq(MB)")"
  index_size_with_rabitq_mb="$(meta_get "$meta" "index_size_with_rabitq(MB)")"

  local stats_csv="$others/ung_build_stats.csv"
  {
    echo "rabitq_build_requested,rabitq_enabled,rabitq_total_bits,rabitq_build_time_ms,rabitq_side_size_bytes,index_time_without_rabitq_ms,index_time_with_rabitq_ms,index_size_without_rabitq_mb,index_size_with_rabitq_mb"
    echo "${rabitq_build_requested},${rabitq_enabled},${rabitq_total_bits},${rabitq_build_time_ms},${rabitq_side_size_bytes},${index_time_without_rabitq_ms},${index_time_with_rabitq_ms},${index_size_without_rabitq_mb},${index_size_with_rabitq_mb}"
  } > "$stats_csv"

  local compare_csv="$others/ung_build_compare_stats.csv"
  {
    echo "variant,build_time_ms,index_size_mb,rabitq_build_time_ms,rabitq_side_size_bytes"
    echo "exact,${index_time_without_rabitq_ms},${index_size_without_rabitq_mb},0,0"
    if [[ "${rabitq_enabled}" == "1" ]]; then
      echo "rabitq,${index_time_with_rabitq_ms},${index_size_with_rabitq_mb},${rabitq_build_time_ms},${rabitq_side_size_bytes}"
    fi
  } > "$compare_csv"
}

for ds in "${DATASETS[@]}"; do
  base="$ROOT/$ds"
  if [[ ! -d "$base" ]]; then
    echo "[WARN] dataset dir missing: $base"
    continue
  fi

  while IFS= read -r side; do
    meta_dir="$(dirname "$side")"
    meta="$meta_dir/meta"
    if [[ ! -f "$meta" ]]; then
      echo "[SKIP] meta missing for side file: $side"
      skipped=$((skipped + 1))
      continue
    fi
    if "$ANALYZER" --side_file "$side" --meta_file "$meta" --update_meta true >/dev/null; then
      refresh_ung_csvs "$meta"
      echo "[OK] updated $meta"
      updated=$((updated + 1))
    else
      echo "[FAIL] $side"
      failed=$((failed + 1))
    fi
  done < <(find "$base" -type f -name "rabitq_side.bin" 2>/dev/null | sort)
done

echo "done: updated=$updated skipped=$skipped failed=$failed"
