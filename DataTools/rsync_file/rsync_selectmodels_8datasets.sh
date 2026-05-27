#!/usr/bin/env bash
set -u

# 批量断点续传 smart_route 文件夹（8 个数据集）
# 用法：
#   bash rsync_selectmodels_8datasets.sh
#   或自定义参数：
#   SRC_ROOT=/path/from DST_HOST=user@ip DST_ROOT=/path/to bash rsync_selectmodels_8datasets.sh

SRC_ROOT="${SRC_ROOT:-/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults}"
DST_HOST="${DST_HOST:-lijiakang@202.112.113.214}"
DST_ROOT="${DST_ROOT:-/home/lijiakang/FilterVector/FilterVectorResults}"

# 按需改这里的数据集列表
DATASETS=(
  Amazon
  BookReviews
  Genome
  Laion
  Music
  Reviews
  Tiktok
  VariousImg
)

SRC_SUFFIX="SelectModels/smart_route"
DST_SUFFIX="SelectModels"

ok=0
failed=0

for ds in "${DATASETS[@]}"; do
  src="${SRC_ROOT}/${ds}/${SRC_SUFFIX}"
  dst_parent="${DST_ROOT}/${ds}/${DST_SUFFIX}"
  dst_dir="${dst_parent}/smart_route"
  dst="${DST_HOST}:${dst_dir}"

  echo "=================================================="
  echo "[$(date '+%F %T')] Sync dataset: ${ds}"
  echo "SRC: ${src}"
  echo "DST: ${dst}"

  if [ ! -d "${src}" ]; then
    echo "[WARN] source not found, skip: ${src}"
    failed=$((failed + 1))
    continue
  fi

  # 先在远端显式创建 smart_route 目录，保证目标路径结构固定。
  ssh "${DST_HOST}" "mkdir -p '${dst_dir}'"
  mkdir_rc=$?
  if [ ${mkdir_rc} -ne 0 ]; then
    echo "[FAIL] ${ds} (remote mkdir exit=${mkdir_rc})"
    failed=$((failed + 1))
    continue
  fi

  # --partial + --append-verify: 断点续传并校验追加部分
  # --info=progress2: 总体进度
  # 源路径加 /，表示同步 smart_route 目录内容到目标 smart_route 目录
  rsync -avz --partial --append-verify --info=progress2 \
    "${src}/" \
    "${dst}"

  rc=$?
  if [ ${rc} -eq 0 ]; then
    echo "[OK] ${ds}"
    ok=$((ok + 1))
  else
    echo "[FAIL] ${ds} (exit=${rc})"
    failed=$((failed + 1))
  fi

done

echo "=================================================="
echo "Done. success=${ok}, failed=${failed}"

if [ ${failed} -gt 0 ]; then
  exit 1
fi
