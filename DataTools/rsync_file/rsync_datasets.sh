#!/usr/bin/env bash
set -euo pipefail

# 批量断点续传 FilterVectorData 下的数据集目录。
# 数据集列表在脚本中固定配置，按需直接修改 DATASETS 数组。
#
# 用法：
#   1) 按脚本内 DATASETS 列表同步
#      bash rsync_datasets.sh
#
#   2) 覆盖默认源/目标
#      SRC_ROOT=/mnt/disk1/syh/ljk/FilterVector/FilterVectorData \
#      DST_HOST=lijiakang@202.112.113.214 \
#      DST_ROOT=/home/lijiakang/FilterVector/FilterVectorData \
#      bash rsync_datasets.sh

SRC_ROOT="${SRC_ROOT:-/mnt/disk1/syh/ljk/FilterVector/FilterVectorData}"
DST_HOST="${DST_HOST:-lijiakang@202.112.113.214}"
DST_ROOT="${DST_ROOT:-/home/lijiakang/FilterVector/FilterVectorData}"

# 在这里指定要同步的数据集名称（目录名）
DATASETS=(
  Amazon
  BookReviews
  Genome
  Music
  Reviews
  Tiktok
)

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] source root not found: ${SRC_ROOT}" >&2
  exit 1
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[WARN] no datasets to sync."
  exit 0
fi

ok=0
failed=0

echo "SRC_ROOT=${SRC_ROOT}"
echo "DST_HOST=${DST_HOST}"
echo "DST_ROOT=${DST_ROOT}"
echo "DATASETS=${DATASETS[*]}"

# 是否启用密码模式（默认开启）。若你使用免密登录，可设为 USE_PASSWORD_AUTH=0
USE_PASSWORD_AUTH="${USE_PASSWORD_AUTH:-1}"
RSYNC_SSH_OPTS="${RSYNC_SSH_OPTS:- -o StrictHostKeyChecking=accept-new }"
RSYNC_BASE_CMD=(rsync -avzP -e "ssh${RSYNC_SSH_OPTS}")

if [[ "${USE_PASSWORD_AUTH}" == "1" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[ERROR] USE_PASSWORD_AUTH=1 需要 sshpass，但系统未安装。" >&2
    echo "        可执行: sudo apt-get install -y sshpass" >&2
    echo "        或者设置 USE_PASSWORD_AUTH=0 使用免密登录。" >&2
    exit 1
  fi

  if [[ -z "${SSHPASS:-}" ]]; then
    read -rsp "请输入 ${DST_HOST} 的 SSH 密码: " SSHPASS
    echo
    export SSHPASS
  fi
fi

for ds in "${DATASETS[@]}"; do
  src="${SRC_ROOT}/${ds}/"
  dst="${DST_HOST}:${DST_ROOT}/${ds}/"

  echo "=================================================="
  echo "[$(date '+%F %T')] Sync dataset: ${ds}"
  echo "SRC: ${src}"
  echo "DST: ${dst}"

  if [[ ! -d "${SRC_ROOT}/${ds}" ]]; then
    echo "[WARN] source dataset not found, skip: ${SRC_ROOT}/${ds}"
    failed=$((failed + 1))
    continue
  fi

  if [[ "${USE_PASSWORD_AUTH}" == "1" ]]; then
    if sshpass -e "${RSYNC_BASE_CMD[@]}" "${src}" "${dst}"; then
      echo "[OK] ${ds}"
      ok=$((ok + 1))
    else
      rc=$?
      echo "[FAIL] ${ds} (exit=${rc})"
      failed=$((failed + 1))
    fi
  elif "${RSYNC_BASE_CMD[@]}" "${src}" "${dst}"; then
    echo "[OK] ${ds}"
    ok=$((ok + 1))
  else
    rc=$?
    echo "[FAIL] ${ds} (exit=${rc})"
    failed=$((failed + 1))
  fi

done

echo "=================================================="
echo "Done. success=${ok}, failed=${failed}"

if [[ ${failed} -gt 0 ]]; then
  exit 1
fi
