#!/usr/bin/env bash
set -euo pipefail

# 批量断点续传 6 个数据集的 Index_parallel 目录到 170 服务器的 Index_parallel_187。
# 特性：
# 1) 断点续传（--partial --append-verify）
# 2) 远端目录不存在先创建
# 3) 仅首次输入一次密码（ssh 复用连接）
#
# 用法：
#   bash rsync_index_parallel_6datasets_187.sh
#
# 可覆盖参数（一般不需要）：
#   SRC_ROOT=/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults \
#   DST_USER=sunyahui DST_HOST=10.77.110.170 DST_PORT=22 \
#   DST_ROOT=/noraiddata/lijiakang/FilterVector/FilterVectorResults \
#   bash rsync_index_parallel_6datasets_187.sh

SRC_ROOT="${SRC_ROOT:-/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults}"
DST_USER="${DST_USER:-sunyahui}"
DST_HOST="${DST_HOST:-10.77.110.170}"
DST_PORT="${DST_PORT:-22}"
DST_ROOT="${DST_ROOT:-/noraiddata/lijiakang/FilterVector/FilterVectorResults}"

# 你给的密码。也可以在运行前 export SSHPASS=... 覆盖。
SSHPASS="${SSHPASS:-Qq984110509!}"

DATASETS=(
  VariousImg
)

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] source root not found: ${SRC_ROOT}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found" >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "[ERROR] 需要 sshpass，但系统未安装。" >&2
  echo "        Ubuntu/Debian: sudo apt-get install -y sshpass" >&2
  exit 1
fi

if [[ -z "${SSHPASS}" ]]; then
  echo "[ERROR] SSHPASS 为空，请设置密码。" >&2
  exit 1
fi

# SSH 连接复用：只在第一次连远端时做密码认证
CONTROL_PATH="/tmp/ssh_mux_filtervector_%r@%h:%p"
SSH_BASE_OPTS=(
  -p "${DST_PORT}"
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto
  -o ControlPersist=8h
  -o ControlPath="${CONTROL_PATH}"
)

DST_LOGIN="${DST_USER}@${DST_HOST}"

cleanup() {
  ssh -O exit "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "SRC_ROOT=${SRC_ROOT}"
echo "DST=${DST_LOGIN}:${DST_ROOT}"
echo "PORT=${DST_PORT}"
echo "DATASETS=${DATASETS[*]}"
echo

echo "[INFO] 建立 SSH 复用主连接（此处只需要输入/使用一次密码）..."
sshpass -p "${SSHPASS}" ssh -Nf "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}"

echo "[INFO] 主连接已建立，开始同步。"

ok=0
failed=0

for ds in "${DATASETS[@]}"; do
  src="${SRC_ROOT}/${ds}/Index/"
  dst_dir="${DST_ROOT}/${ds}/Index_parallel_187"

  echo "=================================================="
  echo "[$(date '+%F %T')] Sync dataset: ${ds}"
  echo "SRC: ${src}"
  echo "DST: ${DST_LOGIN}:${dst_dir}/"

  if [[ ! -d "${SRC_ROOT}/${ds}/Index" ]]; then
    echo "[WARN] source not found, skip: ${SRC_ROOT}/${ds}/Index"
    failed=$((failed + 1))
    continue
  fi

  # 远端目录不存在先创建
  if ! ssh "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}" "mkdir -p '${dst_dir}'"; then
    echo "[FAIL] ${ds} (remote mkdir failed)"
    failed=$((failed + 1))
    continue
  fi

  # --partial + --append-verify: 断点续传并校验追加内容
  # --info=progress2: 展示整体进度
  if rsync -avz --partial --append-verify --info=progress2 \
      -e "ssh ${SSH_BASE_OPTS[*]}" \
      "${src}" "${DST_LOGIN}:${dst_dir}/"; then
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
