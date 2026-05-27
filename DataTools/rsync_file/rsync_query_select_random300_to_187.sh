#!/usr/bin/env bash
set -euo pipefail

# 批量断点续传各数据集下的 query_select_*_random_300 目录到 10.77.110.187。
#
# 默认会同步如下路径：
#   ${SRC_ROOT}/${dataset}/${SUBDIR_NAME}/
# 到：
#   ${DST_ROOT}/${dataset}/${SUBDIR_NAME}/
#
# 用法示例：
# 1) 使用脚本内默认数据集列表
#    bash rsync_query_select_random300_to_187.sh
#
# 2) 指定数据集（空格分隔）
#    DATASETS="Amazon BookReviews Genome" bash rsync_query_select_random300_to_187.sh
#
# 3) 覆盖源/目标目录
#    SRC_ROOT=/mnt/disk1/syh/ljk/FilterVector/FilterVectorData \
#    DST_ROOT=/home/fengxiaoyao/FilterVector/FilterVectorData \
#    bash rsync_query_select_random300_to_187.sh

SRC_ROOT="${SRC_ROOT:-/home/fengxiaoyao/FilterVector/FilterVectorData}"
DST_USER="${DST_USER:-sunyahui}"
DST_HOST="${DST_HOST:-10.77.110.187}"
DST_PORT="${DST_PORT:-22}"
DST_ROOT="${DST_ROOT:-/mnt/disk1/syh/ljk/FilterVector/FilterVectorData}"

# 你的目录名模板（可按需覆盖）
SUBDIR_NAME="${SUBDIR_NAME:-query_select_200_A_B_C-sub-base-123456789_random_300}"

# 默认数据集列表；也可用环境变量 DATASETS 覆盖
DEFAULT_DATASETS=(
  Amazon
  BookReviews
  Genome
  Laion
  Music
  Reviews
  Tiktok
  VariousImg
)

# 允许用空格分隔覆盖，例如：DATASETS="Amazon Genome"
if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_ARR=(${DATASETS})
else
  DATASET_ARR=("${DEFAULT_DATASETS[@]}")
fi

# 密码可在运行前 export SSHPASS 覆盖
SSHPASS="${SSHPASS:-Qq984110509!}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] source root not found: ${SRC_ROOT}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found" >&2
  exit 1
fi

HAS_SSHPASS=0
if command -v sshpass >/dev/null 2>&1; then
  HAS_SSHPASS=1
else
  echo "[WARN] 未检测到 sshpass，将使用交互式 SSH 登录（输入一次密码后复用连接）。"
  echo "       如需安装 sshpass："
  echo "       CentOS 7: sudo yum install -y epel-release && sudo yum install -y sshpass"
  echo "       Ubuntu/Debian: sudo apt-get install -y sshpass"
fi

DST_LOGIN="${DST_USER}@${DST_HOST}"

# SSH 连接复用，避免每个数据集都重复认证
CONTROL_PATH="/tmp/ssh_mux_filtervector_%r@%h:%p"
SSH_BASE_OPTS=(
  -p "${DST_PORT}"
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto
  -o ControlPersist=8h
  -o ControlPath="${CONTROL_PATH}"
)

# CentOS 7 常见旧版 OpenSSH 不支持 StrictHostKeyChecking=accept-new，
# 自动降级为 no 以保证脚本可运行。
if ! ssh -G "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}" >/dev/null 2>&1; then
  echo "[WARN] 当前 ssh 不支持 StrictHostKeyChecking=accept-new，降级为 StrictHostKeyChecking=no"
  SSH_BASE_OPTS=(
    -p "${DST_PORT}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o ControlMaster=auto
    -o ControlPersist=8h
    -o ControlPath="${CONTROL_PATH}"
  )
fi

cleanup() {
  ssh -O exit "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "SRC_ROOT=${SRC_ROOT}"
echo "DST=${DST_LOGIN}:${DST_ROOT}"
echo "PORT=${DST_PORT}"
echo "SUBDIR_NAME=${SUBDIR_NAME}"
echo "DATASETS=${DATASET_ARR[*]}"
echo

echo "[INFO] 建立 SSH 复用主连接（只需认证一次）..."
if [[ "${HAS_SSHPASS}" -eq 1 ]]; then
  if [[ -z "${SSHPASS}" ]]; then
    echo "[ERROR] 已检测到 sshpass，但 SSHPASS 为空，请设置密码。" >&2
    exit 1
  fi
  sshpass -p "${SSHPASS}" ssh -Nf "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}"
else
  echo "[INFO] 请按提示输入 ${DST_LOGIN} 的 SSH 密码..."
  ssh -Nf "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}"
fi

echo "[INFO] 主连接已建立，开始同步。"

ok=0
failed=0

for ds in "${DATASET_ARR[@]}"; do
  src="${SRC_ROOT}/${ds}/${SUBDIR_NAME}/"
  dst_dir="${DST_ROOT}/${ds}/${SUBDIR_NAME}"

  echo "=================================================="
  echo "[$(date '+%F %T')] Sync dataset: ${ds}"
  echo "SRC: ${src}"
  echo "DST: ${DST_LOGIN}:${dst_dir}/"

  if [[ ! -d "${SRC_ROOT}/${ds}/${SUBDIR_NAME}" ]]; then
    echo "[WARN] source not found, skip: ${SRC_ROOT}/${ds}/${SUBDIR_NAME}"
    failed=$((failed + 1))
    continue
  fi

  if ! ssh "${SSH_BASE_OPTS[@]}" "${DST_LOGIN}" "mkdir -p '${dst_dir}'"; then
    echo "[FAIL] ${ds} (remote mkdir failed)"
    failed=$((failed + 1))
    continue
  fi

  # 断点续传 + 校验追加部分 + 显示总体进度
  # 增加简单重试，降低网络抖动影响
  attempt=1
  max_retry=3
  while true; do
    if rsync -avz --partial --append-verify --info=progress2 \
      --exclude "query_rabitq_ctx_cache.bin" \
      -e "ssh ${SSH_BASE_OPTS[*]}" \
      "${src}" "${DST_LOGIN}:${dst_dir}/"; then
      echo "[OK] ${ds}"
      ok=$((ok + 1))
      break
    fi

    rc=$?
    if (( attempt >= max_retry )); then
      echo "[FAIL] ${ds} (exit=${rc}, retries=${attempt})"
      failed=$((failed + 1))
      break
    fi

    echo "[WARN] ${ds} rsync failed (exit=${rc}), retry ${attempt}/${max_retry} after 5s..."
    sleep 5
    attempt=$((attempt + 1))
  done

done

echo "=================================================="
echo "Done. success=${ok}, failed=${failed}"

if [[ ${failed} -gt 0 ]]; then
  exit 1
fi
