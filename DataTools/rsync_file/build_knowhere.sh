#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
KNOWHERE_DIR="${SCRIPT_DIR}/knowhere"
KNOWHERE_INCLUDE_DIR="${KNOWHERE_DIR}/include"
KNOWHERE_LIBRARY="${KNOWHERE_DIR}/build/Release/libknowhere.so"
KNOWHERE_REMOTE_NAME="${KNOWHERE_REMOTE_NAME:-default-conan-local2}"
KNOWHERE_REMOTE_URL="${KNOWHERE_REMOTE_URL:-https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2}"

if [[ ! -d "${KNOWHERE_DIR}" ]]; then
    echo "[ERROR] Knowhere source directory not found: ${KNOWHERE_DIR}"
    exit 1
fi

if [[ -f "${KNOWHERE_LIBRARY}" ]]; then
    echo "[INFO] Knowhere library already exists: ${KNOWHERE_LIBRARY}"
    exit 0
fi

if ! command -v conan &> /dev/null; then
    echo "[ERROR] conan is not installed."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "[ERROR] cmake is not installed."
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "[ERROR] make is not installed."
    exit 1
fi

echo "[INFO] Ensuring Conan remote '${KNOWHERE_REMOTE_NAME}' is configured..."
conan remote add "${KNOWHERE_REMOTE_NAME}" "${KNOWHERE_REMOTE_URL}" --force >/dev/null 2>&1 || true

echo "[INFO] Building local knowhere release library..."
(
    cd "${KNOWHERE_DIR}"
    CMAKE_POLICY_VERSION_MINIMUM=3.5 make
)

if [[ ! -f "${KNOWHERE_LIBRARY}" ]]; then
    echo "[ERROR] Knowhere build finished but library is still missing: ${KNOWHERE_LIBRARY}"
    exit 1
fi

if [[ ! -f "${KNOWHERE_INCLUDE_DIR}/knowhere/index/index_factory.h" ]]; then
    echo "[ERROR] Knowhere include directory looks incomplete: ${KNOWHERE_INCLUDE_DIR}"
    exit 1
fi

echo "[INFO] Knowhere build completed:"
echo "       include=${KNOWHERE_INCLUDE_DIR}"
echo "       library=${KNOWHERE_LIBRARY}"
