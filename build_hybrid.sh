#!/bin/bash

# ==============================================================================
# build_hybrid.sh - 统一构建 UNG、ACORN 和 NaviX 索引
#
# 功能:
# 1. 编译 UNG、ACORN 和 NaviX 的代码
# 2. 检查并转换 fvecs -> bin 数据格式
# 3. 根据 build_mode 构建索引(parallel, serial, ung_only, acorn_only, navix_only)
# ==============================================================================

set -e # 如果任何命令失败，则立即退出
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Step 1: 解析命令行参数 ---
PARAMS=("$@")
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        
        # 处理模式参数
        if [[ $key == "BUILD_MODE" ]]; then
            BUILD_MODE="$2"
            shift 2
            continue
        fi

        # 处理其他参数
        if [ -z "$2" ]; then
            echo "错误: 参数 $1 缺少值"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "未知参数: $1"; exit 1
    fi
done

# 验证构建模式
case "$BUILD_MODE" in
    parallel|serial|ung_only|acorn_only|navix_only|all)
        echo "[INFO] Build mode set to: $BUILD_MODE"
        ;;
    *)
        echo "错误: 无效的 build_mode '$BUILD_MODE'。可用选项: parallel, serial, ung_only, acorn_only, navix_only, all"
        exit 1
        ;;
esac

# --- Step 2: 编译代码 ---

# 仅在首次缺失可执行文件时编译 UNG，避免每次实验都清空并重编译。

# 1. 先编译 NaviX (因为 UNG 依赖它)
if [ -z "$NAVIX_BUILD_DIR" ]; then
    NAVIX_BUILD_DIR="${SCRIPT_DIR}/NaviX/build"
fi

# echo $NAVIX_BUILD_DIR
# whoami

if [ ! -f "${NAVIX_BUILD_DIR}/faiss_navix/libfaiss.a" ] && [ ! -f "${NAVIX_BUILD_DIR}/faiss_navix/libfaiss_avx2.a" ]; then
    echo "[INFO] NaviX library not found. Compiling..."
    rm -rf "$NAVIX_BUILD_DIR" 
    mkdir -p "$NAVIX_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/NaviX" -B "$NAVIX_BUILD_DIR" \
        -DFAISS_OPT_LEVEL=avx2 \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release
    make -C "$NAVIX_BUILD_DIR" -j
else
    echo "[INFO] NaviX library found."
fi

# 2. 再编译 UNG（强制要求 Knowhere）
if [[ -z "${KNOWHERE_INCLUDE_DIR:-}" || -z "${KNOWHERE_LIBRARY:-}" ]]; then
    echo "[ERROR] Knowhere is mandatory for Milvus baseline."
    echo "        Please export:"
    echo "          KNOWHERE_INCLUDE_DIR=/path/to/knowhere/include"
    echo "          KNOWHERE_LIBRARY=/path/to/libknowhere.so"
    exit 1
fi
KNOWHERE_BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/build_knowhere.sh"
LOCAL_KNOWHERE_LIBRARY="${SCRIPT_DIR}/knowhere/build/Release/libknowhere.so"
if [[ ! -f "${KNOWHERE_LIBRARY}" && "${KNOWHERE_LIBRARY}" == "${LOCAL_KNOWHERE_LIBRARY}" && -f "${KNOWHERE_BOOTSTRAP_SCRIPT}" ]]; then
    echo "[INFO] Local knowhere library not found. Bootstrapping build via ${KNOWHERE_BOOTSTRAP_SCRIPT} ..."
    bash "${KNOWHERE_BOOTSTRAP_SCRIPT}"
fi
if [[ ! -f "${KNOWHERE_INCLUDE_DIR}/knowhere/index/index_factory.h" ]]; then
    echo "[ERROR] Invalid KNOWHERE_INCLUDE_DIR: ${KNOWHERE_INCLUDE_DIR}"
    echo "        Missing: ${KNOWHERE_INCLUDE_DIR}/knowhere/index/index_factory.h"
    exit 1
fi
if [[ ! -f "${KNOWHERE_LIBRARY}" ]]; then
    echo "[ERROR] Invalid KNOWHERE_LIBRARY: ${KNOWHERE_LIBRARY}"
    exit 1
fi

echo "[INFO] Compiling UNG with mandatory Knowhere Milvus baseline enabled."
mkdir -p "$UNG_BUILD_DIR"
cmake -S "${SCRIPT_DIR}/UNG/codes" -B "$UNG_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DNAVIX_BUILD_DIR="$NAVIX_BUILD_DIR" \
    -DENABLE_KNOWHERE_MILVUS_BASELINE=ON \
    -DKNOWHERE_INCLUDE_DIR="${KNOWHERE_INCLUDE_DIR}" \
    -DKNOWHERE_LIBRARY="${KNOWHERE_LIBRARY}"
make -C "$UNG_BUILD_DIR" -j

# 3. 编译 ACORN
ACORN_EXECUTABLE="${ACORN_BUILD_DIR}/demos/test_acorn"
if [ ! -f "$ACORN_EXECUTABLE" ]; then
    echo "[INFO] ACORN executable not found. Compiling..."
    mkdir -p "$ACORN_BUILD_DIR"
    # cmake -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
     cmake -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -C "$ACORN_BUILD_DIR" -j test_acorn
else
    echo "[INFO] ACORN executable found."
fi

# --- Step 2.5: 检查并转换数据格式 ---
echo "[INFO] Checking and converting data format (if necessary)..."
FVECS_TO_BIN_TOOL="${UNG_BUILD_DIR}/tools/fvecs_to_bin"
BASE_FVECS_FILE="${DATA_DIR}/${DATASET}_base.fvecs"
BASE_BIN_FILE="${DATA_DIR}/${DATASET}_base.bin"

if [ ! -x "$FVECS_TO_BIN_TOOL" ]; then
    echo "Error: Conversion tool not found after compilation at: ${FVECS_TO_BIN_TOOL}"
    exit 1
fi

if [ ! -f "$BASE_BIN_FILE" ]; then
    if [ -f "$BASE_FVECS_FILE" ]; then
        echo "Target file '${BASE_BIN_FILE}' not found. Starting conversion from '${BASE_FVECS_FILE}'..."
        "$FVECS_TO_BIN_TOOL" --data_type float --input_file "$BASE_FVECS_FILE" --output_file "$BASE_BIN_FILE"
        echo "Conversion complete."
    else
        echo "Warning: Source file '${BASE_FVECS_FILE}' not found. Skipping conversion. Build process might fail if base .bin is also missing."
    fi
else
    echo "Target file '${BASE_BIN_FILE}' already exists. Skipping conversion."
fi


# --- Step 3: 构造与 search.sh 兼容的输出目录 ---
if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi
echo "[INFO] Index base directory set to: $INDEX_BASE_DIR"
INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
INDEX_DIR_NAME_EXACT="$INDEX_DIR_NAME"
if [[ "${BUILD_RABITQ_SIDE_INDEX}" == "true" ]]; then
    INDEX_DIR_NAME="${INDEX_DIR_NAME}_RQB${RABITQ_TOTAL_BITS:-4}"
fi
INDEX_OUTPUT_DIR="${EXP_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"
EXACT_INDEX_OUTPUT_DIR="${EXP_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME_EXACT}"

echo "[INFO] Preparing index directory (if not exists): $INDEX_OUTPUT_DIR"
mkdir -p "$INDEX_OUTPUT_DIR/index_files"
mkdir -p "$INDEX_OUTPUT_DIR/acorn_output"
mkdir -p "$INDEX_OUTPUT_DIR/navix_output"
mkdir -p "$INDEX_OUTPUT_DIR/others"

# Split exact/rabitq index dirs:
# - _RQB* dir stores RabitQ artifacts
# - non-_RQB dir stores exact artifacts
SPLIT_RABITQ_EXACT_DIRS="${SPLIT_RABITQ_EXACT_DIRS:-true}"
if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
    echo "[INFO] Split exact/rabitq dirs enabled."
    echo "[INFO]   exact dir : $EXACT_INDEX_OUTPUT_DIR"
    echo "[INFO]   rabitq dir: $INDEX_OUTPUT_DIR"
    mkdir -p "$EXACT_INDEX_OUTPUT_DIR/index_files"
    mkdir -p "$EXACT_INDEX_OUTPUT_DIR/acorn_output"
    mkdir -p "$EXACT_INDEX_OUTPUT_DIR/navix_output"
    mkdir -p "$EXACT_INDEX_OUTPUT_DIR/others"
fi

UNG_EXACT_OUTPUT_DIR="$INDEX_OUTPUT_DIR"
UNG_RABITQ_OUTPUT_DIR="$INDEX_OUTPUT_DIR"
if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
    UNG_EXACT_OUTPUT_DIR="$EXACT_INDEX_OUTPUT_DIR"
    UNG_RABITQ_OUTPUT_DIR="$INDEX_OUTPUT_DIR"
fi

# 定义标记文件路径 ---
UNG_MARKER_FILE="$UNG_EXACT_OUTPUT_DIR/index_files/.ung_built"
UNG_RABITQ_MARKER_FILE="$UNG_RABITQ_OUTPUT_DIR/index_files/.ung_rabitq_built"
ACORN_MARKER_FILE="$INDEX_OUTPUT_DIR/acorn_output/.acorn_built"
ACORN_RABITQ_MARKER_FILE="$INDEX_OUTPUT_DIR/acorn_output/.acorn_rabitq_built"
NAVIX_MARKER_FILE="$INDEX_OUTPUT_DIR/navix_output/.navix_built"

# Fair ACORN comparison switch:
# - true: exact and rabitq use the same input files; when both are built, run ACORN builds sequentially
# - false: keep previous behavior
ACORN_FAIR_COMPARE="${ACORN_FAIR_COMPARE:-true}"

# --- Step 4: 定义构建函数 ---

build_ung() {
    local variant="${1:-exact}"
    local target_dir marker_file ung_log_file ung_tag build_rabitq_flag
    if [[ "$variant" == "rabitq" ]]; then
        target_dir="$UNG_RABITQ_OUTPUT_DIR"
        marker_file="$UNG_RABITQ_MARKER_FILE"
        ung_log_file="$target_dir/others/ung_rabitq_build.log"
        ung_tag="UNG-RABITQ"
        build_rabitq_flag="true"
    else
        target_dir="$UNG_EXACT_OUTPUT_DIR"
        marker_file="$UNG_MARKER_FILE"
        ung_log_file="$target_dir/others/ung_build.log"
        ung_tag="UNG"
        build_rabitq_flag="false"
    fi

    if [ -f "$marker_file" ]; then
        echo "[$ung_tag] Marker file found ('$marker_file'). Skipping UNG build."
        return 0
    fi

    echo "[$ung_tag] Build process started."
    "$UNG_EXECUTABLE" \
        --data_type float --dist_fn L2 --num_threads 60 \
        --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
        --base_label_info_file "$DATA_DIR/${DATASET}_base_labels_info.log" \
        --base_label_tree_roots "$DATA_DIR/tree_roots.txt" \
        --index_path_prefix "$target_dir/index_files/" \
        --result_path_prefix "$target_dir/results/" \
        --build_rabitq_side_index "$build_rabitq_flag" \
        --rabitq_total_bits "${RABITQ_TOTAL_BITS:-4}" \
        --scenario general --dataset "$DATASET" \
        > "$ung_log_file" 2>&1

    echo "[$ung_tag] Build process finished."
    touch "$marker_file"
}

parse_meta_value() {
    local meta_file="$1"
    local key="$2"
    awk -v k="$key" '
        {
            line=$0
            sub(/\r$/, "", line)
            pos=index(line, ":")
            if (pos == 0) {
                pos=index(line, "=")
            }
            if (pos > 0) {
                lhs=substr(line, 1, pos - 1)
                rhs=substr(line, pos + 1)
                gsub(/^[ \t]+|[ \t]+$/, "", lhs)
                gsub(/^[ \t]+|[ \t]+$/, "", rhs)
                if (lhs == k) {
                    print rhs
                    exit
                }
            }
        }
    ' "$meta_file" 2>/dev/null || true
}

print_acorn_meta_summary() {
    local tag="$1"
    local meta_file="$2"
    if [[ ! -f "$meta_file" ]]; then
        echo "[$tag] Meta file not found: $meta_file"
        return 0
    fi
    local build_time_s total_size_bytes index_only_size_bytes total_logical_memory_bytes index_only_logical_memory_bytes
    build_time_s=$(parse_meta_value "$meta_file" "total_build_time_s")
    if [[ -z "$build_time_s" ]]; then
        build_time_s=$(parse_meta_value "$meta_file" "build_time_s")
    fi
    total_size_bytes=$(parse_meta_value "$meta_file" "total_size_bytes")
    index_only_size_bytes=$(parse_meta_value "$meta_file" "index_only_size_bytes")
    total_logical_memory_bytes=$(parse_meta_value "$meta_file" "total_logical_memory_bytes")
    index_only_logical_memory_bytes=$(parse_meta_value "$meta_file" "index_only_logical_memory_bytes")

    echo "[$tag] build_time_s=${build_time_s}, total_size_bytes=${total_size_bytes}, index_only_size_bytes=${index_only_size_bytes}, total_logical_memory_bytes=${total_logical_memory_bytes}, index_only_logical_memory_bytes=${index_only_logical_memory_bytes}"
}

append_acorn_meta_csv() {
    local variant="$1"
    local meta_file="$2"
    local csv_file="$3"
    if [[ ! -f "$meta_file" ]]; then
        return 0
    fi
    local build_time_s total_size_bytes index_only_size_bytes total_logical_memory_bytes index_only_logical_memory_bytes
    build_time_s=$(parse_meta_value "$meta_file" "total_build_time_s")
    if [[ -z "$build_time_s" ]]; then
        build_time_s=$(parse_meta_value "$meta_file" "build_time_s")
    fi
    total_size_bytes=$(parse_meta_value "$meta_file" "total_size_bytes")
    index_only_size_bytes=$(parse_meta_value "$meta_file" "index_only_size_bytes")
    total_logical_memory_bytes=$(parse_meta_value "$meta_file" "total_logical_memory_bytes")
    index_only_logical_memory_bytes=$(parse_meta_value "$meta_file" "index_only_logical_memory_bytes")
    echo "${variant},${build_time_s},${total_size_bytes},${index_only_size_bytes},${total_logical_memory_bytes},${index_only_logical_memory_bytes}" >> "$csv_file"
}

print_ung_meta_summary() {
    local meta_file="$1"
    if [[ ! -f "$meta_file" ]]; then
        echo "[UNG] Meta file not found: $meta_file"
        return 0
    fi
    local rabitq_build_requested rabitq_enabled rabitq_total_bits rabitq_build_time_ms rabitq_side_size_bytes
    local index_time_without_rabitq_ms index_time_with_rabitq_ms
    local index_size_without_rabitq_mb index_size_with_rabitq_mb

    rabitq_build_requested=$(parse_meta_value "$meta_file" "rabitq_build_requested")
    rabitq_enabled=$(parse_meta_value "$meta_file" "rabitq_enabled")
    rabitq_total_bits=$(parse_meta_value "$meta_file" "rabitq_total_bits")
    rabitq_build_time_ms=$(parse_meta_value "$meta_file" "rabitq_build_time(ms)")
    rabitq_side_size_bytes=$(parse_meta_value "$meta_file" "rabitq_side_size_bytes")
    index_time_without_rabitq_ms=$(parse_meta_value "$meta_file" "index_time_without_rabitq(ms)")
    index_time_with_rabitq_ms=$(parse_meta_value "$meta_file" "index_time_with_rabitq(ms)")
    index_size_without_rabitq_mb=$(parse_meta_value "$meta_file" "index_size_without_rabitq(MB)")
    index_size_with_rabitq_mb=$(parse_meta_value "$meta_file" "index_size_with_rabitq(MB)")

    echo "[UNG] rabitq_build_requested=${rabitq_build_requested}, rabitq_enabled=${rabitq_enabled}, rabitq_total_bits=${rabitq_total_bits}, rabitq_build_time_ms=${rabitq_build_time_ms}, rabitq_side_size_bytes=${rabitq_side_size_bytes}, index_time_without_rabitq_ms=${index_time_without_rabitq_ms}, index_time_with_rabitq_ms=${index_time_with_rabitq_ms}, index_size_without_rabitq_mb=${index_size_without_rabitq_mb}, index_size_with_rabitq_mb=${index_size_with_rabitq_mb}"
}

append_ung_meta_csv() {
    local meta_file="$1"
    local csv_file="$2"
    if [[ ! -f "$meta_file" ]]; then
        return 0
    fi
    local rabitq_build_requested rabitq_enabled rabitq_total_bits rabitq_build_time_ms rabitq_side_size_bytes
    local index_time_without_rabitq_ms index_time_with_rabitq_ms
    local index_size_without_rabitq_mb index_size_with_rabitq_mb

    rabitq_build_requested=$(parse_meta_value "$meta_file" "rabitq_build_requested")
    rabitq_enabled=$(parse_meta_value "$meta_file" "rabitq_enabled")
    rabitq_total_bits=$(parse_meta_value "$meta_file" "rabitq_total_bits")
    rabitq_build_time_ms=$(parse_meta_value "$meta_file" "rabitq_build_time(ms)")
    rabitq_side_size_bytes=$(parse_meta_value "$meta_file" "rabitq_side_size_bytes")
    index_time_without_rabitq_ms=$(parse_meta_value "$meta_file" "index_time_without_rabitq(ms)")
    index_time_with_rabitq_ms=$(parse_meta_value "$meta_file" "index_time_with_rabitq(ms)")
    index_size_without_rabitq_mb=$(parse_meta_value "$meta_file" "index_size_without_rabitq(MB)")
    index_size_with_rabitq_mb=$(parse_meta_value "$meta_file" "index_size_with_rabitq(MB)")

    echo "${rabitq_build_requested},${rabitq_enabled},${rabitq_total_bits},${rabitq_build_time_ms},${rabitq_side_size_bytes},${index_time_without_rabitq_ms},${index_time_with_rabitq_ms},${index_size_without_rabitq_mb},${index_size_with_rabitq_mb}" >> "$csv_file"
}

append_ung_variant_compare_csv() {
    local exact_meta_file="$1"
    local rabitq_meta_file="$2"
    local csv_file="$3"

    local exact_time_ms exact_size_mb
    local rabitq_enabled rabitq_time_ms rabitq_size_mb rabitq_build_time_ms rabitq_side_size_bytes

    if [[ -f "$exact_meta_file" ]]; then
        exact_time_ms=$(parse_meta_value "$exact_meta_file" "index_time_without_rabitq(ms)")
        exact_size_mb=$(parse_meta_value "$exact_meta_file" "index_size_without_rabitq(MB)")
    elif [[ -f "$rabitq_meta_file" ]]; then
        exact_time_ms=$(parse_meta_value "$rabitq_meta_file" "index_time_without_rabitq(ms)")
        exact_size_mb=$(parse_meta_value "$rabitq_meta_file" "index_size_without_rabitq(MB)")
    fi
    if [[ -n "$exact_time_ms" || -n "$exact_size_mb" ]]; then
        echo "exact,${exact_time_ms},${exact_size_mb},0,0" >> "$csv_file"
    fi

    if [[ -f "$rabitq_meta_file" ]]; then
        rabitq_enabled=$(parse_meta_value "$rabitq_meta_file" "rabitq_enabled")
        rabitq_time_ms=$(parse_meta_value "$rabitq_meta_file" "index_time_with_rabitq(ms)")
        rabitq_size_mb=$(parse_meta_value "$rabitq_meta_file" "index_size_with_rabitq(MB)")
        rabitq_build_time_ms=$(parse_meta_value "$rabitq_meta_file" "rabitq_build_time(ms)")
        rabitq_side_size_bytes=$(parse_meta_value "$rabitq_meta_file" "rabitq_side_size_bytes")
        if [[ "${rabitq_enabled}" == "1" ]]; then
            echo "rabitq,${rabitq_time_ms},${rabitq_size_mb},${rabitq_build_time_ms},${rabitq_side_size_bytes}" >> "$csv_file"
        fi
    fi
}

build_acorn() {
    local variant="${1:-exact}"
    local acorn_tag marker_file index_path_acorn index_path_acorn1 acorn_log_file target_dir
    local acorn_build_rabitq_flag="false"
    local acorn_rabitq_bits="${RABITQ_TOTAL_BITS:-4}"

    if [[ "$variant" == "rabitq" ]]; then
        target_dir="$INDEX_OUTPUT_DIR"
        acorn_tag="ACORN-RABITQ"
        acorn_build_rabitq_flag="true"
        marker_file="$target_dir/acorn_output/.acorn_rabitq_built"
        index_path_acorn="$target_dir/acorn_output/acorn_rabitq.index"
        index_path_acorn1="$target_dir/acorn_output/acorn1_rabitq.index"
        acorn_log_file="$target_dir/others/acorn_rabitq_build.log"
    else
        if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
            target_dir="$EXACT_INDEX_OUTPUT_DIR"
        else
            target_dir="$INDEX_OUTPUT_DIR"
        fi
        acorn_tag="ACORN"
        marker_file="$target_dir/acorn_output/.acorn_built"
        index_path_acorn="$target_dir/acorn_output/acorn.index"
        index_path_acorn1="$target_dir/acorn_output/acorn1.index"
        acorn_log_file="$target_dir/others/acorn_build.log"
    fi

    if [[ -f "$marker_file" && -f "$index_path_acorn" && -f "$index_path_acorn1" ]]; then
        echo "[$acorn_tag] Marker file found ('$marker_file') and index files exist. Skipping build."
        return 0
    fi

    echo "[$acorn_tag] Build process started."
    local acorn_base_fvecs_path
    local acorn_base_label_path
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${ACORN_FAIR_COMPARE}" == "true" ]]; then
        # Fair comparison: exact and rabitq ACORN use exactly the same input files.
        acorn_base_fvecs_path="${UNG_EXACT_OUTPUT_DIR}/index_files/reordered_vecs.fvecs"
        acorn_base_label_path="${UNG_EXACT_OUTPUT_DIR}/index_files/reordered_labels.txt"
        if [[ ! -f "$acorn_base_fvecs_path" || ! -f "$acorn_base_label_path" ]]; then
            echo "[$acorn_tag] Reordered files not found, fallback to original base files for BOTH exact/rabitq."
            acorn_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
            acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels.txt"
        fi
        echo "[$acorn_tag] Fair-compare input vectors: $acorn_base_fvecs_path"
    elif [[ "$variant" == "rabitq" ]]; then
        acorn_base_fvecs_path="${INDEX_OUTPUT_DIR}/index_files/reordered_vecs.fvecs"
        acorn_base_label_path="${INDEX_OUTPUT_DIR}/index_files/reordered_labels.txt"
        if [[ ! -f "$acorn_base_fvecs_path" || ! -f "$acorn_base_label_path" ]]; then
            echo "[$acorn_tag] Reordered files not found, fallback to original base files."
            acorn_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
            acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels.txt"
        fi
        echo "[$acorn_tag] Using base vectors: $acorn_base_fvecs_path"
    elif [[ "$BUILD_MODE" == "serial" || "$BUILD_MODE" == "all" ]]; then
        acorn_base_fvecs_path="${INDEX_OUTPUT_DIR}/index_files/reordered_vecs.fvecs"
        acorn_base_label_path="${INDEX_OUTPUT_DIR}/index_files/reordered_labels.txt"
        echo "[$acorn_tag] Using reordered base vectors: $acorn_base_fvecs_path"
    else
        acorn_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
        acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels.txt"
        if [[ "$BUILD_MODE" == "acorn_only" ]]; then 
            acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels_reorder_ori.txt"
        fi
        echo "[$acorn_tag] Using original base vectors: $acorn_base_fvecs_path"
    fi

    if [ ! -f "$acorn_base_fvecs_path" ] || [ ! -f "$acorn_base_label_path" ]; then
        echo "[$acorn_tag] 错误: 必需的输入文件未找到."
        exit 1
    fi

    "$ACORN_EXECUTABLE" build \
        "$ACORN_N" "$ACORN_GAMMA" "$DATASET" "$ACORN_M" "$ACORN_M_BETA" \
        "$acorn_base_fvecs_path" "$acorn_base_label_path" "dummy_query_path" \
        "dummy_csv_dir" "dummy_avg_csv_dir" "dummy_dis_path" \
        60 1 true "10" \
        "$index_path_acorn" \
        "$index_path_acorn1" \
        0 \
        "$acorn_build_rabitq_flag" \
        "$acorn_rabitq_bits" \
        > "$acorn_log_file" 2>&1

    echo "[$acorn_tag] Build process finished."
    touch "$marker_file"
}

build_navix() {
    if [ -f "$NAVIX_MARKER_FILE" ]; then
        echo "[NAVIX] Marker file found ('$NAVIX_MARKER_FILE'). Skipping NaviX build."
        return 0
    fi

    echo "[NAVIX] Build process started."
    "$UNG_BUILD_DIR/apps/build_navix_index" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --index_output "$INDEX_OUTPUT_DIR/navix_output/hnsw_base.index" \
        --M "$ACORN_M" \
        --num_threads 60 \
        > "$INDEX_OUTPUT_DIR/others/navix_build.log" 2>&1
    echo "[NAVIX] Build process finished."
    touch "$NAVIX_MARKER_FILE"
}

# --- Step 5: 根据构建模式执行任务 ---
start_time=$(date +%s)

if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    echo "--- [Executing in PARALLEL/ALL mode] ---"
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" ]]; then
        echo "[INFO] RAbitQ ACORN build enabled. Build UNG first to prepare reordered files."
        if [[ "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
            build_ung exact
            build_ung rabitq
        else
            build_ung rabitq
        fi
        if [[ "${ACORN_FAIR_COMPARE}" == "true" ]]; then
            echo "[INFO] ACORN fair compare enabled: build exact and rabitq sequentially to avoid resource contention."
            build_acorn exact
            build_acorn rabitq
        else
            build_acorn exact &
            acorn_pid=$!
            build_acorn rabitq &
            acorn_rabitq_pid=$!
            wait $acorn_pid
            wait $acorn_rabitq_pid
        fi
    else
        build_ung &
        ung_pid=$!
        build_acorn exact &
        acorn_pid=$!
        # build_navix &
        # navix_pid=$!
        wait $ung_pid
        wait $acorn_pid
        # wait $navix_pid
    fi
elif [ "$BUILD_MODE" == "serial" ]; then
    echo "--- [Executing in SERIAL mode] ---"
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
        build_ung exact
        build_ung rabitq
    elif [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" ]]; then
        build_ung rabitq
    else
        build_ung exact
    fi
    build_acorn exact
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" ]]; then
        build_acorn rabitq
    fi
    # build_navix
elif [ "$BUILD_MODE" == "ung_only" ]; then
    echo "--- [Executing in UNG_ONLY mode] ---"
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
        build_ung exact
        build_ung rabitq
    elif [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" ]]; then
        build_ung rabitq
    else
        build_ung exact
    fi
elif [ "$BUILD_MODE" == "acorn_only" ]; then
    echo "--- [Executing in ACORN_ONLY mode] ---"
    build_acorn exact
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" ]]; then
        build_acorn rabitq
    fi
elif [ "$BUILD_MODE" == "navix_only" ]; then
    echo "--- [Executing in NAVIX_ONLY mode] ---"
    build_navix
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "--- Build Summary ---"
echo "[SUCCESS] All build tasks complete. Total time: $duration seconds."
echo "Indexes are saved in: $INDEX_OUTPUT_DIR"

ACORN_META_CSV="$INDEX_OUTPUT_DIR/others/acorn_build_stats.csv"
if [[ -f "$ACORN_META_CSV" ]]; then
    echo "[INFO] ACORN build stats CSV already exists. Skipping analysis: $ACORN_META_CSV"
else
    echo "variant,build_time_s,total_size_bytes,index_only_size_bytes,total_logical_memory_bytes,index_only_logical_memory_bytes" > "$ACORN_META_CSV"

    ACORN_EXACT_META_DIR="$INDEX_OUTPUT_DIR"
    if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" && "${SPLIT_RABITQ_EXACT_DIRS}" == "true" ]]; then
        ACORN_EXACT_META_DIR="$EXACT_INDEX_OUTPUT_DIR"
    fi

    print_acorn_meta_summary "ACORN-gamma-EXACT" "$ACORN_EXACT_META_DIR/acorn_output/acorn.index.meta"
    append_acorn_meta_csv "acorn_gamma_exact" "$ACORN_EXACT_META_DIR/acorn_output/acorn.index.meta" "$ACORN_META_CSV"
    if [[ -f "$ACORN_EXACT_META_DIR/acorn_output/acorn1.index.meta" ]]; then
        print_acorn_meta_summary "ACORN-1-EXACT" "$ACORN_EXACT_META_DIR/acorn_output/acorn1.index.meta"
        append_acorn_meta_csv "acorn_1_exact" "$ACORN_EXACT_META_DIR/acorn_output/acorn1.index.meta" "$ACORN_META_CSV"
    fi

    if [[ -f "$INDEX_OUTPUT_DIR/acorn_output/acorn_rabitq.index.meta" ]]; then
        print_acorn_meta_summary "ACORN-gamma-RABITQ" "$INDEX_OUTPUT_DIR/acorn_output/acorn_rabitq.index.meta"
        append_acorn_meta_csv "acorn_gamma_rabitq" "$INDEX_OUTPUT_DIR/acorn_output/acorn_rabitq.index.meta" "$ACORN_META_CSV"
    fi
    if [[ -f "$INDEX_OUTPUT_DIR/acorn_output/acorn1_rabitq.index.meta" ]]; then
        print_acorn_meta_summary "ACORN-1-RABITQ" "$INDEX_OUTPUT_DIR/acorn_output/acorn1_rabitq.index.meta"
        append_acorn_meta_csv "acorn_1_rabitq" "$INDEX_OUTPUT_DIR/acorn_output/acorn1_rabitq.index.meta" "$ACORN_META_CSV"
    fi

    echo "ACORN build stats saved to: $ACORN_META_CSV"
fi

UNG_EXACT_META_FILE="$UNG_EXACT_OUTPUT_DIR/index_files/meta"
UNG_RABITQ_META_FILE="$UNG_RABITQ_OUTPUT_DIR/index_files/meta"
UNG_META_FILE="$UNG_RABITQ_META_FILE"
if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" != "true" ]]; then
    UNG_META_FILE="$UNG_EXACT_META_FILE"
fi
UNG_META_CSV="$INDEX_OUTPUT_DIR/others/ung_build_stats.csv"
if [[ -f "$UNG_META_CSV" ]]; then
    echo "[INFO] UNG build stats CSV already exists. Skipping analysis: $UNG_META_CSV"
else
    echo "rabitq_build_requested,rabitq_enabled,rabitq_total_bits,rabitq_build_time_ms,rabitq_side_size_bytes,index_time_without_rabitq_ms,index_time_with_rabitq_ms,index_size_without_rabitq_mb,index_size_with_rabitq_mb" > "$UNG_META_CSV"
    print_ung_meta_summary "$UNG_META_FILE"
    append_ung_meta_csv "$UNG_META_FILE" "$UNG_META_CSV"
    echo "UNG build stats saved to: $UNG_META_CSV"
fi

UNG_COMPARE_CSV="$INDEX_OUTPUT_DIR/others/ung_build_compare_stats.csv"
if [[ -f "$UNG_COMPARE_CSV" ]]; then
    echo "[INFO] UNG compare stats CSV already exists. Skipping analysis: $UNG_COMPARE_CSV"
else
    echo "variant,build_time_ms,index_size_mb,rabitq_build_time_ms,rabitq_side_size_bytes" > "$UNG_COMPARE_CSV"
    append_ung_variant_compare_csv "$UNG_EXACT_META_FILE" "$UNG_RABITQ_META_FILE" "$UNG_COMPARE_CSV"
    echo "UNG compare stats saved to: $UNG_COMPARE_CSV"
fi
