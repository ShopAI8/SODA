#!/bin/bash

# ==============================================================================
# search.sh - 负责在已有的索引和GT上执行搜索任务
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

# --- Step 1: 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        if [[ $key == "QUERY_DIR_NAME" ]]; then
            QUERY_DIR_NAME="$2"
            shift 2
            continue
        fi
        if [[ $key == "ROUTING_MODE" ]]; then
            ROUTING_MODE="$2"
            shift 2
            continue
        fi
        if [[ $key == "BASELINE_ALG" ]]; then
            BASELINE_ALG="$2"
            shift 2
            continue
        fi
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

# 默认参数
if [ -z "$UNG_DISTANCE_MODE" ]; then
    UNG_DISTANCE_MODE="exact"
fi

# --- Step 2: 根据搜索参数构造唯一的结果输出目录 ---
SAFE_QUERY_NAME=$(echo "$QUERY_DIR_NAME" | tr '/' '_')
GT_DIR_NAME="GT_${SAFE_QUERY_NAME}_K${K}"
SEARCH_DIR_NAME="Ls${LSEARCH_START}-Le${LSEARCH_END}-Lp${LSEARCH_STEP}_efsS${EFS_START}-efss${EFS_STEP_SLOW}-efsf${EFS_STEP_FAST}-lt${LSEARCH_THRESHOLD}_K${K}_th${NUM_THREADS}"
RESULT_OUTPUT_DIR="${ALGO_RESULT_DIR}/Index[${INDEX_DIR_NAME}]_GT[${GT_DIR_NAME}]_Search[${SEARCH_DIR_NAME}]"

# --- Step 3: 创建结果目录 ---
mkdir -p "$RESULT_OUTPUT_DIR/results"
mkdir -p "$RESULT_OUTPUT_DIR/others"

# --- Step 4: 准备Lsearch参数序列 ---
LSEARCH_VALUES=$(seq "$LSEARCH_START" "$LSEARCH_STEP" "$LSEARCH_END" | tr '\n' ' ')
echo "将在以下Lsearch值上进行测试: $LSEARCH_VALUES"

# --- Step 5: 定义依赖文件和目录的路径 ---
# 根据构建模式确定索引基础目录
if [[ "$BUILD_MODE" == "parallel" ]]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi
INDEX_PATH="${SHARED_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"
GT_PATH="${SHARED_OUTPUT_DIR}/GroundTruth/${GT_DIR_NAME}"
MODEL_PATH="${SHARED_OUTPUT_DIR}/SelectModels"
# MODEL_PATH="/noraiddata/lijiakang/FilterVector/FilterVectorResults/OLD/${DATASET}/SelectModels"
ACORN_INDEX_PREFIX="${INDEX_PATH}/acorn_output"
NAVIX_INDEX_PATH="${INDEX_PATH}/navix_output/hnsw_base.index"
ACORN_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn.index"
ACORN_1_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn1.index"

QUERY_DIR="${DATA_DIR}/${QUERY_DIR_NAME}"
echo "Using query directory from: $QUERY_DIR"

# SmartRoute++/pre-filter-rabitq 场景下，UNG 走 RQB 索引；
# 若 RQB 目录未构建所需 ACORN/NaviX 文件，则自动回退到非 RQB 目录读取对应索引文件。
if [[ "$UNG_DISTANCE_MODE" == "rabitq" && "$INDEX_DIR_NAME" =~ ^(.*)_RQB[0-9]+$ ]]; then
    EXACT_INDEX_DIR_NAME="${BASH_REMATCH[1]}"
    EXACT_INDEX_PATH="${SHARED_OUTPUT_DIR}/${INDEX_BASE_DIR}/${EXACT_INDEX_DIR_NAME}"

    # rabitq 模式优先要求 rabitq ACORN 文件；若 RQB 目录缺失则回退到 exact 标准 ACORN。
    if [[ ! -f "${ACORN_INDEX_PREFIX}/acorn_rabitq.index" || ! -f "${ACORN_INDEX_PREFIX}/acorn1_rabitq.index" ]]; then
        if [[ -f "${EXACT_INDEX_PATH}/acorn_output/acorn.index" && -f "${EXACT_INDEX_PATH}/acorn_output/acorn1.index" ]]; then
            echo "[INFO] ACORN RAbitQ index files not found in RQB dir, fallback to exact dir: ${EXACT_INDEX_PATH}/acorn_output"
            ACORN_INDEX_PREFIX="${EXACT_INDEX_PATH}/acorn_output"
        fi
    fi

    if [[ ! -f "$NAVIX_INDEX_PATH" && -f "${EXACT_INDEX_PATH}/navix_output/hnsw_base.index" ]]; then
        echo "[INFO] NaviX index not found in RQB dir, fallback to exact dir: ${EXACT_INDEX_PATH}/navix_output/hnsw_base.index"
        NAVIX_INDEX_PATH="${EXACT_INDEX_PATH}/navix_output/hnsw_base.index"
    fi
fi

# 根据最终 ACORN_INDEX_PREFIX 重新设置默认索引路径
ACORN_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn.index"
ACORN_1_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn1.index"

if [[ "$UNG_DISTANCE_MODE" == "rabitq" ]]; then
    if [[ -f "${ACORN_INDEX_PREFIX}/acorn_rabitq.index" && -f "${ACORN_INDEX_PREFIX}/acorn1_rabitq.index" ]]; then
        ACORN_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn_rabitq.index"
        ACORN_1_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn1_rabitq.index"
        echo "[INFO] Using ACORN RAbitQ index files."
    else
        echo "[INFO] ACORN RAbitQ index files not found, fallback to standard ACORN files."
    fi
fi

echo "使用索引: $INDEX_PATH"
echo "使用GT: $GT_PATH"
echo "结果将保存到: $RESULT_OUTPUT_DIR"
echo "使用 ACORN 索引: ${ACORN_INDEX_FILE}"
echo "使用 NaviX 索引: $NAVIX_INDEX_PATH"

# --- Step 6: 执行搜索 ---
PERF_EVENTS="cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss,LLC-loads,LLC-load-misses,branches,branch-misses"
PERF_LOG_PATH="$RESULT_OUTPUT_DIR/others/${DATASET}_perf_stat.log"
echo "性能分析(Perf stat)结果将独立保存到: $PERF_LOG_PATH"

perf stat -e $PERF_EVENTS -o "$PERF_LOG_PATH" \
"$BUILD_DIR"/apps/search_UNG_index \
    --data_type float  --dataset "$DATASET" --dist_fn L2 --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
    --is_new_method true \
    --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
    --routing_mode "$ROUTING_MODE" \
    --baseline_alg "$BASELINE_ALG" \
    --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
    --query_bin_file "$QUERY_DIR/${DATASET}_query.bin" \
    --query_label_file "$QUERY_DIR/${DATASET}_query_labels.txt" \
    --query_group_id_file "$QUERY_DIR/${DATASET}_query_source_groups.txt" \
    --gt_file "$GT_PATH/${DATASET}_gt_labels_containment.bin" \
    --index_path_prefix "$INDEX_PATH/index_files/" \
    --result_path_prefix "$RESULT_OUTPUT_DIR/results/" \
    --acorn_index_path "${ACORN_INDEX_FILE}" --acorn_1_index_path "${ACORN_1_INDEX_FILE}" \
    --selector_modle_prefix "${MODEL_PATH}" \
    --scenario containment \
    --num_entry_points "$NUM_ENTRY_POINTS" \
    --Lsearch $LSEARCH_VALUES \
    --lsearch_start "$LSEARCH_START" \
    --lsearch_step "$LSEARCH_STEP" \
    --efs_start "$EFS_START" \
    --efs_step_slow "$EFS_STEP_SLOW" --efs_step_fast "$EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD" \
    --ung_distance_mode "$UNG_DISTANCE_MODE" \
    --navix_index_path "$NAVIX_INDEX_PATH" \
    --algo_choice_csv "$QUERY_DIR/algo_choice_repeat.csv" \
    --optimize_standalone_prefilter "${OPTIMIZE_STANDALONE_PREFILTER:-false}" > "$RESULT_OUTPUT_DIR/others/${DATASET}_search_output.txt" 2>&1

# --- Step 7: 后处理，计算各指标全局平均值 ---
echo "正在计算所有 Query 指标的全局平均值..."
DETAILS_CSV="${RESULT_OUTPUT_DIR}/results/query_details_repeat${NUM_REPEATS}.csv"
AVERAGE_CSV="${RESULT_OUTPUT_DIR}/results/query_details_global_average.csv"

if [ -f "$DETAILS_CSV" ]; then
    python3 UNG/data/average_query_details.py --input_csv "$DETAILS_CSV" --output_csv "$AVERAGE_CSV"
else
    echo "⚠️ 未找到明细文件 $DETAILS_CSV，跳过平均值计算。"
fi

echo "所有搜索和统计任务已全部结束！"
