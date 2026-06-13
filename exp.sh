#!/bin/bash

set -e # 如果任何命令失败，则立即退出

IS_NEW_TRIE_METHOD=false

# --- 定位脚本和配置文件 ---
if [ -z "$1" ]; then
    echo "错误: 请提供一个 JSON 配置文件作为第一个参数。"
    echo "用法: ./exp.sh [config_file.json]"
    exit 1
fi

CONFIG_FILE="$1"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件未找到，请检查路径: $CONFIG_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装 jq (https://stedolan.github.io/jq/)"
    exit 1
fi

echo "成功找到配置文件: $CONFIG_FILE"
echo "开始执行实验..."

while read -r dataset_config; do
    
    # --- 【步骤A】提取所有共享参数作为默认值 ---
    DATASET=$(echo "$dataset_config" | jq -r '.dataset_name')
    SHARED_CONFIG=$(echo "$dataset_config" | jq '.shared_config')
    
    DATA_DIR=$(echo "$SHARED_CONFIG" | jq -r '.data_dir')
    BASE_OUTPUT_DIR=$(echo "$SHARED_CONFIG" | jq -r '.output_dir')
    BUILD_MODE=$(echo "$SHARED_CONFIG" | jq -r '.build_mode')
    MAX_DEGREE=$(echo "$SHARED_CONFIG" | jq -r '.max_degree')
    LBUILD=$(echo "$SHARED_CONFIG" | jq -r '.Lbuild')
    ALPHA=$(echo "$SHARED_CONFIG" | jq -r '.alpha')
    NUM_CROSS_EDGES=$(echo "$SHARED_CONFIG" | jq -r '.num_cross_edges')
    NUM_ENTRY_POINTS=$(echo "$SHARED_CONFIG" | jq -r '.num_entry_points')
    K=$(echo "$SHARED_CONFIG" | jq -r '.K')
    LSEARCH_START=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_start')
    LSEARCH_END=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_end')
    LSEARCH_STEP=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_step')
    NUM_THREADS=$(echo "$SHARED_CONFIG" | jq -r '.num_threads')
    NUM_REPEATS=$(echo "$SHARED_CONFIG" | jq -r '.num_repeats')
    # 读取共享的ACORN构建参数
    ACORN_N=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.N')
    ACORN_M=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M')
    ACORN_M_BETA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M_beta')
    ACORN_GAMMA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.gamma')
    LSEARCH_THRESHOLD=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.lsearch_threshold')
    BUILD_RABITQ_SIDE_INDEX=$(echo "$SHARED_CONFIG" | jq -r '.build_rabitq_side_index // false')
    RABITQ_TOTAL_BITS=$(echo "$SHARED_CONFIG" | jq -r '.rabitq_total_bits // 4')
    UNG_DISTANCE_MODE_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.ung_distance_mode // "exact"')
    OPTIMIZE_STANDALONE_PREFILTER=$(echo "$SHARED_CONFIG" | jq -r '.optimize_standalone_prefilter // false')
    SELECTOR_MODEL_PATH_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.selector_model_path // ""')
    ROUTER_ZERO_ALGORITHM_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.router_zero_algorithm // ""')
    RESULT_NAME_SUFFIX_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.result_name_suffix // ""')

    # 项目根路径：优先用外部传入，其次自动从当前脚本位置推导
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
    export UNG_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/ung"
    export ACORN_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/acorn"
    export NAVIX_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/navix"
    BUILD_ONLY_INDEX_PREPARED=false

    # Knowhere 路径：若未显式设置，则尝试使用当前仓库下的默认位置
    if [[ -z "${KNOWHERE_INCLUDE_DIR:-}" ]]; then
        export KNOWHERE_INCLUDE_DIR="${SCRIPT_DIR}/knowhere/include"
    fi
    if [[ -z "${KNOWHERE_LIBRARY:-}" ]]; then
        export KNOWHERE_LIBRARY="${SCRIPT_DIR}/knowhere/build/Release/libknowhere.so"
    fi

    # export UNG_BUILD_DIR="/home/fengxiaoyao/FilterVector/build_para/ung"
    # export ACORN_BUILD_DIR="/home/fengxiaoyao/FilterVector/build_para/acorn"
    # export NAVIX_BUILD_DIR="/home/fengxiaoyao/FilterVector/build_para/navix"

    
    # --- 中层循环: 遍历任务 (查询) ---
    while read -r task; do
        QUERY_DIR_NAME=$(echo "$task" | jq -r '.query_dir_name')

        # --- 加载并覆盖任务专属参数 ---
        ACORN_EFS_START=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_start')
        ACORN_EFS_STEP_SLOW=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_slow')
        ACORN_EFS_STEP_FAST=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_fast')
        TASK_SELECTOR_MODEL_PATH=$(echo "$task" | jq -r '.selector_model_path // ""')
        TASK_ROUTER_ZERO_ALGORITHM=$(echo "$task" | jq -r '.router_zero_algorithm // ""')
        TASK_RESULT_NAME_SUFFIX=$(echo "$task" | jq -r '.result_name_suffix // ""')

        EFFECTIVE_SELECTOR_MODEL_PATH="$SELECTOR_MODEL_PATH_DEFAULT"
        if [[ -n "$TASK_SELECTOR_MODEL_PATH" && "$TASK_SELECTOR_MODEL_PATH" != "null" ]]; then
            EFFECTIVE_SELECTOR_MODEL_PATH="$TASK_SELECTOR_MODEL_PATH"
        fi

        EFFECTIVE_ROUTER_ZERO_ALGORITHM="$ROUTER_ZERO_ALGORITHM_DEFAULT"
        if [[ -n "$TASK_ROUTER_ZERO_ALGORITHM" && "$TASK_ROUTER_ZERO_ALGORITHM" != "null" ]]; then
            EFFECTIVE_ROUTER_ZERO_ALGORITHM="$TASK_ROUTER_ZERO_ALGORITHM"
        fi

        EFFECTIVE_RESULT_NAME_SUFFIX="$RESULT_NAME_SUFFIX_DEFAULT"
        if [[ -n "$TASK_RESULT_NAME_SUFFIX" && "$TASK_RESULT_NAME_SUFFIX" != "null" ]]; then
            EFFECTIVE_RESULT_NAME_SUFFIX="$TASK_RESULT_NAME_SUFFIX"
        fi

        # 检查是否成功读取，如果为null或空，则给出错误提示
        if [[ "$ACORN_EFS_START" == "null" || -z "$ACORN_EFS_START" ]]; then
            echo "错误: 任务 '$QUERY_DIR_NAME' 缺少 'acorn_efs_start' 参数！"
            exit 1
        fi

        # --- 内层循环: 遍历算法名称 ---
        while read -r ALGORITHM_NAME; do
            
            echo -e "\n=========================================================="
            echo "Processing: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME]"
            echo "Using ACORN search params: efs_start=${ACORN_EFS_START}, efs_step_slow=${ACORN_EFS_STEP_SLOW}, efs_step_fast=${ACORN_EFS_STEP_FAST}"
            echo "=========================================================="

            # 根据算法名称设置详细参数
            case "$ALGORITHM_NAME" in
                "UNG-nTfalse")    ROUTING_MODE=0; BASELINE_ALG=0 ; IS_REC_MORE_START=false;;
                "ACORN-gamma")    ROUTING_MODE=0; BASELINE_ALG=2 ; IS_REC_MORE_START=false;;
                "NaviX-ACORN")    ROUTING_MODE=0; BASELINE_ALG=4 ; IS_REC_MORE_START=false;;
                "pre-filter")     ROUTING_MODE=0; BASELINE_ALG=5 ; IS_REC_MORE_START=false;;
                "ACORN-1")        ROUTING_MODE=0; BASELINE_ALG=6 ; IS_REC_MORE_START=false;;
                "UNG+")           ROUTING_MODE=0; BASELINE_ALG=8 ; IS_REC_MORE_START=false;;
                "Milvus-IVF")     ROUTING_MODE=0; BASELINE_ALG=9 ; IS_REC_MORE_START=false;;
                "Milvus-HNSW")    ROUTING_MODE=0; BASELINE_ALG=10; IS_REC_MORE_START=false;;
                "FAVOR")          ROUTING_MODE=0; BASELINE_ALG=11; IS_REC_MORE_START=false;;
                "FAVOR-HNSW")     ROUTING_MODE=0; BASELINE_ALG=12; IS_REC_MORE_START=false;;
                "SODA")     ROUTING_MODE=1; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;; 
                "SODA+")    ROUTING_MODE=5; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                *)
                    echo "错误: 未知的算法名称 '$ALGORITHM_NAME'。请在 exp.sh 的 case 语句中定义它。"
                    exit 1;;
            esac

            # 默认遵循 JSON 配置；仅对少数算法做强制覆盖
            UNG_DISTANCE_MODE="$UNG_DISTANCE_MODE_DEFAULT"
            if [[ "$ALGORITHM_NAME" == "SmartRoute++" || "$ALGORITHM_NAME" == "SmartRoute+++" ]]; then
                UNG_DISTANCE_MODE="rabitq"
            fi

            # 默认遵循 JSON 的 build_rabitq_side_index
            EFFECTIVE_BUILD_RABITQ_SIDE_INDEX="$BUILD_RABITQ_SIDE_INDEX"
            # SmartRoute++ / SmartRoute+++ 必定需要 RabitQ side index
            if [[ "$ALGORITHM_NAME" == "SmartRoute++" || "$ALGORITHM_NAME" == "SmartRoute+++" ]]; then
                if [[ "$BUILD_RABITQ_SIDE_INDEX" != "true" ]]; then
                    echo "[WARN] 算法 '$ALGORITHM_NAME' 强制使用 rabitq，已自动将 build_rabitq_side_index 从 '$BUILD_RABITQ_SIDE_INDEX' 切换为 true。"
                fi
                EFFECTIVE_BUILD_RABITQ_SIDE_INDEX="true"
            fi
            
            SHARED_DATASET_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
            RESULT_ALGORITHM_NAME="${ALGORITHM_NAME}"
            if [[ -n "$EFFECTIVE_RESULT_NAME_SUFFIX" ]]; then
                RESULT_ALGORITHM_NAME="${ALGORITHM_NAME}_${EFFECTIVE_RESULT_NAME_SUFFIX}"
            elif [[ "$ROUTING_MODE" -ne 0 && -n "$EFFECTIVE_ROUTER_ZERO_ALGORITHM" ]]; then
                RESULT_ALGORITHM_NAME="${ALGORITHM_NAME}_${EFFECTIVE_ROUTER_ZERO_ALGORITHM}"
            fi
            ALGO_RESULT_DIR="${SHARED_DATASET_DIR}/Results/${RESULT_ALGORITHM_NAME}"

            # --- 调用 build_hybrid.sh ---
            # build_hybrid.sh 内部会负责编译、数据转换和索引构建
            if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "acorn_only" || "$BUILD_MODE" == "navix_only" || "$BUILD_MODE" == "ung_only" || "$BUILD_MODE" == "favor_only" ]]; then
                if [[ "$BUILD_ONLY_INDEX_PREPARED" == true ]]; then
                    echo "Preparing build index..."
                    echo "[INFO] Build-only mode: index already prepared for dataset '$DATASET' in this run. Skipping duplicate rebuild."
                else
                    echo "Preparing build index..."
                    ./build_hybrid.sh \
                       --build_mode "$BUILD_MODE" \
                       --query_dir_name "$QUERY_DIR_NAME" \
                       --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
                       --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
                       --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
                       --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA" \
                       --build_rabitq_side_index "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" \
                       --rabitq_total_bits "$RABITQ_TOTAL_BITS"
                    BUILD_ONLY_INDEX_PREPARED=true
                fi
            else
                echo "Preparing build index..."
                ./build_hybrid.sh \
                   --build_mode "$BUILD_MODE" \
                   --query_dir_name "$QUERY_DIR_NAME" \
                   --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
                   --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
                   --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
                   --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA" \
                   --build_rabitq_side_index "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" \
                   --rabitq_total_bits "$RABITQ_TOTAL_BITS"
            fi
            
            # 新增判断：某些 build_mode 只做构建，不进行 GT 和搜索。
            if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "acorn_only" || "$BUILD_MODE" == "navix_only" || "$BUILD_MODE" == "ung_only" || "$BUILD_MODE" == "favor_only" ]]; then
               echo "[INFO] Skipping GT generation and search steps."
               echo "--- The current experimental configuration processing has been completed (BUILD ONLY) ---"
               continue
            fi
            
            # --- 调用 generate_gt.sh ---
            echo "Preparing Ground Truth (K=$K)..."
            ./generate_gt.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" --build_dir "$UNG_BUILD_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --K "$K"

            # --- 调用 search.sh ---
            INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
            if [[ "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" == "true" ]]; then
               INDEX_DIR_NAME="${INDEX_DIR_NAME}_RQB${RABITQ_TOTAL_BITS}"
            fi
            echo "Using UNG distance mode: $UNG_DISTANCE_MODE"
            echo "Using RabitQ side index: $EFFECTIVE_BUILD_RABITQ_SIDE_INDEX"
            echo "Using index dir name: $INDEX_DIR_NAME"
            if [[ -n "$EFFECTIVE_SELECTOR_MODEL_PATH" ]]; then
               echo "Using selector model path override: $EFFECTIVE_SELECTOR_MODEL_PATH"
            fi
            if [[ -n "$EFFECTIVE_ROUTER_ZERO_ALGORITHM" ]]; then
               echo "Using SODA class-0 override: $EFFECTIVE_ROUTER_ZERO_ALGORITHM"
            fi
            if [[ -n "$EFFECTIVE_RESULT_NAME_SUFFIX" ]]; then
               echo "Using result name suffix override: $EFFECTIVE_RESULT_NAME_SUFFIX"
            fi
            echo "Begin search (K=$K)..."
            if [[ -n "$EFFECTIVE_SELECTOR_MODEL_PATH" ]]; then
               export SELECTOR_MODEL_PATH="$EFFECTIVE_SELECTOR_MODEL_PATH"
            else
               unset SELECTOR_MODEL_PATH
            fi
            if [[ -n "$EFFECTIVE_ROUTER_ZERO_ALGORITHM" ]]; then
               export ROUTER_ZERO_ALGORITHM="$EFFECTIVE_ROUTER_ZERO_ALGORITHM"
            else
               unset ROUTER_ZERO_ALGORITHM
            fi

            ./search.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --shared_output_dir "$SHARED_DATASET_DIR" \
               --algo_result_dir "$ALGO_RESULT_DIR" \
               --build_dir "$UNG_BUILD_DIR" \
               --index_dir_name "$INDEX_DIR_NAME" \
               --build_mode "$BUILD_MODE" \
               --num_entry_points "$NUM_ENTRY_POINTS" \
               --Lsearch_start "$LSEARCH_START" --Lsearch_end "$LSEARCH_END" --Lsearch_step "$LSEARCH_STEP" \
               --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
               --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
               --routing_mode "$ROUTING_MODE" \
               --baseline_alg "$BASELINE_ALG" \
               --ung_distance_mode "$UNG_DISTANCE_MODE" \
               --efs_start "$ACORN_EFS_START" \
               --efs_step_slow "$ACORN_EFS_STEP_SLOW" --efs_step_fast "$ACORN_EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD" \
               --optimize_standalone_prefilter "$OPTIMIZE_STANDALONE_PREFILTER"
                    
            echo "--- Finished: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME] ---"
        done < <(echo "$task" | jq -r '.algorithms[]')
    done < <(echo "$dataset_config" | jq -c '.tasks[]')
done < <(jq -c '.experiments[]' "$CONFIG_FILE")

echo -e "\n所有实验已完成！"
