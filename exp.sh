#!/bin/bash

set -e # Exit immediately if any command fails

IS_NEW_TRIE_METHOD=false

# --- Locate the script and configuration file ---
if [ -z "$1" ]; then
    echo "Error: Please provide a JSON configuration file as the first argument."
    echo "Usage: ./exp.sh [config_file.json]"
    exit 1
fi

CONFIG_FILE="$1"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found. Please verify the path: $CONFIG_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq first: https://stedolan.github.io/jq/"
    exit 1
fi

echo "Configuration file located successfully: $CONFIG_FILE"
echo "Starting experiment execution..."

cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r dataset_config; do
    
    # --- Step A: Extract shared parameters as defaults ---
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
    # Read shared ACORN build parameters
    ACORN_N=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.N')
    ACORN_M=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M')
    ACORN_M_BETA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M_beta')
    ACORN_GAMMA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.gamma')
    LSEARCH_THRESHOLD=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.lsearch_threshold')
    BUILD_RABITQ_SIDE_INDEX=$(echo "$SHARED_CONFIG" | jq -r '.build_rabitq_side_index // false')
    RABITQ_TOTAL_BITS=$(echo "$SHARED_CONFIG" | jq -r '.rabitq_total_bits // 4')
    UNG_DISTANCE_MODE_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.ung_distance_mode // "exact"')
    OPTIMIZE_STANDALONE_PREFILTER=$(echo "$SHARED_CONFIG" | jq -r '.optimize_standalone_prefilter // false')
    if [[ "$BUILD_RABITQ_SIDE_INDEX" == "true" || "$UNG_DISTANCE_MODE_DEFAULT" == "rabitq" ]]; then
        echo "[WARN] RabitQ support has been disabled. Forcing build_rabitq_side_index=false and ung_distance_mode=exact."
    fi
    BUILD_RABITQ_SIDE_INDEX=false
    UNG_DISTANCE_MODE_DEFAULT="exact"

    # Project root: prefer externally provided value, otherwise infer from the script location
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
    export UNG_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/ung"
    export ACORN_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/acorn"
    export NAVIX_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/navix"

    # Knowhere paths: if not explicitly set, fall back to the default repository paths
    if [[ -z "${KNOWHERE_INCLUDE_DIR:-}" ]]; then
        export KNOWHERE_INCLUDE_DIR="${SCRIPT_DIR}/knowhere/include"
    fi
    if [[ -z "${KNOWHERE_LIBRARY:-}" ]]; then
        export KNOWHERE_LIBRARY="${SCRIPT_DIR}/knowhere/build/Release/libknowhere.so"
    fi

    echo "$dataset_config" | jq -c '.tasks[]' | while read -r task; do
        QUERY_DIR_NAME=$(echo "$task" | jq -r '.query_dir_name')

        ACORN_EFS_START=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_start')
        ACORN_EFS_STEP_SLOW=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_slow')
        ACORN_EFS_STEP_FAST=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_fast')

        if [[ "$ACORN_EFS_START" == "null" || -z "$ACORN_EFS_START" ]]; then
            echo "Error: Task '$QUERY_DIR_NAME' is missing the required 'acorn_efs_start' parameter."
            exit 1
        fi


        echo "$task" | jq -r '.algorithms[]' | while read -r ALGORITHM_NAME; do
            
            echo -e "\n=========================================================="
            echo "Processing: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME]"
            echo "Using ACORN search params: efs_start=${ACORN_EFS_START}, efs_step_slow=${ACORN_EFS_STEP_SLOW}, efs_step_fast=${ACORN_EFS_STEP_FAST}"
            echo "=========================================================="

            case "$ALGORITHM_NAME" in
                "UNG-nTfalse")    ROUTING_MODE=0; BASELINE_ALG=0 ; IS_REC_MORE_START=false;;
                "ACORN-gamma")    ROUTING_MODE=0; BASELINE_ALG=2 ; IS_REC_MORE_START=false;;
                "NaviX-ACORN")    ROUTING_MODE=0; BASELINE_ALG=4 ; IS_REC_MORE_START=false;;
                "pre-filter")     ROUTING_MODE=0; BASELINE_ALG=5 ; IS_REC_MORE_START=false;;
                "ACORN-1")        ROUTING_MODE=0; BASELINE_ALG=6 ; IS_REC_MORE_START=false;;
                "UNG+")           ROUTING_MODE=0; BASELINE_ALG=8 ; IS_REC_MORE_START=false;;
                "Milvus-IVF")     ROUTING_MODE=0; BASELINE_ALG=9 ; IS_REC_MORE_START=false;;
                "Milvus-HNSW")    ROUTING_MODE=0; BASELINE_ALG=10; IS_REC_MORE_START=false;;
                "SmartRoute")     ROUTING_MODE=1; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;; 
                "SmartRoute+")    ROUTING_MODE=5; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                *)
                    echo "Error: Unknown algorithm name '$ALGORITHM_NAME'. Please define it in the case statement in exp.sh."
                    exit 1;;
            esac

            UNG_DISTANCE_MODE="$UNG_DISTANCE_MODE_DEFAULT"
            EFFECTIVE_BUILD_RABITQ_SIDE_INDEX=false
            
            SHARED_DATASET_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
            ALGO_RESULT_DIR="${SHARED_DATASET_DIR}/Results/${ALGORITHM_NAME}"
            
            # --- Invoke build_hybrid.sh ---
            # build_hybrid.sh handles compilation, data conversion, and index construction
            echo "Preparing index build..."

            ./build_hybrid.sh \
               --build_mode "$BUILD_MODE" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
               --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
               --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
               --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA"
            
            # Some build modes are build-only and do not run GT generation or search.
            if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "acorn_only" || "$BUILD_MODE" == "navix_only" || "$BUILD_MODE" == "ung_only" ]]; then
               echo "[INFO] Skipping GT generation and search steps."
               echo "--- The current experimental configuration processing has been completed (BUILD ONLY) ---"
               continue
            fi
            
            # --- Invoke generate_gt.sh ---
            echo "Preparing ground truth data (K=$K)..."
            ./generate_gt.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" --build_dir "$UNG_BUILD_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --K "$K"

            # --- Invoke search.sh ---
            INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
            echo "Using UNG distance mode: $UNG_DISTANCE_MODE"
            echo "Using RabitQ side index: $EFFECTIVE_BUILD_RABITQ_SIDE_INDEX"
            echo "Using index dir name: $INDEX_DIR_NAME"
            echo "Begin search (K=$K)..."
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
        done
    done
done

echo -e "\nAll experiments have completed successfully."
