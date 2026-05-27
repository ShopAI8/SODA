#!/bin/bash

LOG_ROOT="log_th-K-split"
mkdir -p "$LOG_ROOT"

cd /home/lijiakang/FilterVector/FilterVectorCode
echo "=== 启动拆分配置批量实验任务流程 ==="
echo "所有执行记录将保存至：$(pwd)/$LOG_ROOT"

# 恢复运行控制：
# 重新启动脚本时，会先跳过恢复点之前的配置组/数据集；
# 在命中恢复点后，会把对应 JSON 临时裁剪到指定算法开始，再继续后续所有任务。
RESUME_SUFFIX="th100-K20"
RESUME_DATASET="Tiktok"
RESUME_ALGORITHM="Milvus-HNSW"
RESUME_ACTIVE=false

DATASETS=(
    # "Genome"
    # "Amazon"
    # "Reviews"
    "BookReviews"  
    "Music"
    "VariousImg"
    "Tiktok"
    "Laion"
)

CONFIG_SUFFIXES=(
    # "th100-K20"
    "th10-K10"
    # "th50-K10"
)

PERF_EVENTS="cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss,LLC-loads,LLC-load-misses,branches,branch-misses"

STEP=0
for SUFFIX in "${CONFIG_SUFFIXES[@]}"; do
    if [[ "$RESUME_ACTIVE" == true && "$SUFFIX" != "$RESUME_SUFFIX" ]]; then
        echo "$(date): [步骤 $STEP] 跳过配置组 $SUFFIX，等待恢复点 $RESUME_SUFFIX。"
        continue
    fi

    LOG_DIR="$LOG_ROOT/$SUFFIX"
    mkdir -p "$LOG_DIR"

    echo "$(date): === 开始执行配置组 $SUFFIX ==="
    echo "   >> 日志目录: $(pwd)/$LOG_DIR"

    for DS_NAME in "${DATASETS[@]}"; do
        JSON_FILE="experiment_json/202604-200-random-300-mix-th-K/experiments-${DS_NAME}-200-random-300-mix-len-${SUFFIX}.json"
        OUTPUT_LOG="$LOG_DIR/${DS_NAME}_output.log"
        PERF_SUMMARY_LOG="$LOG_DIR/${DS_NAME}_perf_summary.log"
        RUN_JSON_FILE="$JSON_FILE"

        if [ ! -f "$JSON_FILE" ]; then
            echo "$(date): [步骤 $STEP] 警告：找不到配置文件 $JSON_FILE，跳过。"
            STEP=$((STEP + 1))
            continue
        fi

        if [[ "$RESUME_ACTIVE" == true && "$SUFFIX" == "$RESUME_SUFFIX" && "$DS_NAME" != "$RESUME_DATASET" ]]; then
            echo "$(date): [步骤 $STEP] 跳过数据集 $DS_NAME ($SUFFIX)，等待恢复点 ${RESUME_DATASET}/${RESUME_ALGORITHM}。"
            STEP=$((STEP + 1))
            continue
        fi

        if [[ "$RESUME_ACTIVE" == true && "$SUFFIX" == "$RESUME_SUFFIX" && "$DS_NAME" == "$RESUME_DATASET" ]]; then
            TMP_JSON_FILE=$(mktemp)
            if ! jq --arg alg "$RESUME_ALGORITHM" '
                .experiments |= (
                  reduce .[] as $exp (
                    {matched:false, out:[]};
                    if .matched then
                      .out += [$exp]
                    else
                      (
                        reduce $exp.tasks[] as $task (
                          {matched:false, out:[]};
                          if .matched then
                            .out += [$task]
                          else
                            (($task.algorithms | index($alg)) // null) as $idx
                            | if $idx == null then
                                .
                              else
                                .matched = true
                                | .out += [($task | .algorithms |= .[$idx:])]
                              end
                          end
                        )
                      ) as $task_state
                      | if $task_state.matched then
                          .matched = true
                          | .out += [($exp | .tasks = $task_state.out)]
                        else
                          .
                        end
                    end
                  )
                  | if .matched then .out else error("未在 JSON 中找到恢复算法: " + $alg) end
                )
            ' "$JSON_FILE" > "$TMP_JSON_FILE"; then
                rm -f "$TMP_JSON_FILE"
                exit 1
            fi
            RUN_JSON_FILE="$TMP_JSON_FILE"
            echo "$(date): [步骤 $STEP] 已命中恢复点：${SUFFIX}/${DS_NAME}/${RESUME_ALGORITHM}。"
            echo "   >> 将从该算法开始继续执行，后续任务保持原顺序。"
            RESUME_ACTIVE=false
        fi

        echo "$(date): [步骤 $STEP] 正在处理数据集: $DS_NAME ($SUFFIX)"
        echo "   >> 配置文件: $RUN_JSON_FILE"
        echo "   >> 运行日志: $OUTPUT_LOG"
        echo "   >> 汇总性能数据: $PERF_SUMMARY_LOG"

        perf stat -e "$PERF_EVENTS" -o "$PERF_SUMMARY_LOG" \
            ./exp.sh "$RUN_JSON_FILE" > "$OUTPUT_LOG" 2>&1

        if [[ "$RUN_JSON_FILE" != "$JSON_FILE" ]]; then
            rm -f "$RUN_JSON_FILE"
        fi

        echo "$(date): [步骤 $STEP] 数据集 $DS_NAME ($SUFFIX) 任务执行完毕。"
        echo "----------------------------------------------------------------"
        STEP=$((STEP + 1))
    done
done

echo "$(date): === 所有拆分配置批量实验已全部执行结束！日志存放在 $LOG_ROOT 文件夹下。 ==="
