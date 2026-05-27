#!/bin/bash

LOG_DIR="log"
mkdir -p "$LOG_DIR"
cd /home/lijiakang/FilterVector/FilterVectorCode
echo "=== 启动批量实验任务流程 ==="
echo "所有执行记录将保存至：$(pwd)/$LOG_DIR"

DATASETS=(
    "Genome"
    "Amazon"
    "Reviews"
)

PERF_EVENTS="cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss,LLC-loads,LLC-load-misses,branches,branch-misses"

for i in "${!DATASETS[@]}"; do
    DS_NAME=${DATASETS[$i]}
    JSON_FILE="experiment_json/202604-200-random-300-mix-th-K/experiments-${DS_NAME}-200-random-300-mix-len.json"
    OUTPUT_LOG="$LOG_DIR/${DS_NAME}_output.log"
    PERF_SUMMARY_LOG="$LOG_DIR/${DS_NAME}_perf_summary.log"

    if [ ! -f "$JSON_FILE" ]; then
        echo "$(date): [步骤 $i] 警告：找不到配置文件 $JSON_FILE，跳过。"
        continue
    fi

    echo "$(date): [步骤 $i] 正在处理数据集: $DS_NAME"
    echo "   >> 运行日志: $OUTPUT_LOG"
    echo "   >> 汇总性能数据: $PERF_SUMMARY_LOG"

    perf stat -e "$PERF_EVENTS" -o "$PERF_SUMMARY_LOG" \
        ./exp.sh "$JSON_FILE" > "$OUTPUT_LOG" 2>&1

    echo "$(date): [步骤 $i] 数据集 $DS_NAME 任务执行完毕。"
    echo "----------------------------------------------------------------"
done

echo "$(date): === 所有批量实验已全部执行结束！日志存放在 $LOG_DIR 文件夹下。 ==="