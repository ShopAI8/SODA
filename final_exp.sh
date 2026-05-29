#!/bin/bash

LOG_DIR="log"
mkdir -p "$LOG_DIR"
cd /home/lijiakang/FilterVector/FilterVectorCode
echo "=== Starting the batch experiment workflow ==="
echo "All execution logs will be saved to: $(pwd)/$LOG_DIR"

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
        echo "$(date): [Step $i] Warning: Configuration file $JSON_FILE was not found. Skipping."
        continue
    fi

    echo "$(date): [Step $i] Processing dataset: $DS_NAME"
    echo "   >> Execution log: $OUTPUT_LOG"
    echo "   >> Performance summary: $PERF_SUMMARY_LOG"

    perf stat -e "$PERF_EVENTS" -o "$PERF_SUMMARY_LOG" \
        ./exp.sh "$JSON_FILE" > "$OUTPUT_LOG" 2>&1

    echo "$(date): [Step $i] Completed processing for dataset: $DS_NAME"
    echo "----------------------------------------------------------------"
done

echo "$(date): === All batch experiments have finished. Logs are available in the $LOG_DIR directory. ==="
