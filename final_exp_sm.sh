#!/bin/bash
set -e

echo "=== 开始执行实验... ==="

# echo "$(date): [步骤 1] 运行 experiments-Reviews-1-2..."
# cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Reviews-1-2.json

echo "$(date): [步骤 2] 运行 experiments-Genome-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-Genome-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 0] 运行 experiments-Amazon-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-Amazon-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 1] 运行 experiments-BookReviews-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-BookReviews-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 3] 运行 experiments-Music-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-Music-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 4] 运行 experiments-Reviews-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-Reviews-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 5] 运行 experiments-Tiktok-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-Tiktok-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 6] 运行 experiments-VariousImg-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-VariousImg-baseline-1000-1.json > output_sm.log

echo "$(date): [步骤 7] 运行 experiments-Laion-baseline..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/202604-smartroute_ljk/experiments-Laion-baseline-1000-1.json > output_sm.log

# echo "$(date): [步骤 8] 运行 experiments-Laion-ACORN-big..."
# cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Laion-ACORN-big.json > output_sm.log



echo "$(date): === 所有任务执行完毕。 ==="