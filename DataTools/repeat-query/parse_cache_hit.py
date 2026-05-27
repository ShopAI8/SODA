import os
import glob
import re
import csv

# ================= 配置区域 =================
BASE_DIR = "/noraiddata/lijiakang/FilterVector/FilterVectorResults"
DATASETS = ["Amazon","BookReviews","Genome","Music","Reviews", "Tiktok","VariousImg","Laion"] 
ALGORITHMS = ["SmartRoute", "SmartRoute+"]
OUTPUT_CSV = "/noraiddata/lijiakang/FilterVector/FilterVectorResults/SelectModels_summary/cache_hit_rates_summary.csv"
# ============================================

def get_raw_count(filepath, keyword):
    """
    通过正则从日志中提取特定指标的绝对数字 (Raw Count)
    匹配模式例如: "      1,234,567      L1-dcache-loads"
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # 匹配数字（包含逗号）和关键字
            pattern = rf"^\s*([\d,]+)\s+{keyword}"
            # 使用 re.MULTILINE 以便 ^ 能匹配每一行的开头
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                # 移除逗号并转为整数
                raw_count_str = match.group(1).replace(',', '')
                return int(raw_count_str)
    except Exception as e:
        print(f"Error reading {keyword} from {filepath}: {e}")
        return None
    return None

def calculate_rate(misses, total):
    """安全地计算命中率并格式化为百分比字符串"""
    if misses is None or total is None or total == 0:
        return "N/A"
    hit_rate = (1.0 - (misses / total)) * 100.0
    # 限制命中率在 0-100% 之间（防止极端情况下 perf 采样误差导致微小负数）
    hit_rate = max(0.0, min(100.0, hit_rate))
    return f"{hit_rate:.2f}%"

def main():
    results = []
    
    # 表头设计
    headers = ["Dataset", "Algorithm", "L1 D-Cache Hit Rate", "L2 Hit Rate", "LLC Hit Rate", "Overall Cache Hit Rate"]
    results.append(headers)
    print(f"{headers[0]:<15} | {headers[1]:<15} | {headers[2]:<20} | {headers[3]:<15} | {headers[4]:<15} | {headers[5]:<25}")
    print("-" * 115)

    for dataset in DATASETS:
        for algo in ALGORITHMS:
            # 动态构建路径
            search_pattern = os.path.join(
                BASE_DIR,
                dataset,
                "Results",
                algo,
                "Index*_GT*_Search*",
                "others",
                f"{dataset}_perf_stat.log"
            )
            
            matched_files = glob.glob(search_pattern)
            
            # 过滤逻辑
            valid_files = [f for f in matched_files if "query_select_200" in f and "len" not in f and "ppass" not in f]
            
            if not valid_files:
                print(f"{dataset:<15} | {algo:<15} | {'No Valid File Found':<20} | {'-':<15} | {'-':<25}")
                results.append([dataset, algo, "N/A", "N/A", "N/A"])
                continue
            
            log_file = valid_files[0]
            
            # 提取 4 个核心的绝对数字
            l1_loads = get_raw_count(log_file, "L1-dcache-loads")
            l1_misses = get_raw_count(log_file, "L1-dcache-load-misses")
            llc_loads = get_raw_count(log_file, "LLC-loads")
            llc_misses = get_raw_count(log_file, "LLC-load-misses")
            l2_loads = get_raw_count(log_file, "l2_rqsts.all_demand_data_rd")
            l2_misses = get_raw_count(log_file, "l2_rqsts.demand_data_rd_miss")
            
            # 如果 LLC 没抓到，尝试退回抓取通用的 cache-misses 和 cache-references
            if llc_loads is None:
                llc_loads = get_raw_count(log_file, "cache-references")
            if llc_misses is None:
                llc_misses = get_raw_count(log_file, "cache-misses")
                
            # 计算命中率
            l1_hit_rate = calculate_rate(l1_misses, l1_loads)
            l2_hit_rate = calculate_rate(l2_misses, l2_loads) # 新增 L2 计算
            llc_hit_rate = calculate_rate(llc_misses, llc_loads)
            overall_hit_rate = calculate_rate(llc_misses, l1_loads) 
            
            # 打印并加入结果列表
            print(f"{dataset:<15} | {algo:<15} | {l1_hit_rate:<20} | {l2_hit_rate:<15} | {llc_hit_rate:<15} | {overall_hit_rate:<25}")
            results.append([dataset, algo, l1_hit_rate, l2_hit_rate, llc_hit_rate, overall_hit_rate])


    # 导出到 CSV 文件
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    
    print("-" * 100)
    print(f"✅ 统计完成！结果已导出至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()