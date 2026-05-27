import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 核心配置区
# ==========================================
datasets_config = {
    "Genome": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Genome/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/Genome/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma":      "/home/fengxiaoyao/FilterVector/FilterVectorResults/Genome/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Genome/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Reviews": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Reviews/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/Reviews/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "NaviX":      "/home/fengxiaoyao/FilterVector/FilterVectorResults/Reviews/Results/NaviX-ACORN/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Reviews/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Amazon": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "NaviX":      "/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Results/NaviX-ACORN/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "VariousImg": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/VariousImg/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/VariousImg/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/FilterVector/FilterVectorResults/VariousImg/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv", 
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/VariousImg/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Music": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv", 
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "BookReviews": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/BookReviews/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls500-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/BookReviews/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls500-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/FilterVector/FilterVectorResults/BookReviews/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls500-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv", 
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/BookReviews/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls500-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Tiktok": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Tiktok/Results/SmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/Tiktok/Results/UNG+/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Tiktok/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv", 
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Tiktok/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Laion": {
        "SmartRoute": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Laion/Results/SmartRoute/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_C_D-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "UNG+":       "/home/fengxiaoyao/FilterVector/FilterVectorResults/Laion/Results/UNG+/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_C_D-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "NaviX":      "/home/fengxiaoyao/FilterVector/FilterVectorResults/Laion/Results/NaviX-ACORN/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_C_D-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "pre-filter": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Laion/Results/pre-filter/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_C_D-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv"
    }
}

# ==========================================
# 算法分组配置 (解耦日志ID、文件键名与展示名称)
# ==========================================
algo_group_configs = {
    'Unified graph': {
        'file_keys': ['UNG+'],           # 在 datasets_config 中可能出现的键名
        'algo_ids': [8]                  # 在 SmartRoute 日志中的 Algo_Choice ID
    },
    'Proximity graph': {
        'file_keys': ['NaviX', 'ACORN-gamma'], # 兼容这两种键名输入
        'algo_ids': [4, 2]                     # 兼容这两种ID（4和2都会被归为此类）
    },
    'Inverted file Index': {
        'file_keys': ['pre-filter'],
        'algo_ids': [5]
    }
}

TIME_METRIC = 'search_time_ms'

# ==========================================
# 2. 数据处理与过滤主循环
# ==========================================
all_results = []

for dataset_name, paths in datasets_config.items():
    
    # 检查当前数据集是否至少有 SmartRoute 文件
    if "SmartRoute" not in paths or not os.path.exists(paths["SmartRoute"]):
        print(f"[跳过] 数据集 {dataset_name} 缺失 SmartRoute 文件。")
        continue

    # 读取 SmartRoute 数据
    df_sr = pd.read_csv(paths["SmartRoute"])
    sr_agg = df_sr.groupby(['QueryID', 'Algo_Choice'])[TIME_METRIC].mean().reset_index()

    # 遍历三大算法组进行清洗与对比
    for display_name, config in algo_group_configs.items():
        
        # 1. 提取 SmartRoute 中属于该组 ID 的记录
        sr_target = sr_agg[sr_agg['Algo_Choice'].isin(config['algo_ids'])]
        total_queries = len(sr_target)
        
        if total_queries == 0:
            continue
            
        # 2. 动态寻找独立的 Baseline 数据文件
        df_base = None
        for key in config['file_keys']:
            if key in paths and os.path.exists(paths[key]):
                df_base = pd.read_csv(paths[key])
                break # 找到任意一个匹配的文件即可跳出
                
        if df_base is None:
            print(f"  > [警告] 数据集 {dataset_name} 中找不到对应 {display_name} 的 Baseline 文件 (尝试过键名: {config['file_keys']})")
            continue
            
        base_agg = df_base.groupby('QueryID')[TIME_METRIC].mean().reset_index()
        
        # 3. 按 QueryID 对齐两张表
        merged_df = pd.merge(sr_target, base_agg, on='QueryID', suffixes=('_sr', '_base'))
        
        # 4. 数据清洗 (过滤异常偏快的查询)
        valid_df = merged_df[merged_df[f'{TIME_METRIC}_sr'] >= merged_df[f'{TIME_METRIC}_base']]
        valid_queries = len(valid_df)
        
        if valid_queries > 0:
            avg_sr_time = valid_df[f'{TIME_METRIC}_sr'].mean()
            avg_base_time = valid_df[f'{TIME_METRIC}_base'].mean()
            increase_pct = ((avg_sr_time - avg_base_time) / avg_base_time) * 100 if avg_base_time > 0 else 0
            
            all_results.append({
                'Dataset': dataset_name,
                'Algorithm': display_name,
                'Valid/Total Queries': f"{valid_queries}/{total_queries}",
                'Baseline Time (ms)': round(avg_base_time, 4),
                'SmartRoute Time (ms)': round(avg_sr_time, 4),
                'Penalty (%)': round(increase_pct, 2)
            })
        else:
            all_results.append({
                'Dataset': dataset_name,
                'Algorithm': display_name,
                'Valid/Total Queries': f"0/{total_queries}",
                'Baseline Time (ms)': None,
                'SmartRoute Time (ms)': None,
                'Penalty (%)': None
            })

# ==========================================
# 3. 统一生成表格输出
# ==========================================
if all_results:
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*75)
    print("🚀 SmartRoute vs Standalone Baseline 性能开销分析 (清洗异常偏快数据后)")
    print("="*75 + "\n")
    
    # 打印格式化的 Markdown 表格
    print(results_df.to_markdown(index=False, colalign=("center", "center", "center", "right", "right", "right")))
    
    print("\n[注] Valid/Total Queries 表示: 符合(SmartRoute >= Baseline)预期的数据条数 / 总被路由数据条数")
else:
    print("未提取到有效数据，请检查文件路径是否正确。")
