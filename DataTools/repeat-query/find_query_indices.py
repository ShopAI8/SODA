import pandas as pd

def find_indices_after_dedup(csv_path, target_qid, repeat_times=300):
    # 1. 读取并立即去重
    df = pd.read_csv(csv_path)[['QueryID', 'Algo_Choice']]
    df_clean = df.drop_duplicates('QueryID').copy()
    
    print(f"原始 CSV 行数: {len(df)}")
    print(f"去重后唯一查询数: {len(df_clean)}")

    # 2. 模拟 process 函数的分组排序逻辑
    # groupby 默认会对 Algo_Choice 进行升序排序 (0 -> 2 -> 5)
    grouped = df_clean.groupby('Algo_Choice')
    
    current_idx = 0
    found = False

    for algo, group in grouped:
        for _, row in group.iterrows():
            qid = int(row['QueryID'])
            if qid == target_qid:
                start = current_idx
                end = current_idx + repeat_times - 1
                print("\n定位成功！")
                print("-" * 30)
                print(f"QueryID: {qid}")
                print(f"所属算法组: {algo}")
                print(f"索引区间: {start} 到 {end}")
                print(f"文件偏移参考: 约第 {start} 条向量处")
                found = True
            
            current_idx += repeat_times
            
    if not found:
        print(f"错误：在去重后的数据中未找到 QueryID {target_qid}")

# 使用方法
csv_file = "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Reviews/Results/FastSmartRoute/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls10-Le500-Lp10_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv" 
find_indices_after_dedup(csv_file, 10, 300)