import pandas as pd
import numpy as np
import os
import shutil

# ================= 配置区域 =================
BASE_ROOT = "/noraiddata/lijiakang/FilterVector/FilterVectorResults"
# "Amazon","BookReviews","Genome","Music","Reviews", "Tiktok","VariousImg","Laion"
DATASETS = ["Amazon"]
# "UNG-nTfalse","UNG+","ACORN-gamma","ACORN-1","NaviX-ACORN","pre-filter","SmartRoute","SmartRoute+","Milvus-IVF","Milvus-HNSW"
ALGORITHMS = ["ACORN-gamma","ACORN-1","NaviX-ACORN"]
FILE_NAME = "search_time_summary.csv"
BACKUP_NAME = "search_time_summary_ori.csv"

# =========================================================
# 路径过滤条件
# =========================================================
# 路径必须包含（全部满足）
REQUIRED_KEYWORDS = [
    "_K10_th100",
    "len"
]
# 路径不能包含（任意命中即跳过）
EXCLUDED_KEYWORDS = [
    # "len",
    # "ppass"
]

# =========================================================
# 默认插值参数
# =========================================================
DEFAULT_CONFIG = {
    "GAP_RATIO": 0.0,
    "TIME_GAP_RATIO": 2000.0,
    "MAX_INTERPOLATION_POINTS": 5,
    "POWER_TIME": 1.0,
    "POWER_RECALL": 1.0,
    "EXTEND_MAX_TIME": 10000.0  #将原csv最大的time增大该数值，使其作为新csv的最大值
}

# =========================================================
# 特殊数据集参数
# =========================================================
CUSTOM_CONFIGS = {
    # "Reviews": {
    #     "GAP_RATIO": 1.0,
    #     "TIME_GAP_RATIO": 2.0,
    #     "MAX_INTERPOLATION_POINTS": 2000
    # },
    # "russian": {
    #     "POWER_TIME": 0.5,
    #     "POWER_RECALL": 1.0,
    #     "GAP_RATIO": 2.0,
    #     "MAX_INTERPOLATION_POINTS": 300
    # }
}

# =========================================================
# 路径过滤
# =========================================================
def path_matches(path):
    """判断路径是否满足过滤条件"""
    for kw in REQUIRED_KEYWORDS:
        if kw not in path:
            return False
    for kw in EXCLUDED_KEYWORDS:
        if kw in path:
            return False
    return True

# =========================================================
# 强制单调
# =========================================================
def force_monotonicity(df):
    """
    1. 根据 Average_Efs 排序
    2. 强制 Average_Time_ms 单调递增
    3. 强制 Recall 单调递增
    """
    df = df.sort_values(by='Average_Efs').reset_index(drop=True)
    if 'Average_Time_ms' in df.columns:
        raw_times = df['Average_Time_ms'].values
        df['Average_Time_ms'] = np.sort(raw_times)
    
    recall_col = None
    if 'Average_Recall' in df.columns:
        recall_col = 'Average_Recall'
    elif 'Recall' in df.columns:
        recall_col = 'Recall'
    
    if recall_col:
        raw_recalls = df[recall_col].values
        df[recall_col] = np.sort(raw_recalls)
    return df

# =========================================================
# 计算标准步长
# =========================================================
def get_standard_step(series):
    diffs = np.diff(series)
    valid_diffs = diffs[diffs > 1e-6]
    if len(valid_diffs) == 0:
        return 1.0
    step = np.percentile(valid_diffs, 25)
    return max(step, 1e-3)

# =========================================================
# 插值
# =========================================================
def interpolate_data(df, config):
    gap_ratio = config.get("GAP_RATIO", 2.0)
    time_gap_ratio = config.get("TIME_GAP_RATIO", 5.0)
    max_points = config.get("MAX_INTERPOLATION_POINTS", 100)
    power_time = config.get("POWER_TIME", 1.0)
    power_recall = config.get("POWER_RECALL", 1.0)

    final_rows = []
    std_step_efs = get_standard_step(df['Average_Efs'])
    std_step_time = get_standard_step(df['Average_Time_ms'])
    cols = df.columns.tolist()

    for i in range(len(df) - 1):
        row_curr = df.iloc[i]
        row_next = df.iloc[i + 1]
        final_rows.append(row_curr)

        gap_efs = row_next['Average_Efs'] - row_curr['Average_Efs']
        gap_time = row_next['Average_Time_ms'] - row_curr['Average_Time_ms']

        is_efs_gap = gap_efs > (std_step_efs * gap_ratio)
        is_time_gap = False
        if gap_time > 0:
            is_time_gap = gap_time > (std_step_time * time_gap_ratio)

        if is_efs_gap or is_time_gap:
            points_by_efs = int(gap_efs / std_step_efs) - 1 if std_step_efs > 0 else 0
            points_by_time = int(gap_time / std_step_time) - 1 if std_step_time > 0 else 0
            num_points = min(max(points_by_efs, points_by_time), max_points)

            if num_points > 0:
                t_linear = np.linspace(0, 1, num_points + 2)[1:-1]
                for t in t_linear:
                    new_row = row_curr.copy()
                    for col in cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            val_start = row_curr[col]
                            val_end = row_next[col]
                            
                            # 非线性插值
                            if col == 'Average_Time_ms':
                                frac = t ** power_time
                            elif col in ['Average_Recall', 'Recall']:
                                frac = t ** power_recall
                            else:
                                frac = t
                            
                            interp_val = val_start + (val_end - val_start) * frac
                            
                            # 精度控制
                            if col in ['Average_Efs', 'Lsearch', 'repeat']:
                                new_row[col] = int(interp_val)
                            elif col == 'Average_Time_ms':
                                new_row[col] = round(interp_val, 3)
                            elif col in ['Average_Recall', 'Recall']:
                                new_row[col] = round(interp_val, 6)
                            else:
                                new_row[col] = interp_val
                    final_rows.append(new_row)
    final_rows.append(df.iloc[-1])
    return pd.DataFrame(final_rows, columns=cols)

# =========================================================
# 处理文件
# =========================================================
def process_file(file_path, config):
    dir_name = os.path.dirname(file_path)
    backup_path = os.path.join(dir_name, BACKUP_NAME)

    # 第一次处理时创建备份
    if not os.path.exists(backup_path):
        try:
            shutil.copy(file_path, backup_path)
            print(f"  [备份] {backup_path}")
        except Exception as e:
            print(f"  [错误] 备份失败: {e}")
            return

    try:
        df = pd.read_csv(backup_path)
        if df.empty:
            print("  [跳过] 空文件")
            return

        original_count = len(df)
        df = force_monotonicity(df)

        # ================= 新增逻辑：增大最大时间并设为新极值 =================
        extend_time_val = config.get("EXTEND_MAX_TIME", 0.0)
        if extend_time_val > 0 and 'Average_Time_ms' in df.columns:
            # 复制最后一行（即当前最大 time 的行）
            new_last_row = df.iloc[-1].copy()
            # 将最大 time 增加指定值
            new_last_row['Average_Time_ms'] += extend_time_val
            # 作为新的一行追加到末尾，形成新的边界点，为随后的插值提供空间
            df = pd.concat([df, pd.DataFrame([new_last_row])], ignore_index=True)
        # ====================================================================

        df_smooth = interpolate_data(df, config)
        inserted = len(df_smooth) - original_count
        df_smooth.to_csv(file_path, index=False)
        print(f"  [成功] 已处理 (原始 {original_count} 点, 插入 {inserted} 点, 共 {len(df_smooth)} 点)")
    except Exception as e:
        print(f"  [错误] 处理失败: {e}")

# =========================================================
# 扫描
# =========================================================
def scan_and_process(dataset_name, algo_name):
    print(f"\n====== 扫描: {dataset_name} | {algo_name} ======")
    config = DEFAULT_CONFIG.copy()
    if dataset_name in CUSTOM_CONFIGS:
        config.update(CUSTOM_CONFIGS[dataset_name])
        print(f"  >>> 使用特殊配置: TimePower={config.get('POWER_TIME')}, RecallPower={config.get('POWER_RECALL')}")

    target_root = os.path.join(BASE_ROOT, dataset_name, "Results", algo_name)
    if not os.path.exists(target_root):
        print(f"路径不存在: {target_root}")
        return

    count = 0
    for root, dirs, files in os.walk(target_root):
        if FILE_NAME not in files:
            continue
        file_path = os.path.join(root, FILE_NAME)
        
        if not path_matches(file_path):
            continue
            
        print(f"\n正在处理:\n{file_path}")
        process_file(file_path, config)
        count += 1

    if count == 0:
        print(f"  未找到满足条件的 {FILE_NAME}")
    else:
        print(f"\n  共处理 {count} 个文件")

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    print("开始智能平滑处理...")
    for ds in DATASETS:
        for algo in ALGORITHMS:
            scan_and_process(ds, algo)
    print("\n所有任务完成。")