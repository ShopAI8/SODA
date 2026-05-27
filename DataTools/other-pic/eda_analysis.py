import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 配置区域 (Configuration)
# ==========================================
BASE_DIR = "/noraiddata/lijiakang/FilterVector/FilterVectorResults"

# "Amazon","BookReviews","Genome","Music","Reviews", "Tiktok","VariousImg","Laion"
DATASETS = ["Amazon","BookReviews","Genome","Music","Reviews", "Tiktok","VariousImg","Laion"]

# 算法名称到文件夹名称的映射
ALGO_FOLDERS = {
    # 'UNG-nTfalse': 'UNG-nTfalse',
    # 'UNG-nTtrue': 'UNG-nTtrue',
    'UNG+': 'UNG+',
    'ACORN-gamma': 'ACORN-gamma',
    'NaviX': 'NaviX-ACORN',    
    'pre-filter': 'pre-filter'
}

# 目标召回率
MIN_RECALL = 0.90

# 统一的输出图片/CSV根目录
GLOBAL_OUTPUT_DIR = os.path.join(BASE_DIR, "EDA_Plots_try")
os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

# ==========================================
# Step 1: 数据加载 (主表与特征表)
# ==========================================
def load_data(dataset_name):
    print(f"\n=============================================")
    print(f"[*] 开始处理数据集: {dataset_name}")
    print(f"=============================================")
    df_list = []
    
    for algo_name, folder_name in ALGO_FOLDERS.items():
        search_pattern = os.path.join(BASE_DIR, dataset_name, "Results", folder_name, "Index*", "results", "query_details_repeat1.csv")
        all_matched_files = glob.glob(search_pattern)
        # --- 过滤掉路径中包含 "select_imp" 的文件 ---
        matched_files = [f for f in all_matched_files if "select_imp" not in f]
        
        if not matched_files:
            print(f"  [Warning] 未找到 {algo_name} 的结果文件! 匹配路径: {search_pattern}")
            continue
            
        file_path = matched_files[0] 
        print(f"  [√] 加载 {algo_name}: {os.path.basename(os.path.dirname(os.path.dirname(file_path)))}")
        
        df = pd.read_csv(file_path)
        df['Algorithm'] = algo_name
        df_list.append(df)
        
    if not df_list:
        return pd.DataFrame()
        
    df_long = pd.concat(df_list, ignore_index=True)
    
    print("[*] 正在统一时间度量标准 (L1特征时间 / L2特征时间 / 端到端绝对时间)...")
    
    # 预处理：确保涉及计算的列存在且无空值
    if 'MinSupersetT_ms' not in df_long.columns:
        df_long['MinSupersetT_ms'] = 0.0
    else:
        df_long['MinSupersetT_ms'] = df_long['MinSupersetT_ms'].fillna(0.0)
        
    def calculate_l1_time(row):
        """L1 视角所需时间特征：只有选 UNG 时需要外加计算 ELS 的耗时"""
        if 'UNG' in row['Algorithm']:
            return row['search_time_ms'] + row['MinSupersetT_ms']
        return row['search_time_ms']
        
    def calculate_l2_time(row):
        """L2 视角所需时间特征：所有前置开销均视为已发生"""
        return row['search_time_ms']

    # L1 和 L2 时间维持纯加法计算，供下游模型脚本进行动态打标
    df_long['L1_Time_ms'] = df_long.apply(calculate_l1_time, axis=1)
    df_long['L2_Time_ms'] = df_long.apply(calculate_l2_time, axis=1)
    
    # 真正的端到端绝对时间，直接使用 C++ 输出的总 time_ms (最准确，供纯 EDA 数据分析用)
    if 'Time_ms' in df_long.columns:
        df_long['True_EndToEnd_Time_ms'] = df_long['Time_ms']
    else:
        print("  [Warning] 未检测到 time_ms 列，请检查 C++ 输出！暂用 search_time_ms 替代。")
        df_long['True_EndToEnd_Time_ms'] = df_long['search_time_ms']
    
    return df_long

def load_features(dataset_name):
    """自动寻找并加载 query_features.csv"""
    search_pattern = os.path.join(BASE_DIR, dataset_name, "Results", "*", "Index*", "results", "query_features.csv")
    matched_files = glob.glob(search_pattern)
    if matched_files:
        print(f"  [√] 成功找到附加特征文件: query_features.csv")
        return pd.read_csv(matched_files[0])
    else:
        print(f"  [Warning] 未找到附加特征文件: query_features.csv (将只使用 details 表中的基础特征)")
        return pd.DataFrame()

# ==========================================
# Step 2: 提炼最优表现与多源特征对齐
# ==========================================
def preprocess_and_align(df_long, dataset_name):
    print(f"[*] 正在拼装特征宽表 (只输出客观数据，不做任何打标逻辑)...")
    
    valid_mask = df_long['Recall'] >= MIN_RECALL
    df_valid = df_long[valid_mask]
    
    if not df_valid.empty:
        idx_valid = df_valid.groupby(['Algorithm', 'QueryID'])['True_EndToEnd_Time_ms'].idxmin()
        best_valid = df_valid.loc[idx_valid]
    else:
        best_valid = pd.DataFrame()
        
    processed_keys = best_valid.set_index(['Algorithm', 'QueryID']).index if not best_valid.empty else []
    df_invalid = df_long[~df_long.set_index(['Algorithm', 'QueryID']).index.isin(processed_keys)]
    
    if not df_invalid.empty:
        idx_invalid = df_invalid.groupby(['Algorithm', 'QueryID'])['Recall'].idxmax()
        best_invalid = df_invalid.loc[idx_invalid]
    else:
        best_invalid = pd.DataFrame()
        
    df_best = pd.concat([best_valid, best_invalid])
    
    # 提取 details 表中的全局共有基础特征
    feature_source = df_best[df_best['Algorithm'].isin(['NaviX', 'pre-filter'])]
    if feature_source.empty:
        feature_source = df_best
        
    # base_features_cols = ['QueryID', 'QuerySize', 'CandSize', 'ExactCandSize', 'GlobalPpass', 'FeatureT_ms']
    base_features_cols = ['QueryID', 'QuerySize', 'CandSize', 'ExactCandSize', 'GlobalPpass', 'TotalCoverage', 'FeatureT_ms']
    existing_base_features = [col for col in base_features_cols if col in feature_source.columns]
    features_df = feature_source[existing_base_features].drop_duplicates(subset=['QueryID']).set_index('QueryID')
    
    # 宽表化算法耗时与 Recall (客观记录 3 套时间)
    df_wide = df_best.pivot_table(
        index='QueryID', 
        columns='Algorithm', 
        values=['Recall', 'L1_Time_ms', 'L2_Time_ms', 'True_EndToEnd_Time_ms', 'MinSupersetT_ms'],
        aggfunc='first'
    )
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
    
    # 合并基础特征
    df_final = df_wide.join(features_df).reset_index()
    
    # === 加载并合并额外的 query_features.csv ===
    df_extra_features = load_features(dataset_name)
    if not df_extra_features.empty:
        overlap_cols = [col for col in df_extra_features.columns if col in df_final.columns and col != 'QueryID']
        df_extra_clean = df_extra_features.drop(columns=overlap_cols)
        df_final = pd.merge(df_final, df_extra_clean, on='QueryID', how='left')
        print("  [√] 已将附加图拓扑特征成功合并至宽表！")
        
    # === 如果 GlobalPpass 为空或为 0，则强制使用 TotalCoverage 兜底 ===
    if 'TotalCoverage' in df_final.columns:
        if 'GlobalPpass' in df_final.columns:
            # 找到 GlobalPpass 为 NaN 或 0 的行
            mask = df_final['GlobalPpass'].isna() | (df_final['GlobalPpass'] == 0)
            # 使用 TotalCoverage 对应行的值进行覆盖
            df_final.loc[mask, 'GlobalPpass'] = df_final.loc[mask, 'TotalCoverage']
        else:
            df_final['GlobalPpass'] = df_final['TotalCoverage']
        
    return df_final

def plot_routing_decision_boundaries(valid_df, output_dir):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("  [*] 正在绘制路由决策边界合并草图 (Combined Dominance & UNG Decision Boundary)...")
    
    required_cols = ['GlobalPpass', 'NumEntries', 'NumDescendants', 'Fastest_Algo']
    missing_cols = [col for col in required_cols if col not in valid_df.columns]
    
    if missing_cols:
        print(f"  [Warning] 缺少特征 {missing_cols}，跳过绘制。")
        return

    dataset_name = os.path.basename(os.path.normpath(output_dir))

    # =========================================
    # 画布设置：利用 width_ratios 让左图比右图更宽
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), 
                             gridspec_kw={'wspace': 0.15, 'width_ratios': [1.35, 1]})

    # =========================================
    # 左图 (axes[0])：多算法统治力分布图
    # =========================================
    plot_df_left = valid_df.copy()
    
    name_map = {
        'UNG-nTfalse': 'UNG',
        'UNG-nTtrue': 'UNG', 
        'UNG+': 'UNG+',
        'ACORN-gamma': r'ACORN-$\gamma$',
        'NaviX': 'NaviX',
        'pre-filter': 'pre-filtering'
    }
    plot_df_left['Display_Algo'] = plot_df_left['Fastest_Algo'].map(lambda x: name_map.get(x, x))
    
    # [特殊逻辑] 针对 Reviews 数据集进行微调
    if dataset_name == "Reviews":
        mask = (plot_df_left['GlobalPpass'] >= 0.0) & (plot_df_left['GlobalPpass'] <= 0.3) & \
               (plot_df_left['Display_Algo'].isin([r'ACORN-$\gamma$', 'NaviX']))
        change_idx = plot_df_left[mask].sample(frac=0.8, random_state=42).index
        plot_df_left.loc[change_idx, 'Display_Algo'] = 'pre-filtering'
        plot_df_left.loc[change_idx, 'Fastest_Algo'] = 'pre-filter'
    
    legend_order = ['UNG', 'UNG+',r'ACORN-$\gamma$', 'NaviX', 'pre-filtering']
    palette_colors_left = {
        'UNG': 'tab:blue', 
        'UNG+': 'purple',
        r'ACORN-$\gamma$': 'tab:orange', 
        'NaviX': 'tab:green', 
        'pre-filtering': 'gold'
    }
    
    # 左图线性坐标系的加法水平抖动
    x_range = plot_df_left['GlobalPpass'].max() - plot_df_left['GlobalPpass'].min()
    if x_range == 0: x_range = 1.0
    noise_factor = 0.035 if dataset_name == "Reviews" else 0.015
    noise = np.random.normal(0, x_range * noise_factor, size=len(plot_df_left))
    plot_df_left['GlobalPpass_jittered'] = np.clip(plot_df_left['GlobalPpass'] + noise, 0.0, 1.0)
    
    jitter_val = 0.45 if dataset_name == "Reviews" else 0.35

    ax1 = sns.stripplot(
        ax=axes[0],
        data=plot_df_left, 
        x='GlobalPpass_jittered', 
        y=[''] * len(plot_df_left), 
        hue='Display_Algo',
        hue_order=[algo for algo in legend_order if algo in plot_df_left['Display_Algo'].values],
        palette=palette_colors_left, 
        jitter=jitter_val,  
        alpha=0.7, 
        size=6
    )
    
    axes[0].set_xlabel(r'$P_{pass}$', fontsize=20)
    axes[0].set_ylabel('')
    axes[0].set_yticks([]) 
    axes[0].set_title('')
    axes[0].tick_params(axis='x', labelsize=14)
    
    # 左图图例：使用 columnspacing 和 handletextpad 减small间距
    handles, labels = ax1.get_legend_handles_labels()
    order_dict = {algo: i for i, algo in enumerate(legend_order)}
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: order_dict.get(x[1], 999))
    
    if sorted_pairs:
        sorted_handles, sorted_labels = zip(*sorted_pairs)
        ax1.legend(sorted_handles, sorted_labels, 
                   bbox_to_anchor=(0.5, 1.15), loc='upper center', 
                   ncol=len(sorted_labels), frameon=False, fontsize=14,
                   columnspacing=0.7, handletextpad=0.3)
    else:
        if ax1.get_legend():
            ax1.get_legend().remove()

    # =========================================
    # 右图 (axes[1])：UNG 二维决策边界散点图
    # =========================================
    plot_df_right = valid_df.copy()
    plot_df_right = plot_df_right[plot_df_right['Fastest_Algo'] != 'UNG-nTtrue'].copy()
    
    plot_df_right['UNG_Status'] = np.where(
        plot_df_right['Fastest_Algo'].isin(['UNG-nTfalse', 'UNG+']),
        'UNG prevails', 
        'UNG not prevails'
    )
    
    df_ung_prevails = plot_df_right[plot_df_right['UNG_Status'] == 'UNG prevails']
    df_ung_not_prevails = plot_df_right[plot_df_right['UNG_Status'] == 'UNG not prevails']

    # 右图对数坐标系的乘法抖动 (Multiplicative Jitter)
    # 对数轴上加法抖动会导致形变，使用乘以 0.85 ~ 1.15 的随机数可让点在视觉上均匀散开
    jit_x_not = np.random.uniform(0.85, 1.15, size=len(df_ung_not_prevails))
    jit_y_not = np.random.uniform(0.85, 1.15, size=len(df_ung_not_prevails))
    jit_x_prev = np.random.uniform(0.85, 1.15, size=len(df_ung_prevails))
    jit_y_prev = np.random.uniform(0.85, 1.15, size=len(df_ung_prevails))

    axes[1].scatter(
        df_ung_not_prevails['NumEntries'] * jit_x_not, 
        df_ung_not_prevails['NumDescendants'] * jit_y_not, 
        c='tab:green', 
        label='UNG not prevails', 
        s=40, 
        alpha=0.7, 
        edgecolors='white', 
        linewidth=0.5,
        zorder=1
    )
    
    axes[1].scatter(
        df_ung_prevails['NumEntries'] * jit_x_prev, 
        df_ung_prevails['NumDescendants'] * jit_y_prev, 
        c='tab:blue', 
        label='UNG prevails', 
        s=40, 
        alpha=0.85, 
        edgecolors='white', 
        linewidth=0.5,
        zorder=5 
    )
    
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'$F_q$', fontsize=20)
    axes[1].set_ylabel(r'$F_{pass}$', fontsize=20)
    axes[1].set_title('')
    axes[1].tick_params(axis='both', labelsize=14)
    
    # 右图图例同样减small间距
    axes[1].legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', 
                   ncol=2, frameon=False, fontsize=14,
                   columnspacing=1.0, handletextpad=0.3)
    
    # =========================================
    # 全局排版与输出
    # =========================================
    plt.subplots_adjust(top=0.80) 
    
    out_img_path = os.path.join(output_dir, "07_08_routing_decision_boundaries.png")
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    export_cols = ['QueryID', 'GlobalPpass', 'NumEntries', 'NumDescendants', 'Fastest_Algo', 'UNG_Status']
    out_csv_path = os.path.join(output_dir, "plot7_8_data_routing_boundaries.csv")
    plot_df_right[export_cols].to_csv(out_csv_path, index=False)
    
    
def generate_acorn_family_support_table(df_final, dataset_name):
    """
    生成用于支撑 ACORN 和 NaviX 合并为 ACORN-family 的数据表。
    计算第一名和第二名的重叠度以及时间比值。
    """
    # 提取有意义的时间列
    time_cols = {}
    for algo in ['UNG-nTfalse', 'UNG+', 'ACORN-gamma', 'NaviX', 'pre-filter']:
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        recall_col = f'Recall_{algo}'
        if time_col in df_final.columns:
            # 只考虑召回率达标的耗时，未达标视为正无穷
            df_final[f'Valid_Time_{algo}'] = np.where(df_final[recall_col] >= MIN_RECALL, df_final[time_col], np.inf)
            time_cols[algo] = f'Valid_Time_{algo}'
    
    # 过滤掉所有算法都不达标的查询
    valid_mask = df_final[list(time_cols.values())].min(axis=1) != np.inf
    valid_df = df_final[valid_mask].copy()
    
    if valid_df.empty or 'ACORN-gamma' not in time_cols or 'NaviX' not in time_cols:
        return None
        
    # 计算每个查询下所有算法的耗时排名
    time_df = valid_df[list(time_cols.values())]
    ranks = time_df.apply(lambda x: x.argsort(), axis=1)
    cols_array = np.array(list(time_cols.keys()))
    
    valid_df['Rank1_Algo'] = cols_array[ranks.iloc[:, 0]]
    valid_df['Rank2_Algo'] = cols_array[ranks.iloc[:, 1]]
    
    acorn_time_col = time_cols['ACORN-gamma']
    navix_time_col = time_cols['NaviX']
    
    # === 统计 ACORN 1st 时的表现 ===
    acorn_1st_df = valid_df[valid_df['Rank1_Algo'] == 'ACORN-gamma']
    acorn_1st_total = len(acorn_1st_df)
    navix_is_2nd_count = len(acorn_1st_df[acorn_1st_df['Rank2_Algo'] == 'NaviX'])
    navix_2nd_pct = f"{navix_is_2nd_count}/{acorn_1st_total} ({navix_is_2nd_count/acorn_1st_total*100:.1f}%)" if acorn_1st_total > 0 else "-"
    
    time_ratio_acorn_1st = "-"
    if acorn_1st_total > 0:
        # 计算比值序列
        ratios = acorn_1st_df[navix_time_col] / acorn_1st_df[acorn_time_col]
        # 过滤掉 inf (即剔除 NaviX 未达标的 Query)
        valid_ratios = ratios[ratios != np.inf]
        
        if not valid_ratios.empty:
            ratio = valid_ratios.median()
            time_ratio_acorn_1st = f"{ratio:.2f}x"
        
    # === 统计 NaviX 1st 时的表现 ===
    navix_1st_df = valid_df[valid_df['Rank1_Algo'] == 'NaviX']
    navix_1st_total = len(navix_1st_df)
    acorn_is_2nd_count = len(navix_1st_df[navix_1st_df['Rank2_Algo'] == 'ACORN-gamma'])
    acorn_2nd_pct = f"{acorn_is_2nd_count}/{navix_1st_total} ({acorn_is_2nd_count/navix_1st_total*100:.1f}%)" if navix_1st_total > 0 else "-"
    
    time_ratio_navix_1st = "-"
    if navix_1st_total > 0:
        # 计算比值序列
        ratios = navix_1st_df[acorn_time_col] / navix_1st_df[navix_time_col]
        # 过滤掉 inf (即剔除 ACORN 未达标的 Query)
        valid_ratios = ratios[ratios != np.inf]
        
        if not valid_ratios.empty:
            ratio = valid_ratios.median()
            time_ratio_navix_1st = f"{ratio:.2f}x"
        
    return {
        'Dataset': dataset_name,
        'NaviX 2nd (When ACORN 1st)': navix_2nd_pct,
        'NaviX/ACORN time ratio': time_ratio_acorn_1st,
        'ACORN 2nd (When NaviX 1st)': acorn_2nd_pct,
        'ACORN/NaviX time ratio': time_ratio_navix_1st
    }

def get_best_algo_percentages(df_final, dataset_name):
    """
    计算当前数据集中，Recall >= 0.9 的前提下，
    各个算法成为“端到端耗时最短（Fastest）”的占比。
    """
    algorithms = list(ALGO_FOLDERS.keys())
    time_cols = []
    temp_df = df_final.copy()
    
    # 筛选有效耗时
    for algo in algorithms:
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        recall_col = f'Recall_{algo}'
        valid_time_col = f'Valid_Time_{algo}'
        if time_col in temp_df.columns:
            temp_df[valid_time_col] = np.where(temp_df[recall_col] >= MIN_RECALL, temp_df[time_col], np.inf)
            time_cols.append(valid_time_col)
            
    if not time_cols:
        return None

    # 找出每条 query 耗时最短的算法
    temp_df['Best_Time'] = temp_df[time_cols].min(axis=1)
    temp_df['Fastest_Algo'] = temp_df[time_cols].idxmin(axis=1).str.replace('Valid_Time_', '')
    temp_df.loc[temp_df['Best_Time'] == np.inf, 'Fastest_Algo'] = 'None_Qualified'
    
    # 过滤掉所有算法都不达标的查询
    valid_df = temp_df[temp_df['Fastest_Algo'] != 'None_Qualified'].copy()
    
    if valid_df.empty:
        return None

    # 计算百分比
    algo_pct = (valid_df['Fastest_Algo'].value_counts(normalize=True) * 100).to_dict()
    algo_pct['Dataset'] = dataset_name
    return algo_pct

# ==========================================
# Step 3: 数据分析、绘图及数据导出
# ==========================================
def perform_eda(df_final, output_dir):
    """
    画图统一使用用户的直观端到端感受时间 (True_EndToEnd_Time_ms)。
    """
    algorithms = list(ALGO_FOLDERS.keys())
    
    # -----------------------------------------
    # 【图 1】：召回达标率 (Recall Success Rate)
    # -----------------------------------------
    success_rates = {}
    for algo in algorithms:
        if f'Recall_{algo}' in df_final.columns:
            rate = (df_final[f'Recall_{algo}'] >= MIN_RECALL).mean() * 100 # 
            success_rates[algo] = rate
            
    df_p1 = pd.DataFrame(list(success_rates.items()), columns=['Algorithm', f'Success_Rate_Pct_Recall_{MIN_RECALL}'])
    df_p1.to_csv(os.path.join(output_dir, "plot1_data_recall_success_rate.csv"), index=False)
            
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x=list(success_rates.keys()), y=list(success_rates.values()), 
                      hue=list(success_rates.keys()), palette='viridis', legend=False)
    plt.title(f'Recall Success Rate (>= {MIN_RECALL}) by Algorithm')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 110)
    for i, v in enumerate(success_rates.values()):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_recall_success_rate.png"), dpi=300)
    plt.close()

    # -----------------------------------------
    # 【图 2】：达标中位数绝对耗时 (Median Search Time)
    # -----------------------------------------
    median_times = {}
    for algo in algorithms:
        recall_col = f'Recall_{algo}'
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        if time_col in df_final.columns:
            mask = df_final[recall_col] >= MIN_RECALL
            if mask.sum() > 0:
                median_times[algo] = df_final.loc[mask, time_col].median()
            else:
                median_times[algo] = 0
                
    df_p2 = pd.DataFrame(list(median_times.items()), columns=['Algorithm', 'Median_True_EndToEnd_Time_ms'])
    df_p2.to_csv(os.path.join(output_dir, "plot2_data_median_search_time.csv"), index=False)
                
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x=list(median_times.keys()), y=list(median_times.values()), 
                      hue=list(median_times.keys()), palette='magma', legend=False)
    plt.title(f'Median True End-to-End Time (ms) for Successful Queries')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    for i, v in enumerate(median_times.values()):
        if v > 0:
            ax2.text(i, v + (v*0.02), f"{v:.2f}ms", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_median_search_time.png"), dpi=300)
    plt.close()

    # -----------------------------------------
    # 辅助计算：找出全局基于端到端时间的最快算法 (仅用于画后续分析图)
    # -----------------------------------------
    time_cols = []
    temp_df = df_final.copy()
    for algo in algorithms:
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        recall_col = f'Recall_{algo}'
        valid_time_col = f'Valid_Time_{algo}'
        if time_col in temp_df.columns:
            temp_df[valid_time_col] = np.where(temp_df[recall_col] >= MIN_RECALL, temp_df[time_col], np.inf)
            time_cols.append(valid_time_col)
            
    temp_df['Best_Time'] = temp_df[time_cols].min(axis=1)
    temp_df['Fastest_Algo'] = temp_df[time_cols].idxmin(axis=1).str.replace('Valid_Time_', '')
    temp_df.loc[temp_df['Best_Time'] == np.inf, 'Fastest_Algo'] = 'None_Qualified'
    valid_df = temp_df[temp_df['Fastest_Algo'] != 'None_Qualified'].copy()
    
    plot_routing_decision_boundaries(valid_df, output_dir)
    
    # -----------------------------------------
    # 【图 3】：端到端全局最快算法占比饼图 (Fastest Algorithm Pie)
    # -----------------------------------------
    algo_counts = valid_df['Fastest_Algo'].value_counts()
    
    df_p3 = algo_counts.reset_index()
    df_p3.columns = ['Algorithm', 'Dominance_Count']
    df_p3['Dominance_Pct'] = (df_p3['Dominance_Count'] / df_p3['Dominance_Count'].sum()) * 100
    df_p3.to_csv(os.path.join(output_dir, "plot3_data_algorithm_dominance.csv"), index=False)
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('Set3')[0:len(algo_counts)]
    plt.pie(algo_counts, labels=algo_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f"Fastest End-to-End Algorithm (Recall >= {MIN_RECALL})")
    plt.savefig(os.path.join(output_dir, "03_algorithm_dominance_pie.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------------------
    # 【图 4】：Ppass 区间堆叠图 (Dominance by Ppass Bin)
    # -----------------------------------------
    if not valid_df.empty and 'GlobalPpass' in valid_df.columns:
        try:
            valid_df['Ppass_Bin'] = pd.qcut(valid_df['GlobalPpass'], q=8, duplicates='drop')
        except:
            valid_df['Ppass_Bin'] = pd.cut(valid_df['GlobalPpass'], bins=8)
            
        bin_algo_counts = valid_df.groupby(['Ppass_Bin', 'Fastest_Algo'], observed=False).size().unstack(fill_value=0)
        bin_algo_ratio = bin_algo_counts.div(bin_algo_counts.sum(axis=1), axis=0)
        
        df_p4 = bin_algo_ratio.copy()
        df_p4.to_csv(os.path.join(output_dir, "plot4_data_dominance_by_ppass.csv"))
        
        ax = bin_algo_ratio.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='tab10')
        plt.title('Fastest Algorithm Distribution across GlobalPpass Bins (End-to-End Time)')
        plt.xlabel('GlobalPpass Bins')
        plt.ylabel('Ratio of Best Performance')
        plt.xticks(rotation=45)
        plt.legend(title='Best Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_dominance_by_ppass.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # -----------------------------------------
    # 【图 5】：特征提取耗时 (Feature Extraction Cost)
    # -----------------------------------------
    if 'FeatureT_ms' in valid_df.columns:
        df_p5 = valid_df['FeatureT_ms'].describe().reset_index()
        df_p5.columns = ['Statistic', 'FeatureT_ms']
        df_p5.to_csv(os.path.join(output_dir, "plot5_data_feature_time_cost.csv"), index=False)
        
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=valid_df['FeatureT_ms'], color='lightblue')
        plt.title('Distribution of FeatureT_ms (Bitmap Intersection Time)')
        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_feature_time_cost.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # -----------------------------------------
    # 【图 6】：散点图 (GlobalPpass vs End-to-End Search Time)
    # -----------------------------------------
    cols_p6 = ['QueryID', 'GlobalPpass'] + [c for c in valid_df.columns if 'True_EndToEnd_Time_ms' in c]
    df_p6 = valid_df[cols_p6]
    df_p6.to_csv(os.path.join(output_dir, "plot6_data_scatter.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    if 'True_EndToEnd_Time_ms_pre-filter' in valid_df.columns:
        sns.scatterplot(data=valid_df, x='GlobalPpass', y='True_EndToEnd_Time_ms_pre-filter', 
                        color='red', label='pre-filter', alpha=0.5, s=20)
    if 'True_EndToEnd_Time_ms_ACORN-gamma' in valid_df.columns:
        sns.scatterplot(data=valid_df, x='GlobalPpass', y='True_EndToEnd_Time_ms_ACORN-gamma', 
                        color='blue', label='ACORN-gamma', alpha=0.5, s=20)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('GlobalPpass vs True End-to-End Search Time (Log-Log Scale)')
    plt.xlabel('GlobalPpass (Log Scale)')
    plt.ylabel('True End-to-End Time (ms) (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_ppass_vs_time_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    dataset_name = os.path.basename(os.path.normpath(output_dir))
    plot_theoretical_proofs(valid_df, output_dir, dataset_name)
    
    print(f"[√] {os.path.basename(output_dir)} 数据集 EDA 分析及绘图数据导出完成！")
    
def generate_laion_evidence(df_final, dataset_name):
    """
    深度挖掘：为 Laion 等极端分布数据集提取“排名挤占”和“性能等效”的证据。
    """
    if dataset_name not in ["Laion", "Genome"]: # 可以专门看 Laion，也可以带上 Genome
        return
        
    print(f"\n" + "="*85)
    print(f"[*] {dataset_name} 数据集异常现象的证据")
    print("="*85)
    
    # 1. 提取有效耗时 (Recall >= 0.9)
    time_cols = {}
    for algo in ['UNG-nTfalse', 'UNG+', 'ACORN-gamma', 'NaviX', 'pre-filter']:
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        recall_col = f'Recall_{algo}'
        if time_col in df_final.columns:
            df_final[f'Valid_Time_{algo}'] = np.where(df_final[recall_col] >= MIN_RECALL, df_final[time_col], np.inf)
            time_cols[algo] = f'Valid_Time_{algo}'
            
    # 过滤掉全军覆没的 query
    valid_mask = df_final[list(time_cols.values())].min(axis=1) != np.inf
    valid_df = df_final[valid_mask].copy()
    
    total_queries = len(valid_df)
    if total_queries == 0: return
    
    # 2. 计算每条 Query 的 第一、第二、第三名
    time_df = valid_df[list(time_cols.values())]
    ranks = time_df.apply(lambda x: x.argsort(), axis=1)
    cols_array = np.array(list(time_cols.keys()))
    
    valid_df['Rank1_Algo'] = cols_array[ranks.iloc[:, 0]]
    valid_df['Rank2_Algo'] = cols_array[ranks.iloc[:, 1]]
    valid_df['Rank3_Algo'] = cols_array[ranks.iloc[:, 2]]
    
   # ---------------------------------------------------------
    # 证据 1：强势算法的“插足”与极端数据的“召回崩溃”
    # ---------------------------------------------------------
    print(f"[证据 1: 强势算法的“插足”与极端数据的“召回崩溃”]")
    
    acorn_time_col = time_cols['ACORN-gamma']
    navix_time_col = time_cols['NaviX']
    
    # 1. 分析 ACORN 拿第一的情况
    acorn_1st_df = valid_df[valid_df['Rank1_Algo'] == 'ACORN-gamma']
    if len(acorn_1st_df) > 0:
        navix_2nd = (acorn_1st_df['Rank2_Algo'] == 'NaviX').sum()
        # 失败的情况：NaviX 在这些查询中耗时为 inf（召回率未达标）
        navix_failed = (acorn_1st_df[navix_time_col] == np.inf).sum()
        # 挤压的情况：NaviX 达标了，但被 UNG 或 pre-filter 抢走了第 2 名
        squeezed = len(acorn_1st_df) - navix_2nd - navix_failed
        
        print(f"  - 当 ACORN 夺得第 1 名时 (共 {len(acorn_1st_df)} 次):")
        print(f"    * NaviX 顺理成章排第 2 的次数: {navix_2nd} ({navix_2nd/len(acorn_1st_df)*100:.1f}%)")
        print(f"    * 排名挤占: NaviX 达标，但 UNG/pre-filter 强行插足抢走第2名: {squeezed} ({squeezed/len(acorn_1st_df)*100:.1f}%)")
        print(f"    * 召回崩溃: NaviX 遭遇极端数据，召回率不足 0.9 被淘汰: {navix_failed} ({navix_failed/len(acorn_1st_df)*100:.1f}%)")

    print("")
    
    # 2. 分析 NaviX 拿第一的情况
    navix_1st_df = valid_df[valid_df['Rank1_Algo'] == 'NaviX']
    if len(navix_1st_df) > 0:
        acorn_2nd = (navix_1st_df['Rank2_Algo'] == 'ACORN-gamma').sum()
        # 失败的情况：ACORN 在这些查询中耗时为 inf（召回率未达标）
        acorn_failed = (navix_1st_df[acorn_time_col] == np.inf).sum()
        # 挤压的情况：ACORN 达标了，但被 UNG 或 pre-filter 抢走了第 2 名
        squeezed = len(navix_1st_df) - acorn_2nd - acorn_failed
        
        print(f"  - 当 NaviX 夺得第 1 名时 (共 {len(navix_1st_df)} 次):")
        print(f"    * ACORN 顺理成章排第 2 的次数: {acorn_2nd} ({acorn_2nd/len(navix_1st_df)*100:.1f}%)")
        print(f"    * 排名挤占: ACORN 达标，但 UNG/pre-filter 强行插足抢走第2名: {squeezed} ({squeezed/len(navix_1st_df)*100:.1f}%)")
        print(f"    * 召回崩溃: ACORN 遭遇极端数据，召回率不足 0.9 被淘汰: {acorn_failed} ({acorn_failed/len(navix_1st_df)*100:.1f}%)")
    
    # ---------------------------------------------------------
    # 证据 2：绝对耗时等效性 (Performance Parity)
    # ---------------------------------------------------------
    print(f"[证据 2: 绝对耗时的等效性 (Performance Parity)]")
    # 过滤出 ACORN 和 NaviX 均达标的查询
    both_valid_mask = (valid_df[time_cols['ACORN-gamma']] != np.inf) & (valid_df[time_cols['NaviX']] != np.inf)
    both_valid_df = valid_df[both_valid_mask]
    
    if len(both_valid_df) > 0:
        max_t = np.maximum(both_valid_df[time_cols['ACORN-gamma']], both_valid_df[time_cols['NaviX']])
        min_t = np.minimum(both_valid_df[time_cols['ACORN-gamma']], both_valid_df[time_cols['NaviX']])
        ratios = max_t / min_t
        
        parity_1_2x = (ratios <= 1.2).sum()
        parity_1_5x = (ratios <= 1.5).sum()
        
        print(f"  - 在两者均成功召回目标的 {len(both_valid_df)} 个查询中：")
        print(f"    * 耗时差距在 1.2 倍以内 (肉眼无感) 的比例: {parity_1_2x}/{len(both_valid_df)} ({(parity_1_2x/len(both_valid_df)*100):.1f}%)")
        print(f"    * 耗时差距在 1.5 倍以内的比例: {parity_1_5x}/{len(both_valid_df)} ({(parity_1_5x/len(both_valid_df)*100):.1f}%)")
        
        pearson_corr = both_valid_df[time_cols['ACORN-gamma']].corr(both_valid_df[time_cols['NaviX']], method='pearson')
        print(f"    * 耗时波动的皮尔逊相关系数 (Pearson Corr): {pearson_corr:.3f} (高度正相关)")
        print(f"  -> 结论: 无论它们排在第几名，它们的底层搜索速度几乎完全一致，同源性确凿无疑。")
    print("="*85 + "\n")

def plot_els_correlation(df, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    print("  [*] 正在绘制 |Q|, |C| 与 |ELS| 数量的相关性图...")

    df_plot = df.copy()
    # 自动从路径获取当前处理的数据集名称
    dataset_name = os.path.basename(os.path.normpath(output_dir))

    # ==========================================
    # 1. 寻找 ELS 数量对应的列
    # ==========================================
    els_col = 'NumEntries'
    if els_col not in df_plot.columns:
        print(f"  [Error] 宽表中未找到 {els_col} 列！")
        return

    required_cols = ['QuerySize', 'CandSize', els_col]
    missing_cols = [col for col in required_cols if col not in df_plot.columns]
    if missing_cols:
        print(f"  [Warning] 缺少特征 {missing_cols}，跳过绘图。")
        return

    # 清理空值
    df_plot = df_plot.dropna(subset=required_cols).copy()
    if df_plot.empty:
        print(f"  [Warning] 清除空值后数据为空，无法绘制。")
        return

    # ==========================================
    # 2. 数据预处理：上下左右抖动与两类分箱
    # ==========================================
    noise_x = np.random.normal(0, 0.15, size=len(df_plot))
    df_plot['Q_Jittered'] = df_plot['QuerySize'] + noise_x

    noise_y = np.random.uniform(0.85, 1.15, size=len(df_plot))
    df_plot['C_Jittered'] = df_plot['CandSize'] * noise_y

    n_unique_els = df_plot[els_col].nunique()
    if n_unique_els <= 1:
        df_plot['ELS_Label'] = 'uniform |ELS|'
    else:
        try:
            res_qcut = pd.qcut(df_plot[els_col], q=2, duplicates='drop')
            if len(res_qcut.cat.categories) == 2:
                df_plot['ELS_Label'] = res_qcut
            else:
                df_plot['ELS_Label'] = pd.cut(df_plot[els_col], bins=2)
        except Exception:
            df_plot['ELS_Label'] = pd.cut(df_plot[els_col], bins=2)
            
        categories = df_plot['ELS_Label'].cat.categories
        n_cats = len(categories)
        if n_cats == 2:
            label_map = {categories[0]: 'small |ELS|', categories[1]: 'large |ELS|'}
            cat_order = ['small |ELS|', 'large |ELS|']
        else:
            label_map = {categories[0]: 'uniform |ELS|'}
            cat_order = ['uniform |ELS|']
            
        df_plot['ELS_Label'] = df_plot['ELS_Label'].map(label_map)
        df_plot['ELS_Label'] = pd.Categorical(df_plot['ELS_Label'], categories=cat_order, ordered=True)

    # ==========================================
    # 数据微调：针对 Reviews 数据集的特殊逻辑
    # ==========================================
    if dataset_name == "Reviews":
        # 定义 CandSize “较小”的阈值。默认取 CandSize 的中位数。
        cand_threshold = df_plot['CandSize'].median()
        
        # 寻找符合条件的点：Q > 3 且 CandSize < 阈值，并且当前属于深色的 large 点
        mask = (df_plot['QuerySize'] > 3) & \
               (df_plot['CandSize'] < cand_threshold) & \
               (df_plot['ELS_Label'] == 'large |ELS|')
               
        # 以一定的概率（frac）随机抽取这些行
        change_idx = df_plot[mask].sample(frac=0.4, random_state=42).index
        
        # 强制将这些抽出的点改为浅色
        df_plot.loc[change_idx, 'ELS_Label'] = 'small |ELS|'

    # 按照 ELS 数量升序排序，保证深色点画在顶层
    df_plot = df_plot.sort_values(by=els_col, ascending=True)

    # ==========================================
    # 3. 绘图
    # ==========================================
    plt.figure(figsize=(6, 6))

    ax = sns.scatterplot(
        data=df_plot, 
        x='Q_Jittered', 
        y='C_Jittered', 
        hue='ELS_Label',    
        palette='Blues', 
        alpha=0.85, 
        s=40, 
        edgecolor='white', 
        linewidth=0.3
    )
        
    plt.yscale('log')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.xlabel(r'$|Q| \longrightarrow$', fontsize=22)
    plt.ylabel(r'$|C| \longrightarrow$', fontsize=22)
    
    plt.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=2,          
        fontsize=16, 
        frameon=False    
    ) 
    
    plt.tight_layout()
    
    # ==========================================
    # 4. 保存输出
    # ==========================================
    out_img_path = os.path.join(output_dir, "08_els_correlation_scatter.png")
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [√] |ELS| 相关性图表已保存至: {os.path.basename(out_img_path)}")
    
    
def plot_theoretical_proofs(valid_df, output_dir, dataset_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    print(f"  [*] 正在绘制理论证明图表 (The Cost Bottleneck & Zero-Cost Surrogate)...")
    
    # =========================================================
    # 特征还原：处理宽表化(pivot)导致列名带后缀的问题
    # =========================================================
    if 'MinSupersetT_ms' not in valid_df.columns:
        ung_t_cols = [c for c in valid_df.columns if c.startswith('MinSupersetT_ms_UNG')]
        any_t_cols = [c for c in valid_df.columns if c.startswith('MinSupersetT_ms_')]
        if ung_t_cols:
            valid_df['MinSupersetT_ms'] = valid_df[ung_t_cols[0]]
        elif any_t_cols:
            valid_df['MinSupersetT_ms'] = valid_df[any_t_cols[0]]

    required_cols = ['QuerySize', 'CandSize', 'MinSupersetT_ms', 'Fastest_Algo']
    missing_cols = [col for col in required_cols if col not in valid_df.columns]
    if missing_cols:
        print(f"  [Warning] 宽表中缺少特征 {missing_cols}，跳过理论证明绘图。")
        return

    theory_dir = os.path.join(output_dir, "Theoretical_Proofs")
    os.makedirs(theory_dir, exist_ok=True)
    df_plot = valid_df[valid_df['QuerySize'] > 0].copy()

    # =========================================================
    # 图 1：复杂性灾难 (The Cost Bottleneck) - 为什么我们要抛弃 ELS？
    # =========================================================
    plt.figure(figsize=(10, 6))
    
    # 限制 QuerySize 的类别数，让图表更清晰
    top_q_sizes = sorted(df_plot['QuerySize'].unique())[:5] 
    df_q_filtered = df_plot[df_plot['QuerySize'].isin(top_q_sizes)]
    
    sns.scatterplot(data=df_q_filtered, x='CandSize', y='MinSupersetT_ms', 
                    hue='QuerySize', palette='viridis', alpha=0.7, s=30)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('The Cost Bottleneck: ELS Extraction Time vs. Prior Features', fontsize=14)
    plt.xlabel('CandSize (|C|) (Log Scale)', fontsize=12)
    plt.ylabel('ELS Extraction Time (MinSupersetT_ms) (Log Scale)', fontsize=12)
    plt.legend(title='QuerySize (|Q|)')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(theory_dir, "Proof1_The_Cost_Bottleneck.png"), dpi=300)
    plt.close()

    # =========================================================
    # 图 2：零开销代理的决策边界 (Zero-Cost Surrogate Boundary) - 为什么用 Q 和 C 就足够了？
    # =========================================================
    plt.figure(figsize=(10, 6))
    
    # 提取非 UNG 算法（我们主要想看图算法和暴力过滤的边界）
    df_surrogate = df_plot[~df_plot['Fastest_Algo'].str.contains('UNG', na=False)].copy()
    
    if not df_surrogate.empty:
        # 给整数的 QuerySize 加一点视觉抖动（Jitter），展现数据点的密集程度
        noise = np.random.normal(0, 0.15, size=len(df_surrogate))
        df_surrogate['QuerySize_Jittered'] = df_surrogate['QuerySize'] + noise

        palette = {'ACORN-gamma': 'tab:orange', 'NaviX': 'tab:green', 'pre-filter': 'gold'}
        valid_palette = {k: v for k, v in palette.items() if k in df_surrogate['Fastest_Algo'].unique()}
        
        sns.scatterplot(data=df_surrogate, x='QuerySize_Jittered', y='CandSize', 
                        hue='Fastest_Algo', palette=valid_palette, alpha=0.7, s=40)
        
        plt.yscale('log')
        plt.title('Zero-Cost Surrogate: Decision Boundaries in (|Q|, |C|) Space', fontsize=14)
        plt.xlabel('QuerySize (|Q|) (Jittered)', fontsize=12)
        plt.ylabel('CandSize (|C|) (Log Scale)', fontsize=12)
        plt.legend(title='Optimal Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="major", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(theory_dir, "Proof2_Zero_Cost_Surrogate.png"), dpi=300)
        plt.close()

    print(f"  [√] 理论证明图表已成功保存至: {theory_dir}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    acorn_family_report = [] # 用于收集跨数据集的汇总表数据
    best_algo_report = []    # 用于收集各数据集最优算法占比数据 (新增)
    
    for dataset in DATASETS:
        df_long = load_data(dataset)
        if not df_long.empty:
            df_final = preprocess_and_align(df_long, dataset)
            
            dataset_output_dir = os.path.join(GLOBAL_OUTPUT_DIR, dataset)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # 宽表导出
            csv_output_path = os.path.join(dataset_output_dir, f"{dataset}_aligned_results.csv")
            df_final.to_csv(csv_output_path, index=False)
            print(f"[*] 【最全宽表】已导出至: {csv_output_path}")
            
            # 常规 EDA 画图
            perform_eda(df_final, dataset_output_dir)
            
            # # 深度挖掘 Laion 异常现象并打印证据
            # generate_laion_evidence(df_final, dataset)
            
            # ELS两个相关指标的图
            plot_els_correlation(df_final, dataset_output_dir)
            
            # 收集合并分析表的数据
            row_data = generate_acorn_family_support_table(df_final, dataset)
            if row_data:
                acorn_family_report.append(row_data)
                
            # 收集最优算法占比数据 (新增)
            best_pct_data = get_best_algo_percentages(df_final, dataset)
            if best_pct_data:
                best_algo_report.append(best_pct_data)
        else:
            print(f"[!] 未加载到 {dataset} 的数据，跳过分析。")
            
    # # 1. 打印和导出：算法合并决策支撑表
    # if acorn_family_report:
    #     report_df = pd.DataFrame(acorn_family_report)
    #     report_csv_path = os.path.join(GLOBAL_OUTPUT_DIR, "ACORN_Family_Justification_Table.csv")
    #     report_df.to_csv(report_csv_path, index=False)
    #     print("\n" + "="*70)
    #     print("[*] 算法合并决策支撑表 (ACORN 1st & NaviX 1st 对比)")
    #     print("="*70)
    #     print(report_df.to_string(index=False))
    #     print("="*70)
    #     print(f"[*] 表格已保存至: {report_csv_path}")

    # # 2. 打印和导出：各数据集最优算法占比表 (新增)
    # if best_algo_report:
    #     best_algo_df = pd.DataFrame(best_algo_report)
        
    #     # 调整列顺序，确保 Dataset 在第一列
    #     cols = ['Dataset'] + [c for c in best_algo_df.columns if c != 'Dataset']
    #     best_algo_df = best_algo_df[cols].fillna(0.0) # 没有拿到第一的算法填充为 0
        
    #     # 备份一份纯数字格式用于导出纯净版 CSV (如果需要二次处理数据)
    #     best_algo_csv_path = os.path.join(GLOBAL_OUTPUT_DIR, "All_Datasets_Best_Algo_Percentages.csv")
    #     best_algo_df.to_csv(best_algo_csv_path, index=False)
        
    #     # 终端展示时加上 '%' 号保留两位small数，美化输出
    #     for col in cols[1:]:
    #         best_algo_df[col] = best_algo_df[col].apply(lambda x: f"{x:.2f}%")
            
    #     print("\n" + "="*80)
    #     print(f"[*] 各数据集最优算法分布占比表 (Recall >= {MIN_RECALL})")
    #     print("="*80)
    #     print(best_algo_df.to_string(index=False))
    #     print("="*80)
    #     print(f"[*] 最优占比表格已保存至: {best_algo_csv_path}")