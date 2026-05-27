import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

# --- 1. 全局控制参数 ---
FILTER_RANGE = (0.0, 1.0)
Y_AXIS_RANGE = (0.0, 1.0)
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 22
MAX_ROWS_TO_READ = 1000  # 设为 None 表示读取全部行

# --- 8 种颜色配色方案 (统一深青色) ---
USER_PALETTE = [
    "#2b7488", "#2b7488", "#2b7488", "#2b7488",
    "#2b7488", "#2b7488", "#2b7488", "#2b7488",
]

# --- 2. 数据加载与处理函数 (单文件模式) ---
def process_single_dataset(dataset_name, path_data_file):
    """
    直接加载一个 CSV 文件，计算 ELS_FilterT_ms / MinSupersetT_ms 的比例。
    不再进行额外的 QueryID 筛选比对。
    """
    print(f"正在处理数据集: {dataset_name} ...")
    try:
        cols_to_use = ["QueryID", "MinSupersetT_ms", "ELS_FilterT_ms"]
        df_raw = pd.read_csv(
            path_data_file,
            usecols=cols_to_use,
            nrows=MAX_ROWS_TO_READ,
        )

        # 防止同一个 QueryID 存在多行，先按 Query 聚合取平均。
        df_data = df_raw.groupby("QueryID").mean()
        rows_hint = "全部行" if MAX_ROWS_TO_READ is None else f"前 {MAX_ROWS_TO_READ} 行"
        print(f"    成功加载数据（{rows_hint}），共 {len(df_data)} 个查询点。")

        # 仅保留分母有效的 query，避免除零。
        df_data = df_data[df_data["MinSupersetT_ms"] > 0]
        df_data["Proportion"] = df_data["ELS_FilterT_ms"] / df_data["MinSupersetT_ms"]

        df_processed = df_data[["Proportion"]].copy()
        df_processed["Dataset"] = dataset_name

        df_processed = df_processed.replace([np.inf, -np.inf], np.nan).dropna(subset=["Proportion"])

        min_val, max_val = FILTER_RANGE
        df_to_plot = df_processed[
            (df_processed["Proportion"] >= min_val) & (df_processed["Proportion"] <= max_val)
        ]

        removed_count = len(df_processed) - len(df_to_plot)
        print(
            f"    范围过滤 [{min_val}, {max_val}]: 保留 {len(df_to_plot)} 个点 "
            f"(剔除 {removed_count} 个异常点)"
        )

        return df_to_plot

    except FileNotFoundError:
        print(f"错误：文件未找到 -> {path_data_file}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"错误：文件 {dataset_name} 中缺少必要的列: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误：处理 {dataset_name} 时发生未知异常: {e}")
        return pd.DataFrame()


# --- 3. 定义数据集路径 ---
datasets_to_load = {
    "Genome": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Genome/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Reviews": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Reviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Amazon": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Amazon/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "VariousImg": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/VariousImg/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Music": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "BookReviews": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/BookReviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_200_B_C_D-weighted-sub-base-123456789_random_300_K10]_Search[Ls500-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Tiktok": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Tiktok/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Laion": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Laion/Results/UNG-nTfalse/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_200_C_D-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
}

all_data_frames = []

dataset_order = [
    "Genome", "Reviews", "Amazon", "VariousImg",
    "Music", "BookReviews", "Tiktok", "Laion",
]

for dataset_name in dataset_order:
    if dataset_name in datasets_to_load:
        clean_path = datasets_to_load[dataset_name].replace("\u00a0", "")
        df_processed = process_single_dataset(dataset_name, clean_path)
        if not df_processed.empty:
            all_data_frames.append(df_processed)

if not all_data_frames:
    print("\n错误：没有加载到任何有效数据，无法绘图。请检查路径。")
else:
    final_df = pd.concat(all_data_frames)
    print("\n所有数据加载完成，开始绘图...")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 3))

    x_axis_order = [name for name in dataset_order if name in final_df["Dataset"].unique()]
    color_map = {name: USER_PALETTE[i % len(USER_PALETTE)] for i, name in enumerate(dataset_order)}

    ax = sns.stripplot(
        data=final_df,
        x="Dataset",
        y="Proportion",
        order=x_axis_order,
        hue="Dataset",
        hue_order=x_axis_order,
        jitter=0.2,
        palette=color_map,
        s=3.5,
        alpha=1.0,
        legend=False,
        zorder=2,
    )

    sns.pointplot(
        data=final_df,
        x="Dataset",
        y="Proportion",
        order=x_axis_order,
        linestyle="none",
        markers="_",
        markersize=15,
        linewidth=2.5,
        color="black",
        errorbar=None,
        ax=ax,
        legend=False,
        zorder=10,
    )

    present_datasets = set(final_df["Dataset"])
    box_width = 0.5

    for i, dataset_name in enumerate(x_axis_order):
        if dataset_name not in present_datasets:
            continue

        dataset_data = final_df[final_df["Dataset"] == dataset_name]
        if dataset_data.empty:
            continue

        y_bottom = dataset_data["Proportion"].min()
        y_top = dataset_data["Proportion"].max()
        rect = patches.Rectangle(
            (i - (box_width / 2), y_bottom),
            box_width,
            y_top - y_bottom,
            linewidth=2.5,
            edgecolor=color_map.get(dataset_name, "#2b7488"),
            facecolor="none",
            zorder=1,
        )
        ax.add_patch(rect)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(Y_AXIS_RANGE[0], Y_AXIS_RANGE[1])
    ax.set_xlabel(None)
    ax.set_ylabel(
        "ELS Filter in MinSuperset",
        fontsize=LABEL_FONT_SIZE,
        color="black",
        labelpad=10,
    )

    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", colors="black", labelsize=TICK_FONT_SIZE)

    # 给左侧 ylabel 留出更稳定的边距，避免保存时被裁切。
    plt.tight_layout()
    plt.subplots_adjust(left=0.20)
    save_name = "ELS filter ratio.png"
    plt.savefig(save_name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"绘图完成，图像已保存为 {save_name}")
