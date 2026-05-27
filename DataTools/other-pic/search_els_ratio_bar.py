import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch, Patch


# -----------------------------
# Publication-oriented settings
# -----------------------------
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
ANNOT_FONT_SIZE = 9
MAX_ROWS_TO_READ = 20000
FILTER_RATIO_THRESHOLD = 0.10
SPECIAL_FILTER_DATASETS = {"Genome", "VariousImg"}
SORT_BY_FILTER_RATIO = False
ANNOTATION_THRESHOLD = 0.07
EXPORT_VECTOR_PDF = False

# New: individually rounded segment controls
BAR_WIDTH = 0.68
SEGMENT_ROUNDING = 0.035  # recommended range: 0.025 ~ 0.045
SEGMENT_GAP = 0.004       # small vertical gap between stacked components
BAR_ALPHA = 0.96
LAST_BAR_RIGHT_SHIFT = 0.18

DATASETS_TO_LOAD = {
    "Genome": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Genome/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Reviews": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Reviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Tiktok": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Tiktok/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "VariousImg": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/VariousImg/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Music": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Amazon": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Amazon/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Laion": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Laion/Results/UNG-nTfalse/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_200_C_D-weighted-sub-base-123456789_random_300_K10]_Search[Ls1000-Le40000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
    "BookReviews": "/noraiddata/lijiakang/FilterVector/FilterVectorResults/BookReviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_200_B_C_D-weighted-sub-base-123456789_random_300_K10]_Search[Ls500-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
}

DATASET_ORDER = [
    "Genome",
    "Reviews",
    "Tiktok",
    "VariousImg",
    "Music",
    "Amazon",
    "Laion",
    "BookReviews",
]

COMPONENT_COLUMNS = [
    "search_time_ms",
    "ELS_FilterT_ms",
    "ELS_TrieT_ms",
]

DISPLAY_LABELS = {
    "search_time_ms": "Unified-graph Search",
    "ELS_TrieT_ms": "ELS Retrieval (Phase 1)",
    "ELS_FilterT_ms": "ELS Retrieval (Phase 2)",
}

LEGEND_ORDER = [
    "ELS_TrieT_ms",
    "ELS_FilterT_ms",
    "search_time_ms",
]

# Softer, more muted colors
# COMPONENT_COLORS = {
#     "search_time_ms": "#D0F488",#DBEEF3
#     "ELS_TrieT_ms": "#D4B0F8",
#     "ELS_FilterT_ms": "#94CDDC",
# }
COMPONENT_COLORS = {
    "search_time_ms": "#A6D5E2",
    "ELS_TrieT_ms": "#D7CEE0",
    "ELS_FilterT_ms": "#D7E3BF",
}


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": TICK_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.linewidth": 0.8,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_dataset_means(dataset_name, csv_path):
    print(f"正在处理数据集: {dataset_name} ...")
    try:
        df = pd.read_csv(csv_path, usecols=COMPONENT_COLUMNS, nrows=MAX_ROWS_TO_READ)
    except FileNotFoundError:
        print(f"错误：文件未找到 -> {csv_path}")
        return None
    except ValueError as exc:
        print(f"错误：文件 {dataset_name} 缺少必要列: {exc}")
        return None

    if df.empty:
        print("    警告：数据为空，跳过。")
        return None

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=COMPONENT_COLUMNS, how="all")
    if df.empty:
        print("    警告：清洗后无有效数据，跳过。")
        return None

    if dataset_name in SPECIAL_FILTER_DATASETS:
        component_total = df[COMPONENT_COLUMNS].sum(axis=1)
        valid_total_mask = component_total > 0
        df = df[valid_total_mask].copy()
        component_total = component_total[valid_total_mask]
        if df.empty:
            print("    警告：分母无有效值，跳过。")
            return None

        filter_ratio = df["ELS_FilterT_ms"] / component_total
        before_count = len(df)
        df = df[filter_ratio > FILTER_RATIO_THRESHOLD].copy()
        print(
            f"    特殊筛选: 仅保留 ELS_FilterT_ms 占比 > {FILTER_RATIO_THRESHOLD:.0%} 的 query，"
            f"保留 {len(df)} / {before_count} 行。"
        )
        if df.empty:
            print("    警告：特殊筛选后无有效数据，跳过。")
            return None

    means = df[COMPONENT_COLUMNS].mean()
    total = means.sum()
    if total <= 0:
        print("    警告：三部分平均值总和 <= 0，跳过。")
        return None

    proportions = means / total
    return {
        "Dataset": dataset_name,
        **{col: means[col] for col in COMPONENT_COLUMNS},
        **{f"{col}_ratio": proportions[col] for col in COMPONENT_COLUMNS},
        "total_mean_time_ms": total,
    }


def maybe_sort_result_df(result_df):
    if SORT_BY_FILTER_RATIO:
        return result_df.sort_values("ELS_FilterT_ms_ratio", ascending=False, kind="mergesort").reset_index(drop=True)
    return result_df.copy().reset_index(drop=True)


def add_centered_percentage_label(ax, x_center, y_center, value):
    text = ax.text(
        x_center,
        y_center,
        f"{value:.0%}",
        ha="center",
        va="center",
        fontsize=ANNOT_FONT_SIZE + 7,
        color="#1A1A1A",
        zorder=6,
    )
    text.set_path_effects([
        path_effects.Stroke(linewidth=1.1, foreground=(1, 1, 1, 0.75)),
        path_effects.Normal(),
    ])


def draw_rounded_segment(ax, x_center, y_bottom, height, width, color):
    """
    Draw one stack component as an individually rounded rectangle.
    A tiny vertical gap is applied between neighboring components to make
    the individual rounded corners visible.
    """
    if height <= 0:
        return None

    visible_height = max(height - SEGMENT_GAP, 0.001)
    y = y_bottom + SEGMENT_GAP / 2.0
    rounding = min(SEGMENT_ROUNDING, visible_height / 2.0, width / 2.0)

    patch = FancyBboxPatch(
        (x_center - width / 2.0, y),
        width,
        visible_height,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=0.85,
        edgecolor=(1, 1, 1, 0.94),
        facecolor=color,
        alpha=BAR_ALPHA,
        zorder=3,
    )
    ax.add_patch(patch)
    return patch


def plot_stacked_ratio_bar(result_df, output_path):
    configure_matplotlib()
    plot_df = maybe_sort_result_df(result_df)

    plot_tick_font_size = TICK_FONT_SIZE + 5.7
    plot_label_font_size = LABEL_FONT_SIZE + 5
    plot_legend_font_size = LEGEND_FONT_SIZE + 5

    fig, ax = plt.subplots(figsize=(9.6, 3.9), constrained_layout=True)

    x = np.arange(len(plot_df), dtype=float)
    if len(x) > 0:
        x[-1] += LAST_BAR_RIGHT_SHIFT
    bottom = np.zeros(len(plot_df))

    for col in COMPONENT_COLUMNS:
        ratios = plot_df[f"{col}_ratio"].to_numpy()

        for idx, value in enumerate(ratios):
            draw_rounded_segment(
                ax=ax,
                x_center=x[idx],
                y_bottom=bottom[idx],
                height=value,
                width=BAR_WIDTH,
                color=COMPONENT_COLORS[col],
            )

            if value >= ANNOTATION_THRESHOLD:
                add_centered_percentage_label(
                    ax=ax,
                    x_center=x[idx],
                    y_center=bottom[idx] + value / 2.0,
                    value=value,
                )

        bottom += ratios

    ax.set_xlim(-0.65, x[-1] + 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["Dataset"],
        rotation=0,
        ha="center",
        fontsize=plot_tick_font_size,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Time proportion", fontsize=plot_label_font_size)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.6, color="#DDDDDD", alpha=0.85)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")
    ax.tick_params(axis="both", length=0, pad=3, labelsize=plot_tick_font_size)

    legend_handles = [
        Patch(
            facecolor=COMPONENT_COLORS[col],
            edgecolor="none",
            alpha=BAR_ALPHA,
            label=DISPLAY_LABELS[col],
        )
        for col in LEGEND_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.45,
        columnspacing=0.9,
        fontsize=plot_legend_font_size,
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    if EXPORT_VECTOR_PDF:
        root, _ = os.path.splitext(output_path)
        fig.savefig(f"{root}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"绘图完成，图像已保存为 {output_path}")


def main():
    records = []
    for dataset_name in DATASET_ORDER:
        csv_path = DATASETS_TO_LOAD.get(dataset_name)
        if not csv_path:
            continue
        records.append(load_dataset_means(dataset_name, csv_path))

    valid_records = [record for record in records if record is not None]
    if not valid_records:
        print("错误：没有加载到任何有效数据，无法绘图。")
        return

    result_df = pd.DataFrame(valid_records)
    result_df.to_csv("search_els_ratio_bar_summary.csv", index=False)
    plot_stacked_ratio_bar(result_df, "search_els_ratio_bar.png")


if __name__ == "__main__":
    main()
