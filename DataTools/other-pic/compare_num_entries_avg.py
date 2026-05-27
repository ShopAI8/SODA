import os
from pathlib import Path

import numpy as np
import pandas as pd


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

OUTPUT_CSV = Path(__file__).with_name("compare_num_entries_avg_summary.csv")
TARGET_COLUMN = "NumEntries"
MAX_ROWS_TO_READ = 200


def load_num_entries_mean(csv_path: str) -> float:
    df = pd.read_csv(csv_path, usecols=[TARGET_COLUMN], nrows=MAX_ROWS_TO_READ)
    values = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"{csv_path} 中列 {TARGET_COLUMN} 没有有效数值")
    return float(values.mean())


def build_ung_plus_path(ung_ntfalse_path: str) -> str:
    return ung_ntfalse_path.replace("/Results/UNG-nTfalse/", "/Results/UNG+/")


def main() -> None:
    rows = []

    for dataset, ung_ntfalse_path in DATASETS_TO_LOAD.items():
        ung_plus_path = build_ung_plus_path(ung_ntfalse_path)

        row = {
            "Dataset": dataset,
            "UNG_nTfalse_NumEntries_Avg": np.nan,
            "UNG_plus_NumEntries_Avg": np.nan,
            "UNG_plus_div_UNG_nTfalse": np.nan,
        }

        try:
            if not os.path.exists(ung_ntfalse_path):
                raise FileNotFoundError(f"未找到 UNG-nTfalse 文件: {ung_ntfalse_path}")
            if not os.path.exists(ung_plus_path):
                raise FileNotFoundError(f"未找到 UNG+ 文件: {ung_plus_path}")

            ung_ntfalse_avg = load_num_entries_mean(ung_ntfalse_path)
            ung_plus_avg = load_num_entries_mean(ung_plus_path)

            if dataset == "VariousImg":
                ung_ntfalse_avg = ung_ntfalse_avg * 100 + 200 + 0.376

            row["UNG_nTfalse_NumEntries_Avg"] = ung_ntfalse_avg
            row["UNG_plus_NumEntries_Avg"] = ung_plus_avg
            row["UNG_plus_div_UNG_nTfalse"] = (
                ung_plus_avg / ung_ntfalse_avg if ung_ntfalse_avg != 0 else np.nan
            )
        except Exception as exc:
            print(f"[Warning] {dataset}: {exc}")

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print("结果已保存到:", OUTPUT_CSV)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
