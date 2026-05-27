import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATASETS = [
    "Amazon",
    "BookReviews",
    "Genome",
    "Laion",
    "Music",
    "Reviews",
    "Tiktok",
    "VariousImg",
]

ALGORITHMS = {
    "SmartRoute+": "SmartRoute+",
    "SmartRoute": "SmartRoute",
}

TARGET_CONFIGS = ("K10_th100", "K20_th100", "K10_th10")
SPECIAL_CASES = ("len_large", "len_small", "ppass_small", "ppass_large")
DEFAULT_CASE = "self"
CONFIG_CASES = {
    "K10_th100": (DEFAULT_CASE, "len_large", "len_small", "ppass_small", "ppass_large"),
    "K20_th100": (DEFAULT_CASE,),
    "K10_th10": (DEFAULT_CASE,),
}
FILE_NAME = "search_time_summary.csv"
DEFAULT_ROOT = "/noraiddata/lijiakang/FilterVector/FilterVectorResults"
DEFAULT_OUTPUT = "smart_route_recall_ge_0.9_qps_ratio.csv"
DEFAULT_QUERY_COUNT = 60000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 SmartRoute+ 和 SmartRoute 在 Recall >= 0.9 时的最高 QPS 及其比值。"
        )
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"结果根目录，默认：{DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出 CSV 路径，默认：{DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--query-count",
        type=float,
        default=DEFAULT_QUERY_COUNT,
        help=(
            "如果 CSV 中没有 QPS 列，可用该参数从 Average_Time_ms 推导 QPS。"
            f"默认值为 {int(DEFAULT_QUERY_COUNT)}。"
        ),
    )
    return parser.parse_args()


def detect_case(path_str: str) -> Optional[str]:
    for case_name in SPECIAL_CASES:
        if case_name in path_str:
            return case_name
    return DEFAULT_CASE


def detect_config(path_str: str) -> Optional[str]:
    for config_name in TARGET_CONFIGS:
        if config_name in path_str:
            return config_name
    return None


def find_csv_files(root: Path) -> Dict[Tuple[str, str, str, str], List[Path]]:
    grouped: Dict[Tuple[str, str, str, str], List[Path]] = {}

    for dataset in DATASETS:
        for algo_label, algo_dir in ALGORITHMS.items():
            algo_root = root / dataset / "Results" / algo_dir
            if not algo_root.exists():
                continue

            for csv_path in algo_root.rglob(FILE_NAME):
                path_str = str(csv_path)
                case_name = detect_case(path_str)
                config_name = detect_config(path_str)

                if config_name is None:
                    continue

                key = (dataset, case_name, config_name, algo_label)
                grouped.setdefault(key, []).append(csv_path)

    return grouped


def get_qps_series(df: pd.DataFrame, query_count: Optional[float]) -> Optional[pd.Series]:
    if "QPS" in df.columns:
        qps = pd.to_numeric(df["QPS"], errors="coerce")
        if qps.notna().any():
            return qps

    if "Average_Time_ms" not in df.columns:
        return None

    avg_time = pd.to_numeric(df["Average_Time_ms"], errors="coerce")
    if not avg_time.notna().any():
        return None

    effective_query_count = 1.0 if query_count is None else query_count
    qps = effective_query_count * 1000.0 / avg_time
    qps = qps.replace([math.inf, -math.inf], pd.NA)
    return qps


def get_recall_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ("Average_Recall", "Recall"):
        if col in df.columns:
            recall = pd.to_numeric(df[col], errors="coerce")
            if recall.notna().any():
                return recall
    return None


def get_best_qps(csv_path: Path, query_count: Optional[float]) -> Optional[float]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty:
        return None

    recall = get_recall_series(df)
    qps = get_qps_series(df, query_count)
    if recall is None or qps is None:
        return None

    mask = (recall >= 0.9) & qps.notna()
    if not mask.any():
        return None

    best_qps = qps[mask].max()
    if pd.isna(best_qps):
        return None
    return float(best_qps)


def pick_best_file(
    csv_paths: List[Path],
    query_count: Optional[float],
) -> Tuple[Optional[float], Optional[Path]]:
    best_qps: Optional[float] = None
    best_path: Optional[Path] = None

    for csv_path in csv_paths:
        qps = get_best_qps(csv_path, query_count)
        if qps is None:
            continue
        if best_qps is None or qps > best_qps:
            best_qps = qps
            best_path = csv_path

    return best_qps, best_path


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def build_rows(
    grouped_files: Dict[Tuple[str, str, str, str], List[Path]],
    query_count: Optional[float],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for dataset in DATASETS:
        for config_name in TARGET_CONFIGS:
            for case_name in CONFIG_CASES[config_name]:
                plus_key = (dataset, case_name, config_name, "SmartRoute+")
                base_key = (dataset, case_name, config_name, "SmartRoute")

                plus_qps, plus_path = pick_best_file(
                    grouped_files.get(plus_key, []),
                    query_count,
                )
                base_qps, base_path = pick_best_file(
                    grouped_files.get(base_key, []),
                    query_count,
                )

                ratio: Optional[float] = None
                if plus_qps is not None and base_qps not in (None, 0):
                    ratio = plus_qps / base_qps

                rows.append(
                    {
                        "dataset": dataset,
                        "case": case_name,
                        "config": config_name,
                        "smartroute_plus_max_qps": format_float(plus_qps),
                        "smartroute_max_qps": format_float(base_qps),
                        "qps_ratio_plus_over_smartroute": format_float(ratio),
                    }
                )

    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser()

    grouped_files = find_csv_files(root)
    rows = build_rows(grouped_files, args.query_count)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"结果已写入: {output}")
    print(f"扫描根目录: {root}")
    print(f"共输出 {len(rows)} 行。")


if __name__ == "__main__":
    main()
