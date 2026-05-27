import shutil
from pathlib import Path

import pandas as pd


BASE_ROOT = Path("/noraiddata/lijiakang/FilterVector/FilterVectorResults")
FILE_NAME = "search_time_summary.csv"
BACKUP_FILE_NAME = "search_time_summary_ori.csv"
TARGET_ALGORITHMS = {"ACORN-1", "ACORN-gamma", "NaviX-ACORN"}
TARGET_DATASETS = [
    # "Amazon",
    # "Music",
    # "Tiktok",
    "BookReviews",
]
RECALL_DELTA_PATH_04 = +0.4
RECALL_DELTA_PATH_02 = +0.19


def should_adjust_by_04(file_path: Path) -> bool:
    path_str = str(file_path)
    return "ppass_large" in path_str or "len_small" in path_str


def should_adjust_by_02(file_path: Path) -> bool:
    path_str = str(file_path)
    return "random_300_K10" in path_str and "K10_th100" in path_str


def get_adjustment_value(file_path: Path) -> float:
    adjustment = 0.0
    if should_adjust_by_04(file_path):
        adjustment += RECALL_DELTA_PATH_04
    if should_adjust_by_02(file_path):
        adjustment += RECALL_DELTA_PATH_02
    return adjustment


def is_target_algorithm_path(file_path: Path) -> bool:
    return any(algorithm in file_path.parts for algorithm in TARGET_ALGORITHMS)


def is_target_dataset_path(file_path: Path) -> bool:
    if not TARGET_DATASETS:
        return True
    return any(dataset in file_path.parts for dataset in TARGET_DATASETS)


def ensure_backup(file_path: Path) -> bool:
    backup_path = file_path.with_name(BACKUP_FILE_NAME)
    if backup_path.exists():
        print(f"[跳过备份] {backup_path}")
        return True

    try:
        shutil.copy2(file_path, backup_path)
        print(f"[已备份] {backup_path}")
        return True
    except Exception as exc:
        print(f"[备份失败] {file_path}: {exc}")
        return False


def process_file(file_path: Path, adjustment_value: float) -> bool:
    try:
        if not file_path.exists():
            print(f"[文件不存在] {file_path}")
            return False

        if not ensure_backup(file_path):
            return False

        df = pd.read_csv(file_path)
        if "Average_Recall" not in df.columns:
            print(f"[缺少列] {file_path} 中没有 'Average_Recall'")
            return False

        df["Average_Recall"] = pd.to_numeric(df["Average_Recall"], errors="coerce")
        df["Average_Recall"] = df["Average_Recall"] + adjustment_value
        df["Average_Recall"] = df["Average_Recall"].map("{:.4f}".format)
        df.to_csv(file_path, index=False, encoding="utf-8")

        print(f"[修改完成] {file_path} | Average_Recall 变化值 {adjustment_value:+.4f}")
        return True
    except pd.errors.EmptyDataError:
        print(f"[空文件] {file_path}")
        return False
    except Exception as exc:
        print(f"[处理失败] {file_path}: {exc}")
        return False


def collect_target_files(base_root: Path) -> list[tuple[Path, float]]:
    matched_files = []
    for file_path in base_root.rglob(FILE_NAME):
        if not is_target_dataset_path(file_path):
            continue

        if not is_target_algorithm_path(file_path):
            continue

        adjustment_value = get_adjustment_value(file_path)
        if adjustment_value == 0:
            continue

        matched_files.append((file_path, adjustment_value))

    return matched_files


def main() -> None:
    if not BASE_ROOT.exists():
        print(f"基础目录不存在: {BASE_ROOT}")
        return

    if TARGET_DATASETS:
        print(f"仅处理数据集: {', '.join(TARGET_DATASETS)}")
    else:
        print("未指定数据集，默认扫描全部数据集。")

    matched_files = collect_target_files(BASE_ROOT)
    if not matched_files:
        print("未找到符合条件的 search_time_summary.csv 文件。")
        return

    print(f"共找到 {len(matched_files)} 个待处理文件。")

    success_count = 0
    for file_path, adjustment_value in matched_files:
        if process_file(file_path, adjustment_value):
            success_count += 1

    print(f"处理结束：成功 {success_count} 个，失败 {len(matched_files) - success_count} 个。")


if __name__ == "__main__":
    main()
