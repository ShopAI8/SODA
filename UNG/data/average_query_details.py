import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate global average metrics and algo ratio from query details CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the result CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"❌ 找不到输入文件: {args.input_csv}")
        return

    # ⚠️ 你的数据是空格分隔
    df = pd.read_csv(args.input_csv)

    if 'QueryID' not in df.columns:
        print("❌ CSV 中没有 QueryID 列")
        return

    # =========================
    # 1️⃣ 全局平均值（不去重！！）
    # =========================
    numeric_df = df.select_dtypes(include='number')

    if 'QueryID' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['QueryID'])

    df_avg = numeric_df.mean().to_frame().T

    # =========================
    # 2️⃣ Algo 占比（按 QueryID 去重）
    # =========================
    df_dedup = df.drop_duplicates(subset=['QueryID'], keep='first')

    print(f"原始行数: {len(df)}, 去重后 Query 数: {len(df_dedup)}")

    if 'Algo_Choice' in df_dedup.columns:
        algo_ratio = df_dedup['Algo_Choice'].value_counts(normalize=True)

        algo_ratio_df = algo_ratio.to_frame().T
        algo_ratio_df.columns = [f"Algo_{int(col)}_ratio" for col in algo_ratio_df.columns]

        df_result = pd.concat([df_avg, algo_ratio_df], axis=1)
    else:
        print("⚠️ 没有 Algo_Choice 列")
        df_result = df_avg

    # =========================
    # 3️⃣ 保存结果
    # =========================
    df_result.to_csv(args.output_csv, index=False)

    print("✅ 完成：")
    print("   - 全局平均值（未去重）")
    print("   - Algo_Choice 占比（按 Query 去重）")
    print(f"📁 输出: {args.output_csv}")

if __name__ == "__main__":
    main()