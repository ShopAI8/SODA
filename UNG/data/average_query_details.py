import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate global average metrics and algo ratio from query details CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the result CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return

    # Read query-level detail records.
    df = pd.read_csv(args.input_csv)

    if 'QueryID' not in df.columns:
        print("Error: CSV file does not contain a QueryID column")
        return

    # =========================
    # 1. Global averages without deduplication
    # =========================
    numeric_df = df.select_dtypes(include='number')

    if 'QueryID' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['QueryID'])

    df_avg = numeric_df.mean().to_frame().T

    # =========================
    # 2. Algorithm ratios after deduplicating by QueryID
    # =========================
    df_dedup = df.drop_duplicates(subset=['QueryID'], keep='first')

    print(f"Original rows: {len(df)}, deduplicated queries: {len(df_dedup)}")

    if 'Algo_Choice' in df_dedup.columns:
        algo_ratio = df_dedup['Algo_Choice'].value_counts(normalize=True)

        algo_ratio_df = algo_ratio.to_frame().T
        algo_ratio_df.columns = [f"Algo_{int(col)}_ratio" for col in algo_ratio_df.columns]

        df_result = pd.concat([df_avg, algo_ratio_df], axis=1)
    else:
        print("Warning: Algo_Choice column not found")
        df_result = df_avg

    # =========================
    # 3. Save results
    # =========================
    df_result.to_csv(args.output_csv, index=False)

    print("Completed:")
    print("   - Global averages without deduplication")
    print("   - Algo_Choice ratios after query-level deduplication")
    print(f"Output: {args.output_csv}")

if __name__ == "__main__":
    main()
