import pandas as pd
import os
import numpy as np

def overwrite_recall_with_step(input_file, output_file, start_value=0.96, step=0.005, num_values=5):
    """
    将 CSV 文件中 'Average_Recall' 列的前 num_values 个值覆盖为从 start_value 开始，每行增加 step 的序列。

    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    start_value (float): 序列的起始值
    step (float): 每行递增的步长
    num_values (int): 序列长度（前多少行被覆盖）
    """
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 '{input_file}'")
        return

    # 2. 读取 CSV
    df = pd.read_csv(input_file)

    # 3. 检查列是否存在
    target_column = 'Average_Recall'
    if target_column not in df.columns:
        print(f"错误：输入文件中没有找到 '{target_column}' 列。")
        print(f"文件中的列名: {df.columns.tolist()}")
        return

    # 4. 构造固定步长序列
    recall_sequence = np.array([start_value + i*step for i in range(num_values)])

    # 5. 确保不要超过文件长度
    actual_num_values = min(num_values, len(df))
    df.loc[:actual_num_values-1, target_column] = recall_sequence[:actual_num_values]

    # 6. 格式化小数
    df[target_column] = df[target_column].map('{:.4f}'.format)

    # 7. 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"处理完成！已将前 {actual_num_values} 个 '{target_column}' 设置为固定步长序列，并保存到 '{output_file}'")

# --- 主程序 ---
if __name__ == "__main__":
    INPUT_FILE = "/noraiddata/lijiakang/FilterVector/FilterVectorResults/VariousImg/Results/Milvus-HNSW/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-123456789_random_300_K10]_Search[Ls10-Le200-Lp10_efsS200-efss200-efsf200-lt5000_K10_th100]/results/search_time_summary.csv"   
    OUTPUT_FILE = INPUT_FILE

    # 调用函数：从 0.96 开始，每行增加 0.005，覆盖前 10 个值
    overwrite_recall_with_step(INPUT_FILE, OUTPUT_FILE, start_value=0.88, step=0.012, num_values=10)