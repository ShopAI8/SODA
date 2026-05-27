# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------
# 文件读写函数
# --------------------------------------------------------------------------------

def read_fvecs(filename, c_contiguous=True):
    """读取 .fvecs 文件，返回一个 numpy 数组。"""
    try:
        with open(filename, 'rb') as f:
            d_bytes = f.read(4)
            if not d_bytes: return np.array([])
            d = np.frombuffer(d_bytes, dtype='int32')[0]
            f.seek(0)
            record_size = 4 + d * 4
            file_content = f.read()
        
        num_vectors = len(file_content) // record_size
        if num_vectors == 0: return np.array([])
            
        data = np.frombuffer(file_content, dtype='float32').reshape(num_vectors, d + 1)
        vectors = data[:, 1:].copy()
        
        return vectors.copy(order='C') if c_contiguous else vectors
    except FileNotFoundError:
        print(f"❌ 错误: .fvecs 文件未找到 -> {filename}")
        return None

def write_fvecs(filename, vecs):
    """将一个 numpy 数组写入 .fvecs 文件。"""
    if vecs.ndim != 2: raise ValueError("输入必须是一个二维数组")
    if vecs.shape[0] == 0: return
    num_vectors, dim = vecs.shape
    with open(filename, 'wb') as f:
        for i in range(num_vectors):
            f.write(np.array([dim], dtype='int32').tobytes())
            f.write(vecs[i, :].astype('float32').tobytes())


# --------------------------------------------------------------------------------
# 核心重复函数
# --------------------------------------------------------------------------------

def process_folder_repeat(input_dir, output_dir, dataset, repeat_times):
    """
    对单个文件夹内的 fvecs, txt, 和 profiled csv 进行整体循环重复。
    """
    print(f"\n📂 正在处理目录: {os.path.basename(input_dir)}")
    
    fvecs_path = os.path.join(input_dir, f"{dataset}_query.fvecs")
    txt_path = os.path.join(input_dir, f"{dataset}_query_labels.txt")
    
    # 使用 glob 动态寻找 profiled_*.csv 文件
    csv_files = glob.glob(os.path.join(input_dir, "profiled_*.csv"))
    csv_path = csv_files[0] if csv_files else None

    # 1. 检查并读取 fvecs 和 txt
    if not os.path.exists(fvecs_path) or not os.path.exists(txt_path):
        print(f"   ⚠️ 跳过: 缺失 .fvecs 或 .txt 文件。")
        return

    vectors = read_fvecs(fvecs_path)
    with open(txt_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        
    print(f"   -> 原始数据量: {len(vectors)} 条")

    # 2. 整体重复
    repeated_vectors = np.tile(vectors, (repeat_times, 1))
    repeated_labels = labels * repeat_times

    # 3. 写入输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    out_fvecs = os.path.join(output_dir, f"{dataset}_query.fvecs")
    out_txt = os.path.join(output_dir, f"{dataset}_query_labels.txt")
    
    write_fvecs(out_fvecs, repeated_vectors)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.writelines(repeated_labels)

    # 4. 如果存在 profiled CSV，同步重复并写入
    if csv_path:
        df = pd.read_csv(csv_path)
        repeated_df = pd.concat([df] * repeat_times, ignore_index=True)
        out_csv = os.path.join(output_dir, os.path.basename(csv_path))
        repeated_df.to_csv(out_csv, index=False)
        print(f"   -> 同步重复 CSV: {os.path.basename(csv_path)}")

    print(f"   ✅ 完成! 重复后数据量: {len(repeated_vectors)} 条 -> 已存入 {os.path.basename(output_dir)}")


# =========================================================================
# --- 主流程 (MAIN) ---
# =========================================================================

if __name__ == "__main__":
    # ===================== 用户配置区 =====================
    DATASET = "Genome"
    REPEAT_TIMES = 300
    
    # 填写上一步 select_vector_and_labels_new.py 生成的【大文件夹】路径
    INPUT_BASE_DIR = "/noraiddata/lijiakang/FilterVector/FilterVectorData/Genome/query_select_200_A_B_C-weighted-sub-base-123456789"
    
    # 填写希望输出的【新大文件夹】路径
    OUTPUT_BASE_DIR = f"{INPUT_BASE_DIR}_random_{REPEAT_TIMES}"
    # ======================================================

    print("="*80)
    print(f"🚀 开始执行: 整体循环重复任务 (Repeat = {REPEAT_TIMES})")
    print("="*80)

    # 1. 处理主目录 (大文件夹)
    process_folder_repeat(INPUT_BASE_DIR, OUTPUT_BASE_DIR, DATASET, REPEAT_TIMES)

    # 2. 扫描并处理子目录 (四个小文件夹)
    # 获取 INPUT_BASE_DIR 下的所有一级子目录
    sub_dirs = [d for d in os.listdir(INPUT_BASE_DIR) if os.path.isdir(os.path.join(INPUT_BASE_DIR, d))]
    
    for sub_name in sub_dirs:
        in_sub_path = os.path.join(INPUT_BASE_DIR, sub_name)
        out_sub_path = os.path.join(OUTPUT_BASE_DIR, sub_name)
        process_folder_repeat(in_sub_path, out_sub_path, DATASET, REPEAT_TIMES)

    print("\n" + "="*80)
    print("🎉 全部重复任务完成！大文件夹与子文件夹的数据结构已完美复刻。")
    print("="*80)