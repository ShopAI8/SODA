# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import csv
from collections import defaultdict

# --------------------------------------------------------------------------------
# 文件读写函数
# --------------------------------------------------------------------------------

def read_fvecs(filename, c_contiguous=True):
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
    if vecs.ndim != 2: raise ValueError("输入必须是一个二维数组")
    if vecs.shape[0] == 0: return
    num_vectors, dim = vecs.shape
    with open(filename, 'wb') as f:
        for i in range(num_vectors):
            f.write(np.array([dim], dtype='int32').tobytes())
            f.write(vecs[i, :].astype('float32').tobytes())

def parse_label_line(line, delimiter=','):
    parts = line.strip().split(delimiter)
    return set(int(p) for p in parts if p)

def write_output_files(output_dir, data_list, dataset, file_suffix):
    """
    将切片后的数据写入 fvecs, txt, 和 csv 文件。
    """
    if not data_list:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    vectors_to_write = np.array([item['vector'] for item in data_list])
    labels_to_write = [item['label'] for item in data_list]
    
    # profiled_*.csv 所需数据
    profiled_to_write = [{'coverage_count': item['coverage'], 'labels': item['sorted_label_str']} for item in data_list]

    fvecs_name = os.path.join(output_dir, f"{dataset}_query.fvecs")
    labels_name = os.path.join(output_dir, f"{dataset}_query_labels.txt")
    csv_path = os.path.join(output_dir, f"profiled_{file_suffix}.csv")

    write_fvecs(fvecs_name, vectors_to_write)
    with open(labels_name, 'w', encoding='utf-8') as f:
        f.writelines(labels_to_write)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['coverage_count', 'labels'])
        writer.writeheader()
        writer.writerows(profiled_to_write)
        
    print(f"   -> 成功为 '{os.path.basename(output_dir)}' 写入 {len(data_list)} 条记录。")


# --------------------------------------------------------------------------------
# 核心分析与拆分函数
# --------------------------------------------------------------------------------

def analyze_and_split_existing_queries(query_dir, dataset, base_labels_path, output_base_dir):
    print("--- 步骤 1/3: 加载已有查询文件与基础标签 ---")
    
    fvecs_path = os.path.join(query_dir, f"{dataset}_query.fvecs")
    txt_path = os.path.join(query_dir, f"{dataset}_query_labels.txt")
    
    # 1. 读取已经存在的查询向量和标签
    all_vectors = read_fvecs(fvecs_path)
    if all_vectors is None: return
    
    with open(txt_path, 'r', encoding='utf-8') as f: 
        all_labels = f.readlines()
        
    # 2. 读取基础标签用于构建倒排索引 (计算 coverage)
    with open(base_labels_path, 'r', encoding='utf-8') as f:
        base_label_sets = [parse_label_line(line, delimiter=',') for line in f if line.strip()]
    total_base_items = len(base_label_sets)
    
    if len(all_vectors) != len(all_labels):
        print("❌ 警告: 向量数量与标签数量不匹配！")
        return

    print(f"✅ 加载完成: {len(all_vectors)} 条查询, {total_base_items} 条基础数据。")

    print("\n--- 步骤 2/3: 构建倒排索引并计算 Length 和 Coverage ---")
    inverted_index = defaultdict(set)
    for i, base_set in enumerate(base_label_sets):
        for label in base_set: inverted_index[label].add(i)
    
    analyzed_data = []
    for index in range(len(all_labels)):
        label_line = all_labels[index]
        query_set = parse_label_line(label_line, delimiter=',')
        
        # 计算 coverage (p_pass)
        coverage = 0
        if query_set:
            try:
                posting_lists = [inverted_index[label] for label in query_set]
                coverage = len(set.intersection(*posting_lists))
            except KeyError: 
                coverage = 0
                
        analyzed_data.append({
            'QueryID': index,
            'vector': all_vectors[index], 
            'label': label_line,
            'coverage': coverage,
            'length': len(query_set),
            'sorted_label_str': " ".join(map(str, sorted(list(query_set))))
        })
        
    profiling_df = pd.DataFrame(analyzed_data)
    print(f"✅ 指标计算完成。")

    print(f"\n--- 步骤 3/3: 拆分数据集并写入文件 ---")
    
    # 提取原文件夹名作为后缀
    dir_name = os.path.basename(query_dir.rstrip('/'))
    file_suffix = dir_name.replace("query_", "") # 提取后缀用于文件命名
    
    N_TOTAL = len(profiling_df)
    N_SPLIT = N_TOTAL // 2
    N_REMAINING = N_TOTAL - N_SPLIT 
    
    print(f"总查询数: {N_TOTAL}, 拆分大小: {N_SPLIT} (Small) / {N_REMAINING} (Large)")
    
    # 拆分逻辑
    df_sorted_len = profiling_df.sort_values('length').reset_index(drop=True)
    df_len_small = df_sorted_len.head(N_SPLIT)
    df_len_large = df_sorted_len.tail(N_REMAINING)
    
    df_sorted_ppass = profiling_df.sort_values('coverage').reset_index(drop=True)
    df_ppass_small = df_sorted_ppass.head(N_SPLIT)
    df_ppass_large = df_sorted_ppass.tail(N_REMAINING)

    # 写入各子文件夹
    len_small_suffix = f"{file_suffix}_len_small"
    write_output_files(os.path.join(output_base_dir, len_small_suffix), df_len_small.to_dict('records'), dataset, len_small_suffix)
    
    len_large_suffix = f"{file_suffix}_len_large"
    write_output_files(os.path.join(output_base_dir, len_large_suffix), df_len_large.to_dict('records'), dataset, len_large_suffix)

    ppass_small_suffix = f"{file_suffix}_ppass_small"
    write_output_files(os.path.join(output_base_dir, ppass_small_suffix), df_ppass_small.to_dict('records'), dataset, ppass_small_suffix)

    ppass_large_suffix = f"{file_suffix}_ppass_large"
    write_output_files(os.path.join(output_base_dir, ppass_large_suffix), df_ppass_large.to_dict('records'), dataset, ppass_large_suffix)

    print("\n🎉 拆分任务完成！")


# =========================================================================
# --- 主流程 ---
# =========================================================================

if __name__ == "__main__":
    # ===================== 用户配置区  =====================

    TASKS = [
        {
            "DATASET": "Amazon",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Amazon/query_select_imp_A_B_C-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Amazon/Amazon_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Amazon/query_select_imp_A_B_C-sub-base-random_300"
        },
        {
            "DATASET": "BookReviews",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/BookReviews/query_select_imp_B_C_D-weighted-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/BookReviews/BookReviews_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/BookReviews/query_select_imp_B_C_D-weighted-sub-base-random_300"
        },
        {
            "DATASET": "Genome",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Genome/query_select_imp_A_B_C-weighted-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Genome/Genome_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Genome/query_select_imp_A_B_C-weighted-sub-base-random_300"
        },
        {
            "DATASET": "Music",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Music/query_select_imp_A_B_C-weighted-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Music/Music_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Music/query_select_imp_A_B_C-weighted-sub-base-random_300"
        },
        {
            "DATASET": "Reviews",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Reviews/Reviews_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-random_300"
        },
        {
            "DATASET": "Tiktok",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Tiktok/query_select_imp_A_B_C-weighted-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Tiktok/Tiktok_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Tiktok/query_select_imp_A_B_C-weighted-sub-base-random_300"
        },
        {
            "DATASET": "VariousImg",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/VariousImg/query_select_imp_A_B_C-weighted-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/VariousImg/VariousImg_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/VariousImg/query_select_imp_A_B_C-weighted-sub-base-random_300"
        },
        {
            "DATASET": "Laion",
            "EXISTING_QUERY_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Laion/query_select_imp_C_D-weighted-sub-base-random_300",
            "BASE_LABELS_PATH": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Laion/Laion_base_labels.txt",
            "OUTPUT_BASE_DIR": "/noraiddata/lijiakang/FilterVector/FilterVectorData/Laion/query_select_imp_C_D-weighted-sub-base-random_300"
        }
    ]
    # =========================================================================
    
    print("="*80)
    print(f"🚀 开始批量执行现存查询数据集的二次拆分任务 (共 {len(TASKS)} 个任务)")
    print("="*80)
    
    for i, task in enumerate(TASKS, 1):
        print(f"\n▶ 开始执行任务 [{i}/{len(TASKS)}]: 数据集 {task['DATASET']}")
        print(f"  输入目录: {task['EXISTING_QUERY_DIR']}")
        
        try:
            analyze_and_split_existing_queries(
                query_dir=task['EXISTING_QUERY_DIR'],
                dataset=task['DATASET'],
                base_labels_path=task['BASE_LABELS_PATH'],
                output_base_dir=task['OUTPUT_BASE_DIR']
            )
            print(f"✅ 任务 [{i}/{len(TASKS)}] ({task['DATASET']}) 执行成功！")
        except Exception as e:
            print(f"❌ 任务 [{i}/{len(TASKS)}] ({task['DATASET']}) 发生致命错误: {e}")
            print("➡️ 跳过此任务，继续执行下一个...\n")
            
    print("\n" + "="*80)
    print("🎉 所有批量拆分任务处理完毕！")
    print("="*80)