# 实验 1：低 Lsearch 参数，仅主目录，运行 SmartRoute 系列算法。
# 实验 2：高 Lsearch 参数，仅主目录，运行 UNG 系列算法。
# 实验 3：低 Lsearch 参数，主目录（运行 ACORN 系列算法） + 4个子目录（运行全部 6 个低延迟算法）。
# 实验 4：高 Lsearch 参数，仅4个子目录，运行 UNG 系列算法。
# 下面的脚本会自动读取指定的文件夹中的所有旧 JSON 文件，应用上述逻辑重构数据结构，并输出到新的文件夹中。

import json
import os
import glob
import copy

def process_single_json(input_path, output_path):
    """处理单个 JSON 文件的转换逻辑"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 校验：旧的格式通常包含 2 个 experiment 块 (低 Lsearch 和 高 Lsearch)
    experiments = data.get("experiments", [])
    if len(experiments) != 2:
        print(f"⚠️ 跳过 {os.path.basename(input_path)}: 预期有 2 个实验组，但发现了 {len(experiments)} 个。")
        return

    exp_low = experiments[0]   # 低 Lsearch 组 (SmartRoute / ACORN)
    exp_high = experiments[1]  # 高 Lsearch 组 (UNG)

    dataset_name = exp_low["dataset_name"]
    shared_config_low = exp_low["shared_config"]
    shared_config_high = exp_high["shared_config"]

    # 区分主目录任务和子目录任务
    main_task = None
    subdir_tasks = []

    for task in exp_low["tasks"]:
        name = task["query_dir_name"]
        # 通过后缀判断是否为拆分的子目录
        if any(suffix in name for suffix in ["len_large", "len_small", "ppass_large", "ppass_small"]):
            subdir_tasks.append(task)
        else:
            main_task = task

    if not main_task:
        print(f"⚠️ 跳过 {os.path.basename(input_path)}: 找不到主目录的 task。")
        return

    # 定义算法组
    algos_smart = ["SmartRoute", "SmartRoute+", "pre-filter"]
    algos_acorn = ["ACORN-gamma", "ACORN-1", "NaviX-ACORN"]
    algos_all_low = algos_smart + algos_acorn
    algos_ung = ["UNG+", "UNG-nTfalse"]

    # ================= 构建新的实验列表 =================
    new_experiments = []

    # 实验 1：低配置，仅主目录，SmartRoute 系列
    task_exp1 = copy.deepcopy(main_task)
    task_exp1["algorithms"] = algos_smart
    new_experiments.append({
        "dataset_name": dataset_name,
        "shared_config": copy.deepcopy(shared_config_low),
        "tasks": [task_exp1]
    })

    # 实验 2：高配置，仅主目录，UNG 系列
    task_exp2 = copy.deepcopy(main_task)
    task_exp2["algorithms"] = algos_ung
    new_experiments.append({
        "dataset_name": dataset_name,
        "shared_config": copy.deepcopy(shared_config_high),
        "tasks": [task_exp2]
    })

    # 实验 3：低配置，主目录(ACORN) + 子目录(全部低延迟算法)
    tasks_exp3 = []
    task_exp3_main = copy.deepcopy(main_task)
    task_exp3_main["algorithms"] = algos_acorn
    tasks_exp3.append(task_exp3_main)

    for st in subdir_tasks:
        new_st = copy.deepcopy(st)
        new_st["algorithms"] = algos_all_low
        tasks_exp3.append(new_st)

    new_experiments.append({
        "dataset_name": dataset_name,
        "shared_config": copy.deepcopy(shared_config_low),
        "tasks": tasks_exp3
    })

    # 实验 4：高配置，仅子目录，UNG 系列
    tasks_exp4 = []
    for st in subdir_tasks:
        new_st = copy.deepcopy(st)
        new_st["algorithms"] = algos_ung
        tasks_exp4.append(new_st)

    new_experiments.append({
        "dataset_name": dataset_name,
        "shared_config": copy.deepcopy(shared_config_high),
        "tasks": tasks_exp4
    })

    # ================= 写入新文件 =================
    new_data = {"experiments": new_experiments}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=3) # 保持 3 个空格的缩进，与原格式一致
    
    print(f"✅ 成功转换: {os.path.basename(output_path)}")

# ==========================================
# 主程序：批量处理目录
# ==========================================
if __name__ == "__main__":
    INPUT_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorCode/experiment_json/202603-random-300-mix-len"   # 存放旧 json 的文件夹
    OUTPUT_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorCode/experiment_json/202603-200-random-300-mix-len"  # 输出新 json 的文件夹

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 匹配输入目录下所有的 .json 文件
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    if not json_files:
        print(f"❌ 在 {INPUT_DIR} 目录下没有找到 JSON 文件！")
    else:
        print(f"🚀 开始批量转换，共发现 {len(json_files)} 个文件...\n")
        for file_path in json_files:
            file_name = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_DIR, file_name)
            process_single_json(file_path, output_path)
        
        print("\n🎉 全部 JSON 转换完成！")