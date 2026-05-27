import os
import shutil

# ================= 配置区域 =================
# 原始结果所在的主文件夹
SOURCE_ROOT = "/noraiddata/lijiakang/FilterVector/FilterVectorResults"

# 备份文件存放的统一目标路径
BACKUP_ROOT = "/noraiddata/lijiakang/FilterVector/FilterVectorResults_Backup"

# 指定要备份的文件名。如果想备份文件夹下的所有文件，请将此项设为 None
TARGET_FILE = None

# =========================================================
# 路径匹配逻辑
# =========================================================
def get_path_condition(path):
    """
    判断路径是否满足你定义的两种情况之一。
    情况1: 必须包含"_K10_th100", 且不包含"len"和"ppass"
    情况2: 必须包含"_K10_th100"和"len"
    
    返回对应的数字用于日志统计，返回 0 表示跳过。
    """
    has_k10 = "_K10_th100" in path
    has_len = "len" in path
    has_ppass = "ppass" in path
    
    if has_k10 and not has_len and not has_ppass:
        return 1
    if has_k10 and has_len:
        return 2
        
    return 0

# =========================================================
# 主备份逻辑
# =========================================================
def run_backup():
    print("开始执行自动化备份任务...\n")
    count_cond1 = 0
    count_cond2 = 0
    
    # 检查原始目录是否存在
    if not os.path.exists(SOURCE_ROOT):
        print(f"[错误] 找不到数据源目录: {SOURCE_ROOT}")
        return

    # 遍历源目录
    for root, dirs, files in os.walk(SOURCE_ROOT):
        # 检查当前目录的文件中是否包含我们需要备份的 TARGET_FILE
        if TARGET_FILE and TARGET_FILE not in files:
            continue
            
        # 判断当前目录是否满足我们的过滤条件
        cond = get_path_condition(root)
        if cond == 0:
            continue
            
        # 计算相对路径 (例如: Amazon/Results/ACORN-gamma/...)
        rel_path = os.path.relpath(root, SOURCE_ROOT)
        
        # 【修改点】不再区分 Condition 文件夹，直接基于相对路径保留原有的数据集划分结构
        dest_dir = os.path.join(BACKUP_ROOT, rel_path)
        
        # 记录统计信息
        if cond == 1:
            count_cond1 += 1
        elif cond == 2:
            count_cond2 += 1
            
        # 创建目标级联目录
        os.makedirs(dest_dir, exist_ok=True)
        
        # 执行文件复制
        for file_name in files:
            if TARGET_FILE and file_name != TARGET_FILE:
                continue
                
            src_file = os.path.join(root, file_name)
            dest_file = os.path.join(dest_dir, file_name)
            
            try:
                # copy2 会尽可能保留文件的元数据（如时间戳）
                shutil.copy2(src_file, dest_file)
                print(f"  [成功 - 匹配情况{cond}] 已备份: {rel_path}/{file_name}")
            except Exception as e:
                print(f"  [错误] 备份失败 {src_file}: {e}")

    print("\n================ 备份任务完成 ================")
    print(f"情况一 (无 len/ppass) 共备份了 {count_cond1} 个路径点。")
    print(f"情况二 (含 len) 共备份了 {count_cond2} 个路径点。")
    print(f"所有文件已统一按原数据集结构合并保存至: {BACKUP_ROOT}")

if __name__ == "__main__":
    run_backup()