import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

# ==========================================
# 1. 全局配置区域
# ==========================================
DATASET_NAME = "BookReviews"  
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"
CSV_PATH = os.path.join(BASE_DIR, "EDA_Plots", DATASET_NAME, f"{DATASET_NAME}_aligned_results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, DATASET_NAME, "SelectModels", "fast_smart_route")

ALGO_LIST = ['ACORN-gamma', 'ACORN-improved', 'NaviX', 'UNG-nTfalse', 'UNG-nTtrue', 'pre-filter']
ACORN_FAMILY = ['ACORN-gamma', 'ACORN-improved', 'NaviX']

MIN_SAMPLES_RATIO = 0.025  
CORE_MODEL_TYPE = "RandomForest"

# ==========================================
# 2. 核心功能与工厂函数
# ==========================================
def create_classifier(model_type="RandomForest", **kwargs):
    if model_type == "RandomForest":
        params = {'n_estimators': 100, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params.update(kwargs)
        return RandomForestClassifier(**params)
    elif model_type == "XGBoost":
        import xgboost as xgb
        params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        params.update(kwargs)
        return xgb.XGBClassifier(**params)
    elif model_type == "LightGBM":
        import lightgbm as lgb
        params = {'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def label_best_algorithm(df, min_recall=0.90):
    best_algos = []
    for idx, row in df.iterrows():
        candidates = []
        for algo in ALGO_LIST:
            recall_col = f'Recall_{algo}'
            time_col = f'true_search_time_ms_{algo}'
            if recall_col in row and time_col in row and pd.notna(row[recall_col]) and pd.notna(row[time_col]):
                candidates.append({'algo': algo, 'recall': row[recall_col], 'time': row[time_col]})
                
        if not candidates:
            best_algos.append('Unknown')
            continue
            
        qualified = [c for c in candidates if c['recall'] >= min_recall]
        best = min(qualified, key=lambda x: x['time']) if qualified else max(candidates, key=lambda x: x['recall'])
        best_algos.append(best['algo'])
        
    return pd.Series(best_algos, index=df.index)

def generate_cascade_features(df):
    # ==========================================
    # L1 特征：QuerySize, ExactCandSize, CandSize, GlobalPpass 及其组合
    # ==========================================
    X_L1 = pd.DataFrame(index=df.index)
    
    # 基础特征
    X_L1['QuerySize'] = df['QuerySize']
    X_L1['ExactCandSize'] = df['ExactCandSize']
    X_L1['CandSize'] = df['CandSize']
    X_L1['GlobalPpass'] = df['GlobalPpass']
    
    # 组合特征
    X_L1['Log_ExactCandSize'] = np.log1p(df['ExactCandSize'])
    X_L1['Log_CandSize'] = np.log1p(df['CandSize'])
    X_L1['Cand_per_Query'] = df['ExactCandSize'] / (df['QuerySize'] + 1e-9)
    X_L1['Ppass_x_Query'] = df['GlobalPpass'] * df['QuerySize']
    X_L1['Exact_vs_Cand'] = df['ExactCandSize'] / (df['CandSize'] + 1e-9) # 过滤效率指标

    # ==========================================
    # L2 特征：L1全特征 + NumEntries, NumDescendants 及其组合
    # ==========================================
    X_L2 = X_L1.copy() # L2 继承 L1 的所有特征
    
    # 基础特征
    X_L2['NumEntries'] = df['NumEntries']
    X_L2['NumDescendants'] = df['NumDescendants']
    
    # 追加 L2 独有组合特征
    X_L2['Log_NumDescendants'] = np.log1p(df['NumDescendants'])
    X_L2['Desc_per_Entry'] = df['NumDescendants'] / (df['NumEntries'] + 1e-9)
    X_L2['Interaction_Desc_Ppass'] = df['NumDescendants'] * df['GlobalPpass']
    X_L2['Desc_per_Global'] = df['NumDescendants'] / (df['GlobalPpass'] * len(df) + 1e-9) # 相对膨胀率
    X_L2['Desc_vs_ExactCand'] = df['NumDescendants'] / (df['ExactCandSize'] + 1e-9) # 后代冗余度
    X_L2['Entry_vs_QuerySize'] = df['NumEntries'] / (df['QuerySize'] + 1e-9)
    
    # 清理异常值
    X_L1.replace([np.inf, -np.inf], np.nan, inplace=True); X_L1.fillna(0, inplace=True)
    X_L2.replace([np.inf, -np.inf], np.nan, inplace=True); X_L2.fillna(0, inplace=True)
    
    return X_L1, X_L2

def run_ablation_study(df):
    """
    针对各种特征族的消融实验。
    自动将基础特征与其衍生组合特征一并切除，进行公平对比。
    """
    print("--- 正在运行特征消融实验 ---")
    
    # 重新构建完整的 L2 特征作为完全体候选
    _, X_abl = generate_cascade_features(df)
    
    valid_mask = df['Best_Algo'].notna() & (df['Best_Algo'] != 'Unknown')
    X_clean = X_abl[valid_mask]
    y_clean = df.loc[valid_mask, 'Best_Algo']
    
    if len(X_clean) == 0:
        return {}

    all_features = list(X_clean.columns)
    
    # 辅助函数：通过关键词屏蔽一组特征（含衍生特征）
    def get_ablated_features(exclude_keywords):
        return [f for f in all_features if not any(k in f for k in exclude_keywords)]

    configs = {
        "完全体 (All L2 Features)": all_features,
        "消融: 移除 NumEntries 族": get_ablated_features(["NumEntries", "Entry"]),
        "消融: 移除 NumDescendants 族": get_ablated_features(["NumDescendants", "Desc"]),
        "消融: 移除 GlobalPpass 族": get_ablated_features(["GlobalPpass", "Ppass", "Global"]),
        "极端消融: 仅用 L1 特征 (无图导航)": get_ablated_features(["NumEntries", "Entry", "NumDescendants", "Desc"])
    }
    
    results = {}
    for config_name, feat_cols in configs.items():
        X_curr = X_clean[feat_cols]
        X_train, X_test, y_train, y_test = train_test_split(X_curr, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
        
        clf = create_classifier(model_type=CORE_MODEL_TYPE)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[config_name] = acc
        
    return results

def extract_cpp_rules_from_tree(tree_model, feature_names, class_names):
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            cpp_code = f"{indent}if ({name} <= {threshold:.4f}) {{\n"
            cpp_code += recurse(tree_.children_left[node], depth + 1)
            cpp_code += f"{indent}}} else {{\n"
            cpp_code += recurse(tree_.children_right[node], depth + 1)
            cpp_code += f"{indent}}}\n"
            return cpp_code
        else:
            class_idx = np.argmax(tree_.value[node][0])
            try:
                real_cls = class_names[class_idx]
            except ValueError:
                real_cls = str(class_idx)
            return f"{indent}return {real_cls};\n"

    return recurse(0, 2)

def train_and_evaluate_model(X, y, target_map, model_name, output_dir, use_smote=True):
    y_mapped = y.map(target_map)
    valid_mask = y_mapped.notna()
    X_clean, y_clean = X[valid_mask], y_mapped[valid_mask].astype(int)
    
    if CORE_MODEL_TYPE == "XGBoost":
        unique_labels = sorted(y_clean.unique())
        remapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels)}
        y_clean = y_clean.map(remapping)
        internal_to_original = {new_lbl: old_lbl for old_lbl, new_lbl in remapping.items()}
    else:
        internal_to_original = {lbl: lbl for lbl in y_clean.unique()}

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
    
    train_size_orig = len(X_train)
    
    if use_smote:
        try:
            min_samples = y_train.value_counts().min()
            safe_k = min(5, min_samples - 1) if min_samples > 1 else 1
            if min_samples > 1:
                smote = SMOTE(random_state=42, k_neighbors=safe_k)
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception:
            pass

    classifier = create_classifier(model_type=CORE_MODEL_TYPE)
    
    start_time = time.perf_counter()
    classifier.fit(X_train, y_train)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    y_test_orig = y_test.map(internal_to_original)
    y_pred_orig = pd.Series(y_pred).map(internal_to_original)
    
    inv_map = {v: k for k, v in target_map.items()}
    present_labels = sorted(y_test_orig.unique())
    target_names = [inv_map[l] for l in present_labels]
    
    report_str  = f"  ▶ 底层引擎     : {CORE_MODEL_TYPE}\n"
    smote_status = f" (SMOTE后 {len(X_train)} 条)" if use_smote else " (未启用SMOTE)"
    report_str += f"  ▶ 数据分布     : 训练集 {train_size_orig} 条{smote_status} | 测试集 {len(X_test)} 条\n"
    report_str += f"  ▶ 拟合耗时     : {duration_ms:.2f} ms\n"
    report_str += f"  ▶ 测试集准确率 : {acc:.4%}\n\n"
    
    report_str += "  [核心指标 (Classification Report)]\n"
    cls_report = classification_report(y_test_orig, y_pred_orig, labels=present_labels, target_names=target_names, zero_division=0)
    report_str += "  " + cls_report.replace('\n', '\n  ') + "\n"
    
    cm = confusion_matrix(y_test_orig, y_pred_orig, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{name}" for name in target_names], columns=[f"Pred_{name}" for name in target_names])
    report_str += "  [混淆矩阵 (Confusion Matrix)]\n"
    report_str += "  " + cm_df.to_string().replace('\n', '\n  ') + "\n"
    
    if SKL2ONNX_AVAILABLE and CORE_MODEL_TYPE == "RandomForest":
        onnx_filename = os.path.join(output_dir, f"{model_name}.onnx")
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(classifier, initial_types=initial_type, target_opset=15)
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
    if hasattr(classifier, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': classifier.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_str = importance_df.to_string(formatters={'Importance': '{:.4f}'.format})
    else:
        importance_str = "当前模型引擎不支持直接提取 Feature Importances。"
        
    return report_str, importance_str, duration_ms, classifier

# ==========================================
# 3. 主干执行流
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f"SmartRoute_Cascade_Report_{DATASET_NAME}.txt")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到数据文件: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    total_queries = len(df)
    
    df['Best_Algo'] = label_best_algorithm(df, min_recall=0.90)
    
    # 运行消融实验
    ablation_results = run_ablation_study(df)
    
    X_L1, X_L2 = generate_cascade_features(df)
    
    acorn_mask = df['Best_Algo'].isin(ACORN_FAMILY)
    acorn_count = acorn_mask.sum()
    acorn_ratio = acorn_count / total_queries
    use_l1_5_model = acorn_ratio >= MIN_SAMPLES_RATIO
    
    ung_mask = df['Best_Algo'].str.startswith('UNG')
    optimal_l2_inflow_rate = ung_mask.sum() / total_queries
    
    cpp_deployment_code = ""
    l1_5_duration_ms = 0.0
    l1_5_acc = 0.0
    l1_5_status_msg = ""

    # --- Step 1: 动态 L1.5 专家模型 ---
    if use_l1_5_model:
        X_L1_5 = X_L1[acorn_mask]
        y_L1_5 = df.loc[acorn_mask, 'Best_Algo']
        
        l1_5_classes = y_L1_5.unique().tolist()
        class_map = {cls: idx for idx, cls in enumerate(l1_5_classes)}
        y_L1_5_mapped = y_L1_5.map(class_map)
        
        dt_expert = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
        
        t_start = time.perf_counter()
        dt_expert.fit(X_L1_5, y_L1_5_mapped)
        l1_5_duration_ms = (time.perf_counter() - t_start) * 1000.0
        
        y_l1_5_pred = dt_expert.predict(X_L1_5)
        l1_5_acc = accuracy_score(y_L1_5_mapped, y_l1_5_pred)
        
        if l1_5_acc < 0.60:
            majority_algo = y_L1_5.value_counts().index[0]
            l1_5_status_msg = f"已触发智能熔断！准确率仅为 {l1_5_acc:.2%}，不可分。采用直接映射至 majority class: {majority_algo}。"
            cpp_deployment_code = (
                "  if (l1_res == 1) {\n"
                f"      // [L1 极速拦截]: 命中 ACORN_Family，L1.5 已熔断，直接走 {majority_algo}\n"
                f"      return {majority_algo};\n"
                "  }\n"
            )
        else:
            l1_5_status_msg = f"正常生成。反映嵌套 if-else 分支的纯度与可靠性。"
            cpp_rules = extract_cpp_rules_from_tree(dt_expert, X_L1_5.columns, l1_5_classes)
            cpp_deployment_code = (
                "  if (l1_res == 1) {\n"
                "      // [L1 极速拦截]: 命中 ACORN_Family，进入 L1.5 微路由规则\n"
                f"{cpp_rules}"
                "  }\n"
            )
    else:
        cpp_deployment_code = (
            "  if (l1_res == 1) {\n"
            "      // [L1 极速拦截]: 命中 ACORN_Family，样本过少，采用默认硬规则兜底\n"
            "      return ACORN-gamma;\n"
            "  }\n"
        )
    
    # --- Step 2: Layer 1 ---
    df_l1_target = df['Best_Algo'].replace({a: 'ACORN_Family' for a in ACORN_FAMILY})
    df_l1_target = df_l1_target.replace({'UNG-nTfalse': 'NEED_ELS', 'UNG-nTtrue': 'NEED_ELS'})
    
    L1_TARGET_MAP = {'NEED_ELS': 0, 'ACORN_Family': 1, 'pre-filter': 2}
    report_l1, imp_l1, l1_duration_ms, l1_clf = train_and_evaluate_model(X_L1, df_l1_target, L1_TARGET_MAP, "smart_route_L1_router", OUTPUT_DIR, use_smote=True)
    
    y_pred_all_l1 = l1_clf.predict(X_L1)
    leak_count = np.sum(y_pred_all_l1 == 0)
    leak_ratio = leak_count / total_queries
    
    # --- Step 3: Layer 2 ---
    L2_TARGET_MAP = {
        'UNG-nTfalse': 0, 'UNG-nTtrue': 1, 'ACORN-gamma': 2, 
        'ACORN-improved': 3, 'NaviX': 4, 'pre-filter': 5
    }
    
    report_l2, imp_l2, l2_duration_ms, l2_clf = train_and_evaluate_model(X_L2, df['Best_Algo'], L2_TARGET_MAP, "smart_route_L2_router", OUTPUT_DIR, use_smote=False)

    # ==========================================
    # 终极详尽报告生成
    # ==========================================
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃                           FastSmartRoute                           ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 基座模型: {CORE_MODEL_TYPE:<16}  数据集: {DATASET_NAME:<15}            ┃\n")
        f.write(f"┃ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<40} ┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
        
        f.write("【壹 | 原始数据算法支配域分布 (Ground Truth)】\n")
        f.write("-" * 70 + "\n")
        for algo, count in df['Best_Algo'].value_counts().items():
            f.write(f"  > {algo:<18} : {count:>5} 次 ({count/total_queries:>6.2%})\n")
        f.write("\n")
        
        # 新增：消融实验板块
        if ablation_results:
            base_acc = ablation_results.get("完全体 (All L2 Features)", 0)
            f.write("【贰 | 特征消融实验评估 (Ablation Study on Graph Routing Features)】\n")
            f.write("-" * 70 + "\n")
            f.write("  [实验目的] 在直接预测全域 6 种算法的最佳配置时，评测核心图特征族及其衍生特征的消融衰减。\n\n")
            f.write(f"  {'配置方案':<40} | {'预测准确率':<12} | {'性能衰减':<12}\n")
            f.write("  " + "-" * 70 + "\n")
            for config, acc in ablation_results.items():
                drop = base_acc - acc
                drop_str = f"↓ {drop:.2%}" if drop > 0 else "-"
                if config == "完全体 (All L2 Features)":
                    drop_str = "Baseline"
                f.write(f"  {config:<40} | {acc:>10.2%}   | {drop_str:>10}\n")
            f.write("\n")
        
        if use_l1_5_model:
            f.write("【附 | L1.5 专家树 (C++ 硬规则) 训练详情】\n")
            f.write("-" * 70 + "\n")
            f.write(f"  ▶ 拟合耗时 : {l1_5_duration_ms:.2f} ms\n")
            f.write(f"  ▶ 规则准确率 : {l1_5_acc:.4%} ({l1_5_status_msg})\n\n")
            
        f.write("【叁 | Layer 1 (极速先验拦截网关) 解析】\n")
        f.write("-" * 70 + "\n")
        f.write(f"  [特征池] 严格限定为 QuerySize, ExactCandSize, CandSize, GlobalPpass 及其衍生组合。\n")
        f.write(f"  [输出映射] {L1_TARGET_MAP}\n")
        f.write(f"  ▶▶ L2 拓扑分发率 : 本次共有 {leak_count} 次 ({leak_ratio:.2%}) 触发代价惩罚进入 L2 计算。\n")
        f.write(f"                     (理论最优值: {optimal_l2_inflow_rate:.2%}，即真实需要 UNG 图的比例)\n\n")
        
        f.write(report_l1 + "\n")
        f.write("  [特征重要性权重]\n")
        f.write("  " + imp_l1.replace('\n', '\n  ') + "\n\n")
        
        f.write("【肆 | Layer 2 (全视野兜底裁判层) 解析】\n")
        f.write("-" * 70 + "\n")
        f.write(f"  [特征池] 包含 L1 全集，并追加 NumEntries, NumDescendants 及其跨界衍生组合。\n")
        f.write(f"  [输出映射] {L2_TARGET_MAP}\n")
        f.write(f"  [防幻觉策略] 已为该层强制关闭 SMOTE，防止在极端小样本上产生决策幻觉。\n\n")
        f.write(report_l2 + "\n")
        f.write("  [特征重要性权重]\n")
        f.write("  " + imp_l2.replace('\n', '\n  ') + "\n\n")
        
        f.write("【伍 | C++ code (Deployment Logic)】\n")
        f.write("-" * 70 + "\n")
        f.write("  // Step 1: 获取 O(1) 维度的 Layer 1 特征\n")
        f.write("  int l1_res = L1_Model.predict(L1_Features);\n\n")
        f.write(cpp_deployment_code)
        f.write("  else if (l1_res == 2) {\n")
        f.write("      // [L1 极速拦截]: 命中 pre-filter\n")
        f.write("      return pre-filter;\n  }\n")
        f.write("  else { \n")
        f.write("      // [L1 无法决断]: l1_res == 0 (NEED_ELS)\n")
        f.write("      // 此阶段触发代价惩罚机制，花时间计算图网络入口组 (ELS) 与图特征\n")
        f.write("      auto els_features = get_min_super_sets_debug(...);\n\n")
        f.write("      // Step 2: 获取完整的 Layer 2 特征，呼叫兜底层\n")
        f.write("      int l2_res = L2_Model.predict(L2_Features);\n")
        f.write("      \n      // 依据 L2_TARGET_MAP 执行具体的底层路由分发\n")
        f.write("      switch(l2_res) {\n")
        f.write("          case 0: return UNG-nTfalse;\n")
        f.write("          case 1: return UNG-nTtrue;\n")
        f.write("          case 2: return ACORN-gamma;\n")
        f.write("          case 3: return ACORN-improved;\n")
        f.write("          case 4: return NaviX;\n")
        f.write("          case 5: return pre-filter;\n")
        f.write("      }\n")
        f.write("  }\n")
        
    print(f"\n✅ 全部构建完成！log已同步至: {report_path}")

if __name__ == "__main__":
    main()