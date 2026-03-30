import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

try:
    import onnxmltools
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False

# ==========================================
# 1. 全局配置区域
# ==========================================
DATASET_LIST = ["Amazon","BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"]
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"
PERCENTAGE_THRESHOLD = 0.35    # 性能差异阈值

FINAL_FEATURES = ['QuerySize', 'CandSize', 'TrieTotalNodes']

# 引入竞技场的模型候选者
MODELS_TO_TRY = ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]

# ==========================================
# 2. 核心功能与特征工厂
# ==========================================

def create_classifier(model_type="RandomForest", **kwargs):
    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params.update(kwargs)
        return RandomForestClassifier(**params)
        
    elif model_type == "LightGBM":
        import lightgbm as lgb
        params = {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 150, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced', 'verbose': -1}
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)
        
    elif model_type == "XGBoost":
        import xgboost as xgb
        params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 150, 'random_state': 42, 'n_jobs': -1}
        params.update(kwargs)
        return xgb.XGBClassifier(**params)
        
    elif model_type == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        params = {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
        params.update(kwargs)
        return DecisionTreeClassifier(**params)
        
    else:
        raise ValueError(f"不支持的模型引擎: {model_type}")

def load_and_label_els_data(csv_path, threshold=0.2):
    df = pd.read_csv(csv_path)
    col_f = 'MinSupersetT_ms_UNG-nTfalse'
    col_t = 'MinSupersetT_ms_UNG-nTtrue'
    
    for feat in [col_f, col_t] + FINAL_FEATURES:
        if feat not in df.columns:
            raise ValueError(f"宽表中缺失必需列: {feat}")
            
    # 修复数据骤降: 对 Trie 静态全局特征进行智能填充
    if 'TrieTotalNodes' in df.columns:
        valid_val = df['TrieTotalNodes'].dropna().iloc[0] if not df['TrieTotalNodes'].dropna().empty else 0
        df['TrieTotalNodes'].fillna(valid_val, inplace=True)
            
    df = df.dropna(subset=[col_f, col_t] + FINAL_FEATURES).copy()
    
    df['time_diff_abs'] = np.abs(df[col_t] - df[col_f])
    df['min_time'] = np.minimum(df[col_t], df[col_f])
    df['time_diff_percent'] = df['time_diff_abs'] / (df['min_time'] + 1e-9)
    
    df_clean = df[df['time_diff_percent'] > threshold].copy()
    y = (df_clean[col_t] < df_clean[col_f]).astype(int)
    
    return df_clean, y

def generate_els_features(df):
    features = df[FINAL_FEATURES].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    return features

def train_and_evaluate_model(X, y, model_type):
    class_counts = y.value_counts()
    if len(class_counts) > 1 and class_counts.min() >= 2:
        stratify_param = y
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
    
    classifier = create_classifier(model_type=model_type)
    
    t0 = time.perf_counter()
    classifier.fit(X_train.values, y_train.values)
    train_time_ms = (time.perf_counter() - t0) * 1000.0
    
    t1 = time.perf_counter()
    y_pred = classifier.predict(X_test.values)
    pred_latency_us = ((time.perf_counter() - t1) * 1e6) / len(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['nTfalse (0)', 'nTtrue (1)'], zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=['True_nTfalse(0)', 'True_nTtrue(1)'], columns=['Pred_nTfalse(0)', 'Pred_nTtrue(1)'])
    
    if hasattr(classifier, 'feature_importances_'):
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': classifier.feature_importances_})
        importance_str = imp_df.sort_values(by='Importance', ascending=False).to_string(index=False)
    else:
        importance_str = "当前模型不支持直接提取特征重要性。"
    
    return {
        "model_type": model_type,
        "classifier": classifier,
        "acc": acc,
        "train_time_ms": train_time_ms,
        "pred_latency_us": pred_latency_us,
        "report": report,
        "cm_df": cm_df,
        "importance_str": importance_str
    }

def save_onnx_model(classifier, model_type, num_features, output_dir, filename="idea1_selector_model_final.onnx"):
    onnx_filename = os.path.join(output_dir, filename)
    try:
        if model_type in ["RandomForest", "DecisionTree"] and SKL2ONNX_AVAILABLE:
            from skl2onnx import convert_sklearn
            initial_type = [('float_input', FloatTensorType([None, num_features]))]
            onnx_model = convert_sklearn(classifier, initial_types=initial_type, target_opset=15)
        elif model_type in ["LightGBM", "XGBoost"] and ONNXMLTOOLS_AVAILABLE:
            from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
            initial_type = [('float_input', OnnxFloatTensorType([None, num_features]))]
            if model_type == "LightGBM":
                import onnxmltools
                onnx_model = onnxmltools.convert_lightgbm(classifier, initial_types=initial_type, target_opset=15)
            else:
                import onnxmltools
                onnx_model = onnxmltools.convert_xgboost(classifier, initial_types=initial_type, target_opset=15)
        else:
            print(f"⚠️ 无法导出 {model_type} 的 ONNX 模型。")
            return
            
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ ONNX 模型成功导出: {onnx_filename}")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型时发生错误: {e}")

# ==========================================
# 3. 主干业务流
# ==========================================

def process_dataset(dataset_name):
    print(f"\n{'='*70}")
    print(f"🚀 开始训练极简 ELS Router | 数据集: {dataset_name}")
    print(f"{'='*70}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots_UNGnTtrue", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "intelElS") 
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到宽表文件: {csv_path}，跳过。")
        return
        
    try:
        df_clean, y = load_and_label_els_data(csv_path, threshold=PERCENTAGE_THRESHOLD)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
        
    print(f"[*] 数据过滤 (Threshold > {PERCENTAGE_THRESHOLD*100}% 差异):")
    print(f"  - 过滤后剩余有效样本数: {len(df_clean)}")
    counts = y.value_counts()
    print(f"  - Label 0 (nTfalse 更快): {counts.get(0, 0)} 条")
    print(f"  - Label 1 (nTtrue 更快) : {counts.get(1, 0)} 条")
    
    if len(df_clean) < 10:
        print("❌ 有效样本过少，停止训练。")
        return

    X_features = generate_els_features(df_clean)
    
    print(f"\n[模型竞技场启动] 使用特征: {FINAL_FEATURES}")
    results_cache = {}
    metrics_list = []
    
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate_model(X_features, y, model_type)
            results_cache[model_type] = res
            metrics_list.append(res)
            print(f"  > {model_type:<15} | 准确率: {res['acc']:.4%} | 训练耗时: {res['train_time_ms']:.2f} ms")
        except ImportError:
            print(f"  > ⚠️ 缺少 {model_type} 引擎依赖库。")
        except Exception as e:
            print(f"  > ⚠️ {model_type} 训练失败: {e}")
            
    if not metrics_list:
        print("❌ 所有模型训练失败。")
        return
        
    # 选出 Accuracy 最高的模型
    best_result = max(metrics_list, key=lambda x: x['acc'])
    best_model_name = best_result['model_type']
    best_model = best_result['classifier']
    
    print(f"\n🏆 [策略选定] ELS 胜出模型: {best_model_name} (Acc: {best_result['acc']:.4%})")
    
    print("\n" + "*"*60)
    print(f"🔥 C++ 重点对齐: C++ 端 calculate_idea1_features 必须严格按此顺序返回:")
    for idx, feat in enumerate(X_features.columns):
        print(f"   [{idx+1}] {feat}")
    print("*"*60 + "\n")
    
    # 统一模型持久化导出 (Joblib + ONNX)
    joblib_path = os.path.join(output_dir, "idea1_selector_model_final.joblib")
    joblib.dump(best_model, joblib_path)
    save_onnx_model(best_model, best_model_name, X_features.shape[1], output_dir, "idea1_selector_model_final.onnx")
        
    # 生成带有大比武表格的综合战报
    report_path = os.path.join(output_dir, f"ELS_Router_Report_{dataset_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃                ELS Router                                          ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 数据集  : {dataset_name:<15} 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} ┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
        
        f.write("【零 | 模型竞技】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'算法模型(Model)':<16} | {'准确率(Accuracy)':<16} | {'训练耗时(Train ms)':<18} | {'单次推理(Pred μs)':<18}\n")
        f.write("  " + "-" * 76 + "\n")
        sorted_metrics = sorted(metrics_list, key=lambda x: x["acc"], reverse=True)
        for m in sorted_metrics:
            f.write(f"  {m['model_type']:<16} | {m['acc']:<18.4%} | {m['train_time_ms']:<18.2f} | {m['pred_latency_us']:<18.2f}\n")
        f.write(f"\n  🏆 最终部署选定模型 : {best_model_name}\n\n")

        f.write("【壹 | 数据集与标签分布】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  ▶ 过滤阈值 (MinSupersetT_ms 相对差异) : > {PERCENTAGE_THRESHOLD*100}%\n")
        f.write(f"  ▶ 保留有效样本数 : {len(df_clean)}\n")
        f.write(f"  ▶ Label 0 (nTfalse 胜出) : {counts.get(0, 0)}\n")
        f.write(f"  ▶ Label 1 (nTtrue 胜出)  : {counts.get(1, 0)}\n\n")
        
        f.write("【贰 | 特征列表 (严格对齐顺序)】\n")
        f.write("-" * 80 + "\n")
        f.write("  " + " | ".join(X_features.columns) + "\n\n")
        
        f.write("【叁 | 胜出模型透视】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  ▶ 测试集准确率 : {best_result['acc']:.4%}\n")
        f.write(f"  ▶ 拟合耗时     : {best_result['train_time_ms']:.2f} ms\n\n")
        f.write("  [分类报告]\n")
        f.write("  " + best_result['report'].replace('\n', '\n  ') + "\n\n")
        f.write("  [混淆矩阵]\n")
        f.write("  " + best_result['cm_df'].to_string().replace('\n', '\n  ') + "\n\n")
        f.write("  [特征重要性]\n")
        f.write("  " + best_result['importance_str'].replace('\n', '\n  ') + "\n")

if __name__ == "__main__":
    for dataset in DATASET_LIST:
        process_dataset(dataset)