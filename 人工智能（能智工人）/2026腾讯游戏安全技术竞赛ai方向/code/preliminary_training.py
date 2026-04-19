import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from catboost import CatBoostClassifier


# =========================
# 配置区
# =========================
INPUT_CSV = "processed_data/train_cleaned.csv"
LABEL_MAPPING_PATH = "processed_data/label_mapping.json"
OUTPUT_DIR = "model_outputs"

TARGET_COL = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_STATE,
    "verbose": 50,                  # 只保留 CatBoost 训练日志
    "early_stopping_rounds": 100,
    "auto_class_weights": "Balanced"
}


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_label_mapping(path):
    if not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    label2id = obj.get("label2id", None)
    id2label = obj.get("id2label", None)

    if id2label is not None:
        id2label = {int(k): v for k, v in id2label.items()}

    return label2id, id2label


def detect_categorical_columns(df, target_col):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col]
    return cat_cols


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    print(f"读取训练数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"数据形状: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到标签列: {TARGET_COL}")

    label2id, id2label = load_label_mapping(LABEL_MAPPING_PATH)

    # 特征 / 标签
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # 自动识别类别列
    categorical_cols = detect_categorical_columns(df, TARGET_COL)
    cat_feature_indices = [X.columns.get_loc(c) for c in categorical_cols]

    print(f"特征数: {X.shape[1]}")
    print(f"类别特征数: {len(categorical_cols)}")
    print(f"类别特征列: {categorical_cols}")

    # 分层划分
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_valid.shape}")

    # 构建模型
    model = CatBoostClassifier(**CATBOOST_PARAMS)

    # 训练
    print("\n开始训练 CatBoost ...")
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )
    print("训练完成")

    # 预测
    print("\n开始验证集预测与评估 ...")
    y_pred = model.predict(X_valid)
    y_pred = np.array(y_pred).reshape(-1).astype(int)

    # 指标
    acc = accuracy_score(y_valid, y_pred)
    macro_f1 = f1_score(y_valid, y_pred, average="macro")
    weighted_f1 = f1_score(y_valid, y_pred, average="weighted")

    if id2label is not None:
        label_order = sorted(id2label.keys())
        target_names = [id2label[i] for i in label_order]
    else:
        label_order = sorted(np.unique(y))
        target_names = [str(i) for i in label_order]

    cls_report = classification_report(
        y_valid,
        y_pred,
        labels=label_order,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_valid, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    # 特征重要性
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance()
    }).sort_values(by="importance", ascending=False)

    # 验证集预测结果
    pred_df = X_valid.copy()
    pred_df["y_true"] = y_valid.values
    pred_df["y_pred"] = y_pred

    if id2label is not None:
        pred_df["y_true_name"] = pred_df["y_true"].map(id2label)
        pred_df["y_pred_name"] = pred_df["y_pred"].map(id2label)

    # 输出摘要
    summary_lines = []
    summary_lines.append("===== CatBoost 多分类训练结果 =====")
    summary_lines.append(f"训练数据: {INPUT_CSV}")
    summary_lines.append(f"总样本数: {len(df)}")
    summary_lines.append(f"特征数: {X.shape[1]}")
    summary_lines.append(f"类别特征数: {len(categorical_cols)}")
    summary_lines.append(f"训练集大小: {X_train.shape}")
    summary_lines.append(f"验证集大小: {X_valid.shape}")
    summary_lines.append("")
    summary_lines.append(f"Accuracy: {acc:.6f}")
    summary_lines.append(f"Macro-F1: {macro_f1:.6f}")
    summary_lines.append(f"Weighted-F1: {weighted_f1:.6f}")
    summary_lines.append("")
    summary_lines.append("===== Classification Report =====")
    summary_lines.append(cls_report)

    summary_text = "\n".join(summary_lines)

    # 保存结果
    model_path = os.path.join(OUTPUT_DIR, "catboost_multiclass_model.cbm")
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.txt")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    pred_path = os.path.join(OUTPUT_DIR, "valid_predictions.csv")
    used_cat_cols_path = os.path.join(OUTPUT_DIR, "categorical_columns_used.json")

    model.save_model(model_path)
    save_text(summary_path, summary_text)
    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    feature_importance.to_csv(fi_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    save_json(used_cat_cols_path, categorical_cols)

    # 控制台打印
    print("\n" + "=" * 80)
    print("训练完成，验证集结果如下：")
    print("=" * 80)
    print(f"Accuracy   : {acc:.6f}")
    print(f"Macro-F1   : {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")
    print()
    print("Classification Report:")
    print(cls_report)
    print()
    print("Top 20 特征重要性：")
    print(feature_importance.head(20))
    print()
    print(f"模型已保存: {model_path}")
    print(f"指标摘要已保存: {summary_path}")
    print(f"混淆矩阵已保存: {cm_path}")
    print(f"特征重要性已保存: {fi_path}")
    print(f"验证集预测结果已保存: {pred_path}")


if __name__ == "__main__":
    main()