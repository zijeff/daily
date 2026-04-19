import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
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
INPUT_CSV = "processed_data/train_cleaned_geo.csv"
LABEL_MAPPING_PATH = "processed_data/label_mapping.json"
OUTPUT_DIR = "cv_model_outputs"

TARGET_COL = "label"
N_SPLITS = 5
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_STATE,
    "verbose": 50,
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

    print(f"读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"数据形状: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到标签列: {TARGET_COL}")

    label2id, id2label = load_label_mapping(LABEL_MAPPING_PATH)

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy().astype(int)

    categorical_cols = detect_categorical_columns(df, TARGET_COL)
    cat_feature_indices = [X.columns.get_loc(c) for c in categorical_cols]

    print(f"特征数: {X.shape[1]}")
    print(f"类别特征数: {len(categorical_cols)}")
    print(f"类别列: {categorical_cols}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros(len(df), dtype=int)
    oof_proba = np.zeros((len(df), y.nunique()), dtype=float)

    fold_metrics = []
    feature_importance_list = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print("\n" + "=" * 80)
        print(f"开始第 {fold} 折")
        print("=" * 80)

        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_valid.shape}")

        model = CatBoostClassifier(**CATBOOST_PARAMS)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_valid, y_valid),
            use_best_model=True
        )

        pred = model.predict(X_valid)
        pred = np.array(pred).reshape(-1).astype(int)

        proba = model.predict_proba(X_valid)

        oof_pred[valid_idx] = pred
        oof_proba[valid_idx] = proba

        acc = accuracy_score(y_valid, pred)
        macro_f1 = f1_score(y_valid, pred, average="macro")
        weighted_f1 = f1_score(y_valid, pred, average="weighted")

        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "best_iteration": model.get_best_iteration()
        })

        fold_fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.get_feature_importance(),
            "fold": fold
        })
        feature_importance_list.append(fold_fi)

        model_path = os.path.join(OUTPUT_DIR, f"catboost_fold_{fold}.cbm")
        model.save_model(model_path)

        print(f"Fold {fold} Accuracy   : {acc:.6f}")
        print(f"Fold {fold} Macro-F1   : {macro_f1:.6f}")
        print(f"Fold {fold} Weighted-F1: {weighted_f1:.6f}")
        print(f"Fold {fold} 模型已保存: {model_path}")

    # -------------------------
    # 汇总结果
    # -------------------------
    metrics_df = pd.DataFrame(fold_metrics)

    mean_acc = metrics_df["accuracy"].mean()
    std_acc = metrics_df["accuracy"].std()

    mean_macro_f1 = metrics_df["macro_f1"].mean()
    std_macro_f1 = metrics_df["macro_f1"].std()

    mean_weighted_f1 = metrics_df["weighted_f1"].mean()
    std_weighted_f1 = metrics_df["weighted_f1"].std()

    if id2label is not None:
        label_order = sorted(id2label.keys())
        target_names = [id2label[i] for i in label_order]
    else:
        label_order = sorted(np.unique(y))
        target_names = [str(i) for i in label_order]

    cls_report = classification_report(
        y,
        oof_pred,
        labels=label_order,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y, oof_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    fi_all = pd.concat(feature_importance_list, axis=0, ignore_index=True)
    fi_mean = (
        fi_all.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values(by="importance", ascending=False)
    )

    oof_df = X.copy()
    oof_df["y_true"] = y.values
    oof_df["y_pred"] = oof_pred

    if id2label is not None:
        oof_df["y_true_name"] = oof_df["y_true"].map(id2label)
        oof_df["y_pred_name"] = oof_df["y_pred"].map(id2label)

    # 每个类别概率
    for class_id in range(oof_proba.shape[1]):
        col_name = f"proba_class_{class_id}"
        if id2label is not None and class_id in id2label:
            col_name = f"proba_{id2label[class_id]}"
        oof_df[col_name] = oof_proba[:, class_id]

    # 文本摘要
    summary_lines = []
    summary_lines.append("===== 5折交叉验证 CatBoost 结果 =====")
    summary_lines.append(f"训练数据: {INPUT_CSV}")
    summary_lines.append(f"总样本数: {len(df)}")
    summary_lines.append(f"特征数: {X.shape[1]}")
    summary_lines.append(f"类别特征数: {len(categorical_cols)}")
    summary_lines.append("")
    summary_lines.append("===== 每折指标 =====")
    summary_lines.append(metrics_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("===== 平均指标 =====")
    summary_lines.append(f"Accuracy   : {mean_acc:.6f} ± {std_acc:.6f}")
    summary_lines.append(f"Macro-F1   : {mean_macro_f1:.6f} ± {std_macro_f1:.6f}")
    summary_lines.append(f"Weighted-F1: {mean_weighted_f1:.6f} ± {std_weighted_f1:.6f}")
    summary_lines.append("")
    summary_lines.append("===== OOF Classification Report =====")
    summary_lines.append(cls_report)

    summary_text = "\n".join(summary_lines)

    # 保存
    metrics_path = os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv")
    summary_path = os.path.join(OUTPUT_DIR, "cv_metrics_summary.txt")
    cm_path = os.path.join(OUTPUT_DIR, "cv_confusion_matrix.csv")
    fi_path = os.path.join(OUTPUT_DIR, "cv_feature_importance_mean.csv")
    fi_raw_path = os.path.join(OUTPUT_DIR, "cv_feature_importance_all_folds.csv")
    oof_path = os.path.join(OUTPUT_DIR, "cv_oof_predictions.csv")
    cat_cols_path = os.path.join(OUTPUT_DIR, "cv_categorical_columns_used.json")

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    save_text(summary_path, summary_text)
    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    fi_mean.to_csv(fi_path, index=False, encoding="utf-8-sig")
    fi_all.to_csv(fi_raw_path, index=False, encoding="utf-8-sig")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    save_json(cat_cols_path, categorical_cols)

    # 打印
    print("=" * 80)
    print("5折交叉验证完成")
    print("=" * 80)
    print(f"Accuracy   : {mean_acc:.6f} ± {std_acc:.6f}")
    print(f"Macro-F1   : {mean_macro_f1:.6f} ± {std_macro_f1:.6f}")
    print(f"Weighted-F1: {mean_weighted_f1:.6f} ± {std_weighted_f1:.6f}")
    print()
    print("OOF Classification Report:")
    print(cls_report)
    print()
    print("Top 20 平均特征重要性：")
    print(fi_mean.head(20))
    print()
    print(f"每折指标已保存: {metrics_path}")
    print(f"摘要已保存: {summary_path}")
    print(f"混淆矩阵已保存: {cm_path}")
    print(f"平均特征重要性已保存: {fi_path}")
    print(f"OOF预测已保存: {oof_path}")


if __name__ == "__main__":
    main()