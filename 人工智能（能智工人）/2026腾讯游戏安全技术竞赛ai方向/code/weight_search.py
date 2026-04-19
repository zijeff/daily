import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from catboost import CatBoostClassifier


# =========================
# 配置区
# =========================
INPUT_CSV = "processed_data/train_cleaned_geo.csv"
LABEL_MAPPING_PATH = "processed_data/label_mapping.json"
OUTPUT_DIR = "manual_weight_search_outputs"

TARGET_COL = "label"
N_SPLITS = 5
RANDOM_STATE = 42

BASE_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_STATE,
    "verbose": 50,
    "early_stopping_rounds": 100
}

# 小范围搜索
# 默认假设 label 顺序:
# 0: 丢雷, 1: 开火, 2: 搜, 3: 放技能, 4: 救援
WEIGHT_CANDIDATES = {
    "W1": [1.8, 1.0, 1.0, 1.2, 1.4],
    "W2": [2.0, 1.0, 1.0, 1.2, 1.5],
    "W3": [2.2, 1.0, 1.0, 1.2, 1.5],
    "W4": [2.0, 1.0, 1.0, 1.3, 1.5],
    "W5": [2.2, 1.0, 1.0, 1.3, 1.5],
    "W6": [1.8, 1.0, 1.0, 1.1, 1.4],
}


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_label_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    label2id = obj.get("label2id", None)
    id2label = obj.get("id2label", None)

    if id2label is not None:
        id2label = {int(k): v for k, v in id2label.items()}

    return label2id, id2label


def detect_categorical_columns(df, target_col):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cat_cols if c != target_col]


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run_one_weight_setting(
    exp_name,
    class_weights,
    X,
    y,
    cat_feature_indices,
    id2label
):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros(len(y), dtype=int)
    fold_rows = []

    params = BASE_PARAMS.copy()
    params["class_weights"] = class_weights

    print("\n" + "#" * 100)
    print(f"开始实验: {exp_name}")
    print(f"class_weights = {class_weights}")
    print("#" * 100)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print("\n" + "=" * 80)
        print(f"[{exp_name}] 第 {fold} 折")
        print("=" * 80)

        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_valid, y_valid),
            use_best_model=True
        )

        pred = model.predict(X_valid)
        pred = np.array(pred).reshape(-1).astype(int)
        oof_pred[valid_idx] = pred

        acc = accuracy_score(y_valid, pred)
        macro_f1 = f1_score(y_valid, pred, average="macro")
        weighted_f1 = f1_score(y_valid, pred, average="weighted")

        fold_rows.append({
            "experiment": exp_name,
            "fold": fold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "best_iteration": model.get_best_iteration(),
        })

        print(f"[{exp_name}] Fold {fold} Accuracy   : {acc:.6f}")
        print(f"[{exp_name}] Fold {fold} Macro-F1   : {macro_f1:.6f}")
        print(f"[{exp_name}] Fold {fold} Weighted-F1: {weighted_f1:.6f}")
        print(f"[{exp_name}] Fold {fold} Best Iter  : {model.get_best_iteration()}")

    metrics_df = pd.DataFrame(fold_rows)

    mean_acc = metrics_df["accuracy"].mean()
    std_acc = metrics_df["accuracy"].std()
    mean_macro_f1 = metrics_df["macro_f1"].mean()
    std_macro_f1 = metrics_df["macro_f1"].std()
    mean_weighted_f1 = metrics_df["weighted_f1"].mean()
    std_weighted_f1 = metrics_df["weighted_f1"].std()

    label_order = sorted(id2label.keys())
    target_names = [id2label[i] for i in label_order]

    cls_report_text = classification_report(
        y,
        oof_pred,
        labels=label_order,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    cls_report_dict = classification_report(
        y,
        oof_pred,
        labels=label_order,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    summary_lines = []
    summary_lines.append(f"===== 实验: {exp_name} =====")
    summary_lines.append(f"class_weights: {class_weights}")
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
    summary_lines.append(cls_report_text)

    return {
        "summary_text": "\n".join(summary_lines),
        "metrics_df": metrics_df,
        "cls_report_dict": cls_report_dict,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "mean_weighted_f1": mean_weighted_f1,
        "std_weighted_f1": std_weighted_f1,
    }


def main():
    ensure_dir(OUTPUT_DIR)

    print(f"读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"数据形状: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到标签列: {TARGET_COL}")

    label2id, id2label = load_label_mapping(LABEL_MAPPING_PATH)
    print("标签映射如下：")
    print(label2id)

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy().astype(int)

    categorical_cols = detect_categorical_columns(df, TARGET_COL)
    cat_feature_indices = [X.columns.get_loc(c) for c in categorical_cols]

    print(f"特征数: {X.shape[1]}")
    print(f"类别特征数: {len(categorical_cols)}")
    print(f"类别列: {categorical_cols}")

    all_rows = []

    for exp_name, weights in WEIGHT_CANDIDATES.items():
        result = run_one_weight_setting(
            exp_name=exp_name,
            class_weights=weights,
            X=X,
            y=y,
            cat_feature_indices=cat_feature_indices,
            id2label=id2label
        )

        summary_path = os.path.join(OUTPUT_DIR, f"{exp_name}_summary.txt")
        metrics_path = os.path.join(OUTPUT_DIR, f"{exp_name}_fold_metrics.csv")

        save_text(summary_path, result["summary_text"])
        result["metrics_df"].to_csv(metrics_path, index=False, encoding="utf-8-sig")

        row = {
            "experiment": exp_name,
            "class_weights": str(weights),
            "accuracy_mean": result["mean_acc"],
            "accuracy_std": result["std_acc"],
            "macro_f1_mean": result["mean_macro_f1"],
            "macro_f1_std": result["std_macro_f1"],
            "weighted_f1_mean": result["mean_weighted_f1"],
            "weighted_f1_std": result["std_weighted_f1"],
        }

        # 只抽最关键类别
        for cls_name in ["丢雷", "放技能", "救援", "开火", "搜"]:
            if cls_name in result["cls_report_dict"]:
                row[f"{cls_name}_precision"] = result["cls_report_dict"][cls_name]["precision"]
                row[f"{cls_name}_recall"] = result["cls_report_dict"][cls_name]["recall"]
                row[f"{cls_name}_f1"] = result["cls_report_dict"][cls_name]["f1-score"]

        all_rows.append(row)

    compare_df = pd.DataFrame(all_rows)
    compare_df = compare_df.sort_values(
        by=["macro_f1_mean", "丢雷_f1", "放技能_f1", "accuracy_mean"],
        ascending=[False, False, False, False]
    )

    compare_path = os.path.join(OUTPUT_DIR, "manual_weight_search_comparison.csv")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("全部实验完成，结果对比如下：")
    print("=" * 100)
    print(compare_df.to_string(index=False))
    print()
    print(f"对比结果已保存: {compare_path}")


if __name__ == "__main__":
    main()