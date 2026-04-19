import os
import json
import numpy as np
import pandas as pd


# =========================
# 配置区
# =========================
INPUT_CSV = "data.csv"
OUTPUT_DIR = "processed_data"

TARGET_COL = "决策内容"

# 明确删除的列
DROP_COLUMNS = [
    "样本编号",
    "来源文件",
    "决策日志",      # 明显标签泄漏
    "决策参数1",     # 暂时视为高风险泄漏
    "决策参数2",     # 暂时视为高风险泄漏
    "主玩家ID",
    "最近敌人ID",
    "最近队友ID",
    "主玩家Buff列表",
    "最近敌人Buff列表",
    "最近队友Buff列表",
]

# 可能不存在，删前会自动检查
OPEN_SCOPE_COLUMNS = [
    "主玩家开镜状态",
    "最近敌人开镜状态",
    "最近队友开镜状态",
]

ROLE_COLUMNS = [
    "主玩家角色",
    "最近敌人角色",
    "最近队友角色",
]

# 用于构造缺失指示变量的字段组
MISSING_GROUPS = {
    "has_recent_enemy": [
        "最近敌人队伍", "最近敌人角色", "最近敌人距离",
        "最近敌人位置X", "最近敌人位置Y", "最近敌人位置Z"
    ],
    "has_recent_teammate": [
        "最近队友队伍", "最近队友角色", "最近队友距离",
        "最近队友位置X", "最近队友位置Y", "最近队友位置Z"
    ],
    "has_main_player_pose": [
        "主玩家位置X", "主玩家位置Y", "主玩家位置Z",
        "主玩家武器偏航角", "主玩家武器俯仰角"
    ]
}


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_scope_value(x):
    """
    统一开镜状态：
    开镜 -> 1
    关镜 -> 0
    其他异常值 -> NaN
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s == "开镜":
        return 1
    if s == "关镜":
        return 0

    # 一些异常值先视为缺失
    # 例如 0.0 / -0.1 / 0.1 / -0.0 等
    return np.nan


def normalize_role_value(x):
    """
    统一角色字段为字符串。
    对空值填成 MISSING。
    对纯数字型角色编码保留为 UNKNOWN_ROLE_xxx，避免直接混淆到正常角色名。
    """
    if pd.isna(x):
        return "MISSING"

    s = str(x).strip()
    if s == "":
        return "MISSING"

    # 如果是纯数字，例如 30008
    if s.replace(".", "", 1).isdigit():
        # 防止出现 30008.0 这种情况
        if s.endswith(".0"):
            s = s[:-2]
        return f"UNKNOWN_ROLE_{s}"

    return s


def build_missing_indicator(df: pd.DataFrame, cols: list, new_col: str):
    """
    若这一组字段全都缺失，则认为该实体不存在，标记为 0；否则为 1
    """
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return df

    all_missing = df[existing_cols].isna().all(axis=1)
    df[new_col] = (~all_missing).astype(int)
    return df


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_report(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    print(f"读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"原始数据形状: {df.shape}")

    report_lines = []
    report_lines.append("===== 数据预处理报告 =====")
    report_lines.append(f"原始形状: {df.shape}")

    # -------------------------
    # 1. 检查标签列
    # -------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到标签列: {TARGET_COL}")

    # 删除标签缺失样本
    before_rows = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    after_rows = len(df)
    report_lines.append(f"删除标签缺失样本数: {before_rows - after_rows}")

    # -------------------------
    # 2. 构造缺失指示变量
    # -------------------------
    for new_col, cols in MISSING_GROUPS.items():
        df = build_missing_indicator(df, cols, new_col)
        report_lines.append(f"新增缺失指示列: {new_col}")

    # -------------------------
    # 3. 统一开镜状态
    # -------------------------
    for col in OPEN_SCOPE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_scope_value)
            report_lines.append(f"已标准化开镜状态列: {col}")

    # -------------------------
    # 4. 统一角色字段
    # -------------------------
    for col in ROLE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_role_value)
            report_lines.append(f"已标准化角色列: {col}")

    # -------------------------
    # 5. 删除高风险列
    # -------------------------
    actual_drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=actual_drop_cols)
    report_lines.append(f"删除列数: {len(actual_drop_cols)}")
    report_lines.append("删除列如下:")
    report_lines.extend(actual_drop_cols)

    # -------------------------
    # 6. 标签编码
    # -------------------------
    label_values = sorted(df[TARGET_COL].astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(label_values)}
    id2label = {idx: label for label, idx in label2id.items()}

    df["label"] = df[TARGET_COL].astype(str).map(label2id)
    report_lines.append("标签映射:")
    for k, v in label2id.items():
        report_lines.append(f"{k} -> {v}")

    # 删除原始标签列，训练时直接用 label
    df = df.drop(columns=[TARGET_COL])

    # -------------------------
    # 7. 区分类别列 / 数值列
    # -------------------------
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c != "label"]

    report_lines.append(f"类别特征数: {len(categorical_cols)}")
    report_lines.append(f"数值特征数: {len(numeric_cols)}")

    # -------------------------
    # 8. 缺失填补
    # -------------------------
    fill_values = {
        "numeric": {},
        "categorical": {}
    }

    # 数值列：中位数填补
    for col in numeric_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)
        fill_values["numeric"][col] = float(median_val)

    # 类别列：统一填 MISSING
    for col in categorical_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        fill_values["categorical"][col] = "MISSING"

    # -------------------------
    # 9. 再做一次简单清理
    # -------------------------
    # 去掉单一值列
    nunique_map = df.nunique(dropna=False)
    constant_cols = [c for c in df.columns if c != "label" and nunique_map[c] <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        report_lines.append(f"删除单一值列数: {len(constant_cols)}")
        report_lines.append("单一值列如下:")
        report_lines.extend(constant_cols)
    else:
        report_lines.append("未发现单一值列")

    # -------------------------
    # 10. 输出结果
    # -------------------------
    feature_cols = [c for c in df.columns if c != "label"]

    output_csv = os.path.join(OUTPUT_DIR, "train_cleaned.csv")
    output_feature_json = os.path.join(OUTPUT_DIR, "feature_columns.json")
    output_label_json = os.path.join(OUTPUT_DIR, "label_mapping.json")
    output_fill_json = os.path.join(OUTPUT_DIR, "fill_values.json")
    output_report = os.path.join(OUTPUT_DIR, "preprocess_report.txt")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    save_json(feature_cols, output_feature_json)
    save_json(
        {
            "label2id": label2id,
            "id2label": id2label
        },
        output_label_json
    )
    save_json(fill_values, output_fill_json)

    report_lines.append(f"处理后形状: {df.shape}")
    report_lines.append(f"最终特征数: {len(feature_cols)}")
    report_lines.append(f"输出文件: {output_csv}")
    report_lines.append(f"输出文件: {output_feature_json}")
    report_lines.append(f"输出文件: {output_label_json}")
    report_lines.append(f"输出文件: {output_fill_json}")

    write_report(output_report, report_lines)

    print(f"处理后数据形状: {df.shape}")
    print(f"最终特征数: {len(feature_cols)}")
    print(f"已保存到目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()