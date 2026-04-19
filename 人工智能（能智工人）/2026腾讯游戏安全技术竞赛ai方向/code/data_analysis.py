import os
import json
import numpy as np
import pandas as pd


# =========================
# 配置区
# =========================
CSV_PATH = "data.csv"              # 你的数据文件路径
OUTPUT_DIR = "eda_outputs"         # 输出目录
TARGET_COL = None                  # 如果有标签列，填列名，例如 "label"，没有就保持 None

# 判定阈值
MISSING_THRESHOLDS = [0.3, 0.5, 0.8]   # 高缺失阈值
HIGH_CARDINALITY_THRESHOLD = 50        # 高基数类别列阈值
NEAR_CONSTANT_THRESHOLD = 0.95         # 单一值占比超过该阈值，视为近常数列
TOP_N_CATEGORIES = 10                  # 类别列展示前 N 个高频类别
NUMERIC_UNIQUE_AS_CATEGORY_THRESHOLD = 20  # 数值列若唯一值很少，可提示可能是类别型


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def detect_semantic_type(series: pd.Series) -> str:
    """
    推断更符合建模视角的字段类型：
    - numeric
    - categorical
    - boolean
    - datetime
    - text
    """
    s = series.dropna()
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if pd.api.types.is_numeric_dtype(dtype):
        unique_count = s.nunique(dropna=True)
        if unique_count <= 2:
            return "boolean_or_binary"
        if unique_count <= NUMERIC_UNIQUE_AS_CATEGORY_THRESHOLD:
            return "numeric_low_cardinality"
        return "numeric"

    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        if len(s) == 0:
            return "categorical_or_text"

        unique_count = s.astype(str).nunique(dropna=True)
        avg_len = s.astype(str).map(len).mean()

        if unique_count <= HIGH_CARDINALITY_THRESHOLD and avg_len < 30:
            return "categorical"
        if unique_count <= 5:
            return "categorical"
        return "text"

    return "unknown"


def infer_dirty_string_stats(series: pd.Series) -> dict:
    """
    检查字符串列中的常见脏值
    """
    s = series.dropna().astype(str).str.strip()
    lower_s = s.str.lower()

    dirty_tokens = ["", "none", "null", "nan", "na", "unknown", "undefined"]
    stats = {}
    for token in dirty_tokens:
        stats[token if token != "" else "<empty_string>"] = int((lower_s == token).sum())
    return stats


def compute_outlier_ratio_iqr(series: pd.Series) -> float:
    """
    用 IQR 粗略估计异常值比例
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return np.nan

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_ratio = ((s < lower) | (s > upper)).mean()
    return float(outlier_ratio)


# =========================
# 主分析函数
# =========================
def analyze_dataframe(df: pd.DataFrame, target_col: str = None):
    results = {}
    n_rows, n_cols = df.shape

    # ===== 表级信息 =====
    duplicated_rows = int(df.duplicated().sum())
    mem_mb = memory_usage_mb(df)

    bad_colnames = []
    seen_cols = set()
    duplicate_colnames = []

    for c in df.columns:
        if c in seen_cols:
            duplicate_colnames.append(c)
        seen_cols.add(c)

        if str(c).strip() != str(c) or " " in str(c) or any(ch in str(c) for ch in ["/", "\\", ":", ";"]):
            bad_colnames.append(c)

    # ===== 列级分析 =====
    column_records = []
    numeric_records = []
    categorical_records = []
    top_values_details = {}

    constant_cols = []
    near_constant_cols = []
    high_missing_cols = {thr: [] for thr in MISSING_THRESHOLDS}
    high_cardinality_cols = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_ratio = float(s.isna().mean())
        non_null_count = int(s.notna().sum())
        nunique_dropna = int(s.nunique(dropna=True))
        raw_dtype = str(s.dtype)
        semantic_type = detect_semantic_type(s)

        # 最频繁值占比
        if non_null_count > 0:
            top_value_counts = s.value_counts(dropna=True)
            top_freq = int(top_value_counts.iloc[0]) if len(top_value_counts) > 0 else 0
            top_ratio = float(top_freq / non_null_count) if non_null_count > 0 else np.nan
            top_value = top_value_counts.index[0] if len(top_value_counts) > 0 else None
        else:
            top_freq = 0
            top_ratio = np.nan
            top_value = None

        # 常数列 / 近常数列
        if nunique_dropna <= 1:
            constant_cols.append(col)
        elif not np.isnan(top_ratio) and top_ratio >= NEAR_CONSTANT_THRESHOLD:
            near_constant_cols.append(col)

        # 高缺失列
        for thr in MISSING_THRESHOLDS:
            if missing_ratio >= thr:
                high_missing_cols[thr].append(col)

        # 列汇总
        column_records.append({
            "column": col,
            "raw_dtype": raw_dtype,
            "semantic_type": semantic_type,
            "rows": n_rows,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 6),
            "nunique_dropna": nunique_dropna,
            "top_value": str(top_value) if top_value is not None else None,
            "top_freq": top_freq,
            "top_ratio_among_non_null": round(top_ratio, 6) if not np.isnan(top_ratio) else np.nan,
        })

        # ===== 数值列分析 =====
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            desc = s_num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

            zero_ratio = float((s_num.fillna(0) == 0).mean()) if len(s_num) > 0 else np.nan
            negative_ratio = float((s_num < 0).mean()) if s_num.notna().sum() > 0 else np.nan
            skewness = float(s_num.skew()) if s_num.notna().sum() > 2 else np.nan
            outlier_ratio = compute_outlier_ratio_iqr(s_num)

            numeric_records.append({
                "column": col,
                "count": float(desc.get("count", np.nan)),
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "min": float(desc.get("min", np.nan)),
                "1%": float(desc.get("1%", np.nan)),
                "5%": float(desc.get("5%", np.nan)),
                "25%": float(desc.get("25%", np.nan)),
                "50%": float(desc.get("50%", np.nan)),
                "75%": float(desc.get("75%", np.nan)),
                "95%": float(desc.get("95%", np.nan)),
                "99%": float(desc.get("99%", np.nan)),
                "max": float(desc.get("max", np.nan)),
                "skewness": skewness,
                "zero_ratio": zero_ratio,
                "negative_ratio": negative_ratio,
                "outlier_ratio_iqr": outlier_ratio,
                "possible_categorical_hint": s_num.nunique(dropna=True) <= NUMERIC_UNIQUE_AS_CATEGORY_THRESHOLD
            })

        # ===== 类别列 / 字符列分析 =====
        if semantic_type in ["categorical", "text", "categorical_or_text"] or \
           pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):

            vc = s.astype(str).fillna("NaN_PLACEHOLDER").value_counts(dropna=False).head(TOP_N_CATEGORIES)
            top_categories = {str(k): int(v) for k, v in vc.items()}
            dirty_stats = infer_dirty_string_stats(s)

            if nunique_dropna >= HIGH_CARDINALITY_THRESHOLD:
                high_cardinality_cols.append(col)

            categorical_records.append({
                "column": col,
                "semantic_type": semantic_type,
                "nunique_dropna": nunique_dropna,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 6),
                "top_1_value": list(top_categories.keys())[0] if len(top_categories) > 0 else None,
                "top_1_freq": list(top_categories.values())[0] if len(top_categories) > 0 else None,
                "high_cardinality": nunique_dropna >= HIGH_CARDINALITY_THRESHOLD,
                "dirty_empty_string_count": dirty_stats.get("<empty_string>", 0),
                "dirty_none_count": dirty_stats.get("none", 0),
                "dirty_null_count": dirty_stats.get("null", 0),
                "dirty_nan_count": dirty_stats.get("nan", 0),
                "dirty_na_count": dirty_stats.get("na", 0),
                "dirty_unknown_count": dirty_stats.get("unknown", 0),
                "dirty_undefined_count": dirty_stats.get("undefined", 0),
            })

            top_values_details[col] = {
                "top_categories": top_categories,
                "dirty_value_stats": dirty_stats
            }

    # ===== 标签分布 =====
    target_distribution_df = None
    if target_col is not None and target_col in df.columns:
        target_distribution_df = (
            df[target_col]
            .value_counts(dropna=False)
            .rename_axis("target_value")
            .reset_index(name="count")
        )
        target_distribution_df["ratio"] = target_distribution_df["count"] / len(df)

    # ===== 汇总结果 =====
    results["overview"] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_usage_mb": round(mem_mb, 4),
        "duplicated_rows": duplicated_rows,
        "duplicate_column_names": duplicate_colnames,
        "abnormal_column_names": bad_colnames,
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "high_missing_cols": high_missing_cols,
        "high_cardinality_cols": high_cardinality_cols,
    }

    results["column_summary"] = pd.DataFrame(column_records).sort_values(
        by=["missing_ratio", "nunique_dropna"], ascending=[False, False]
    )
    results["numeric_summary"] = pd.DataFrame(numeric_records)
    results["categorical_summary"] = pd.DataFrame(categorical_records).sort_values(
        by=["nunique_dropna", "missing_ratio"], ascending=[False, False]
    )
    results["target_distribution"] = target_distribution_df
    results["top_values_details"] = top_values_details

    return results


def save_results(results: dict, output_dir: str):
    ensure_dir(output_dir)

    # 保存 overview.txt
    overview_path = os.path.join(output_dir, "overview.txt")
    with open(overview_path, "w", encoding="utf-8") as f:
        ov = results["overview"]
        f.write("===== DATA OVERVIEW =====\n")
        f.write(f"Rows: {ov['n_rows']}\n")
        f.write(f"Columns: {ov['n_cols']}\n")
        f.write(f"Memory Usage (MB): {ov['memory_usage_mb']}\n")
        f.write(f"Duplicated Rows: {ov['duplicated_rows']}\n\n")

        f.write("Duplicate Column Names:\n")
        f.write(json.dumps(ov["duplicate_column_names"], ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("Abnormal Column Names:\n")
        f.write(json.dumps(ov["abnormal_column_names"], ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("Constant Columns:\n")
        f.write(json.dumps(ov["constant_cols"], ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("Near Constant Columns:\n")
        f.write(json.dumps(ov["near_constant_cols"], ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("High Missing Columns:\n")
        f.write(json.dumps(ov["high_missing_cols"], ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("High Cardinality Columns:\n")
        f.write(json.dumps(ov["high_cardinality_cols"], ensure_ascii=False, indent=2))
        f.write("\n")

    # 保存 csv
    results["column_summary"].to_csv(
        os.path.join(output_dir, "column_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    results["numeric_summary"].to_csv(
        os.path.join(output_dir, "numeric_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    results["categorical_summary"].to_csv(
        os.path.join(output_dir, "categorical_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if results["target_distribution"] is not None:
        results["target_distribution"].to_csv(
            os.path.join(output_dir, "target_distribution.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # 保存类别细节
    with open(os.path.join(output_dir, "top_values_details.json"), "w", encoding="utf-8") as f:
        json.dump(results["top_values_details"], f, ensure_ascii=False, indent=2)


def print_brief_report(results: dict):
    ov = results["overview"]
    col_df = results["column_summary"]
    num_df = results["numeric_summary"]
    cat_df = results["categorical_summary"]

    print("=" * 80)
    print("数据概览")
    print("=" * 80)
    print(f"样本数: {ov['n_rows']}")
    print(f"特征数: {ov['n_cols']}")
    print(f"内存占用: {ov['memory_usage_mb']} MB")
    print(f"重复行数: {ov['duplicated_rows']}")

    print("=" * 80)
    print("缺失值最多的前10列")
    print("=" * 80)
    print(col_df[["column", "missing_count", "missing_ratio", "raw_dtype", "semantic_type"]].head(10))

    print("=" * 80)
    print("常数列")
    print("=" * 80)
    print(ov["constant_cols"][:30], "..." if len(ov["constant_cols"]) > 30 else "")
    print(f"共 {len(ov['constant_cols'])} 列")

    print("=" * 80)
    print("近常数列")
    print("=" * 80)
    print(ov["near_constant_cols"][:30], "..." if len(ov["near_constant_cols"]) > 30 else "")
    print(f"共 {len(ov['near_constant_cols'])} 列")

    print("=" * 80)
    print("高基数类别列")
    print("=" * 80)
    print(ov["high_cardinality_cols"][:30], "..." if len(ov["high_cardinality_cols"]) > 30 else "")
    print(f"共 {len(ov['high_cardinality_cols'])} 列")

    print("=" * 80)
    print("数值列数量 / 类别列数量")
    print("=" * 80)
    print(f"数值列: {len(num_df)}")
    print(f"类别列: {len(cat_df)}")

    if results["target_distribution"] is not None:
        print("=" * 80)
        print("标签分布")
        print("=" * 80)
        print(results["target_distribution"])


def main():
    ensure_dir(OUTPUT_DIR)

    print(f"正在读取数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print("数据读取完成")
    print()

    results = analyze_dataframe(df, target_col=TARGET_COL)
    save_results(results, OUTPUT_DIR)
    print_brief_report(results)

    print(f"分析结果已保存到目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()