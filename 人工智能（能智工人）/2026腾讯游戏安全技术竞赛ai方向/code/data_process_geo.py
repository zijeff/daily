import os
import json
import math
import numpy as np
import pandas as pd

# 在之前的数据处理上提取了一部分几何特征
INPUT_CSV = "processed_data/train_cleaned.csv"
OUTPUT_CSV = "processed_data/train_cleaned_geo.csv"
REPORT_PATH = "processed_data/geo_feature_report.txt"


def safe_divide(a, b):
    b = np.where(np.abs(b) < 1e-8, np.nan, b)
    return a / b


def angle_diff_deg(a, b):
    """
    计算两个角度的最小差值，结果范围 [0, 180]
    """
    diff = (a - b + 180) % 360 - 180
    return np.abs(diff)


def compute_target_yaw_deg(dx, dy):
    """
    用相对平面向量计算目标方向角（单位：度）
    atan2(dy, dx) -> [-180, 180]
    """
    return np.degrees(np.arctan2(dy, dx))


def main():
    print(f"读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"原始形状: {df.shape}")

    report_lines = []
    report_lines.append("===== 几何关系特征增强报告 =====")
    report_lines.append(f"输入文件: {INPUT_CSV}")
    report_lines.append(f"输入形状: {df.shape}")

    # -------------------------
    # 1. 主玩家 -> 最近敌人 几何特征
    # -------------------------
    enemy_rel_cols = ["最近敌人相对位置X", "最近敌人相对位置Y", "最近敌人相对位置Z"]
    if all(c in df.columns for c in enemy_rel_cols):
        dx = df["最近敌人相对位置X"]
        dy = df["最近敌人相对位置Y"]
        dz = df["最近敌人相对位置Z"]

        df["主到敌人水平距离"] = np.sqrt(dx ** 2 + dy ** 2)
        df["主到敌人三维距离_重算"] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        df["主到敌人高度差"] = dz

        target_yaw = compute_target_yaw_deg(dx, dy)
        if "主玩家武器偏航角" in df.columns:
            df["主玩家朝向_敌人夹角"] = angle_diff_deg(df["主玩家武器偏航角"], target_yaw)

        if "主玩家武器俯仰角" in df.columns:
            horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
            target_pitch = np.degrees(np.arctan2(dz, np.maximum(horizontal_dist, 1e-8)))
            df["主玩家俯仰_敌人夹角"] = np.abs(df["主玩家武器俯仰角"] - target_pitch)

        report_lines.append("已生成：主玩家 -> 最近敌人 几何特征")

    # -------------------------
    # 2. 主玩家 -> 最近队友 几何特征
    # -------------------------
    mate_rel_cols = ["最近队友相对位置X", "最近队友相对位置Y", "最近队友相对位置Z"]
    if all(c in df.columns for c in mate_rel_cols):
        dx = df["最近队友相对位置X"]
        dy = df["最近队友相对位置Y"]
        dz = df["最近队友相对位置Z"]

        df["主到队友水平距离"] = np.sqrt(dx ** 2 + dy ** 2)
        df["主到队友三维距离_重算"] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        df["主到队友高度差"] = dz

        target_yaw = compute_target_yaw_deg(dx, dy)
        if "主玩家武器偏航角" in df.columns:
            df["主玩家朝向_队友夹角"] = angle_diff_deg(df["主玩家武器偏航角"], target_yaw)

        if "主玩家武器俯仰角" in df.columns:
            horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
            target_pitch = np.degrees(np.arctan2(dz, np.maximum(horizontal_dist, 1e-8)))
            df["主玩家俯仰_队友夹角"] = np.abs(df["主玩家武器俯仰角"] - target_pitch)

        report_lines.append("已生成：主玩家 -> 最近队友 几何特征")

    # -------------------------
    # 3. 距离比率特征
    # -------------------------
    if "最近敌人距离" in df.columns and "敌人平均距离" in df.columns:
        df["最近敌人距离_除以_敌人平均距离"] = safe_divide(
            df["最近敌人距离"], df["敌人平均距离"]
        )
        report_lines.append("已生成：最近敌人距离比")

    if "最近队友距离" in df.columns and "队友平均距离" in df.columns:
        df["最近队友距离_除以_队友平均距离"] = safe_divide(
            df["最近队友距离"], df["队友平均距离"]
        )
        report_lines.append("已生成：最近队友距离比")

    # -------------------------
    # 4. 局部压力特征
    # -------------------------
    if "10米内敌人数" in df.columns and "10米内队友数" in df.columns:
        df["10米压力差_敌减队"] = df["10米内敌人数"] - df["10米内队友数"]
        report_lines.append("已生成：10米压力差")

    if "20米内敌人数" in df.columns and "20米内队友数" in df.columns:
        df["20米压力差_敌减队"] = df["20米内敌人数"] - df["20米内队友数"]
        report_lines.append("已生成：20米压力差")

    if "敌方玩家数量" in df.columns and "队友玩家数量" in df.columns:
        df["敌我数量差"] = df["敌方玩家数量"] - df["队友玩家数量"]
        df["敌我数量比"] = safe_divide(df["敌方玩家数量"], df["队友玩家数量"] + 1)
        report_lines.append("已生成：敌我数量差/比")

    # -------------------------
    # 5. 速度关系特征
    # -------------------------
    if "主玩家速度大小" in df.columns:
        df["主玩家是否静止"] = (df["主玩家速度大小"] < 0.5).astype(int)
        report_lines.append("已生成：主玩家是否静止")

    if "最近敌人速度大小" in df.columns:
        df["最近敌人是否静止"] = (df["最近敌人速度大小"] < 0.5).astype(int)
        report_lines.append("已生成：最近敌人是否静止")

    if "最近队友速度大小" in df.columns:
        df["最近队友是否静止"] = (df["最近队友速度大小"] < 0.5).astype(int)
        report_lines.append("已生成：最近队友是否静止")

    if "主玩家速度大小" in df.columns and "最近敌人速度大小" in df.columns:
        df["主敌速度差"] = df["主玩家速度大小"] - df["最近敌人速度大小"]
        report_lines.append("已生成：主敌速度差")

    if "主玩家速度大小" in df.columns and "最近队友速度大小" in df.columns:
        df["主队速度差"] = df["主玩家速度大小"] - df["最近队友速度大小"]
        report_lines.append("已生成：主队速度差")

    # -------------------------
    # 6. 覆盖关系特征
    # -------------------------
    if "主玩家视野范围" in df.columns and "最近敌人视野范围" in df.columns:
        df["主敌视野差"] = df["主玩家视野范围"] - df["最近敌人视野范围"]
        report_lines.append("已生成：主敌视野差")

    if "主玩家Buff数量" in df.columns and "敌方玩家数量" in df.columns:
        df["主玩家Buff数量_x_敌方玩家数量"] = df["主玩家Buff数量"] * df["敌方玩家数量"]
        report_lines.append("已生成：Buff数量×敌方玩家数")

    if "主玩家Buff数量" in df.columns and "10米内敌人数" in df.columns:
        df["主玩家Buff数量_x_10米内敌人数"] = df["主玩家Buff数量"] * df["10米内敌人数"]
        report_lines.append("已生成：Buff数量×近敌人数")

    # -------------------------
    # 7. 无穷值和异常值处理
    # -------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    report_lines.append(f"输出形状: {df.shape}")
    report_lines.append(f"新增特征后总列数: {df.shape[1]}")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"增强后形状: {df.shape}")
    print(f"已保存: {OUTPUT_CSV}")
    print(f"报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()