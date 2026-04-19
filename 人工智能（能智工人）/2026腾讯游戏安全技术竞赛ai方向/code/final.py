import os
import re
import ast
import json
import math
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from data_extract import *

# =========================================================
# 配置区
# =========================================================
TASK_DIR = "task"   # 你的测试txt文件夹
TRAIN_CSV = "processed_data/train_cleaned_geo.csv"
LABEL_MAPPING_PATH = "processed_data/label_mapping.json"

OUTPUT_DIR = "final_outputs"
RAW_TASK_CSV = os.path.join(OUTPUT_DIR, "task_raw_snapshot.csv")
PROCESSED_TASK_CSV = os.path.join(OUTPUT_DIR, "task_cleaned_geo.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "final_catboost_model.cbm")
SUBMIT_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

TARGET_COL = "label"
ID_COL = "样本编号"

# 示例答题卡的三列
SUBMIT_COLS = ["题目序号", "意图决策", "动作行为"]

# 最终采用的权重
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.2, 1.5]

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "random_seed": 42,
    "verbose": 50,
    "class_weights": CLASS_WEIGHTS
}

# 原始快照列，和你训练时 data.csv 保持一致
RAW_COLUMNS = [
    "样本编号", "来源文件", "决策时刻", "主玩家ID", "主玩家队伍", "主玩家角色",
    "决策日志", "决策内容", "决策参数1", "决策参数2",
    "主玩家位置X", "主玩家位置Y", "主玩家位置Z",
    "主玩家武器偏航角", "主玩家武器俯仰角",
    "主玩家速度X", "主玩家速度Y", "主玩家速度Z", "主玩家速度大小",
    "主玩家相机X坐标", "主玩家相机Y坐标", "主玩家相机Z坐标",
    "主玩家视野范围", "主玩家开镜状态", "主玩家Buff数量", "主玩家Buff列表",
    "当前已追踪玩家数量", "敌方玩家数量", "队友玩家数量",
    "敌人平均距离", "队友平均距离",
    "10米内敌人数", "20米内敌人数", "10米内队友数", "20米内队友数",
    "最近敌人ID", "最近敌人队伍", "最近敌人角色", "最近敌人距离",
    "最近敌人相对位置X", "最近敌人相对位置Y", "最近敌人相对位置Z",
    "最近敌人位置X", "最近敌人位置Y", "最近敌人位置Z",
    "最近敌人武器偏航角", "最近敌人武器俯仰角",
    "最近敌人速度X", "最近敌人速度Y", "最近敌人速度Z", "最近敌人速度大小",
    "最近敌人相机X坐标", "最近敌人相机Y坐标", "最近敌人相机Z坐标",
    "最近敌人视野范围", "最近敌人开镜状态", "最近敌人Buff数量", "最近敌人Buff列表",
    "最近队友ID", "最近队友队伍", "最近队友角色", "最近队友距离",
    "最近队友相对位置X", "最近队友相对位置Y", "最近队友相对位置Z",
    "最近队友位置X", "最近队友位置Y", "最近队友位置Z",
    "最近队友武器偏航角", "最近队友武器俯仰角",
    "最近队友速度X", "最近队友速度Y", "最近队友速度Z", "最近队友速度大小",
    "最近队友相机X坐标", "最近队友相机Y坐标", "最近队友相机Z坐标",
    "最近队友视野范围", "最近队友开镜状态", "最近队友Buff数量", "最近队友Buff列表"
]

DROP_COLUMNS_FOR_MODEL = [
    "来源文件",
    "决策日志",
    "决策参数1",
    "决策参数2",
    "主玩家ID",
    "最近敌人ID",
    "最近队友ID",
    "主玩家Buff列表",
    "最近敌人Buff列表",
    "最近队友Buff列表",
]

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


# =========================================================
# 通用函数
# =========================================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_label_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    label2id = obj["label2id"]
    id2label = {int(k): v for k, v in obj["id2label"].items()}
    return label2id, id2label


def try_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        return float(s)
    except:
        return x


def safe_divide(a, b):
    b = np.where(np.abs(b) < 1e-8, np.nan, b)
    return a / b


def angle_diff_deg(a, b):
    diff = (a - b + 180) % 360 - 180
    return np.abs(diff)


def compute_target_yaw_deg(dx, dy):
    return np.degrees(np.arctan2(dy, dx))


def normalize_scope_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "开镜":
        return 1
    if s == "关镜":
        return 0
    return np.nan


def normalize_role_value(x):
    if pd.isna(x):
        return "MISSING"
    s = str(x).strip()
    if s == "":
        return "MISSING"
    if s.replace(".", "", 1).isdigit():
        if s.endswith(".0"):
            s = s[:-2]
        return f"UNKNOWN_ROLE_{s}"
    return s


def build_missing_indicator(df: pd.DataFrame, cols: list, new_col: str):
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return df
    all_missing = df[existing_cols].isna().all(axis=1)
    df[new_col] = (~all_missing).astype(int)
    return df


def detect_categorical_columns(df, target_col=None):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col is not None and target_col in cat_cols:
        cat_cols.remove(target_col)
    return cat_cols


def map_action_behavior(decision_name: str) -> str:
    if decision_name in ["搜", "救援"]:
        return "避战"
    return "交战"


def extract_question_id_from_filename(filename: str) -> int:
    """
    从 task/xxx.txt 文件名提取题目序号。
    优先取文件名中的数字；若没有数字则按遍历顺序外部补。
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return None


# =========================================================
# 1) task/*.txt -> 原始快照 DataFrame
# =========================================================
def parse_flat_kv_text(text: str):
    """
    支持:
    key: value
    key：value
    key=value
    """
    kv = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        sep = None
        for s in ["：", ":", "="]:
            if s in line:
                sep = s
                break
        if sep is None:
            continue

        left, right = line.split(sep, 1)
        key = left.strip()
        val = right.strip()
        kv[key] = try_float(val)
    return kv


def try_parse_as_dict(text: str):
    """
    依次尝试 JSON / Python dict literal / 平铺 key-value 文本
    """
    # JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except:
        pass

    # Python literal
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except:
        pass

    # flat key-value
    kv = parse_flat_kv_text(text)
    if len(kv) > 0:
        return kv

    return None


def extract_task_snapshot_from_txt(txt_path: str, fallback_qid: int):
    """
    基于你训练集 data_extract.py 的逻辑，为 task/*.txt 提取一条静态快照。

    规则：
    1. 测试集文件格式与训练日志一致
    2. 第一位玩家视为主玩家
    3. 测试集没有“真实决策标签”，所以在整份 txt 解析完成后，
       使用“最后时刻”的玩家状态冻结快照
    4. 输出字段尽量与训练时 data.csv 保持一致
    """
    player_states = {}
    first_player_id = None
    last_event_time = None

    with open(txt_path, "rb") as f:
        for raw_line in f:
            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue

            # 记录最新时间
            event_time = safe_float(parts[0], None)
            if event_time is not None:
                last_event_time = event_time

            log_type = parts[1].strip()

            # 尝试识别第一位玩家（主玩家）
            if first_player_id is None and len(parts) > 2:
                raw_pid = parts[2].strip()
                if raw_pid.startswith("玩家"):
                    first_player_id = normalize_player_id(raw_pid)

            if log_type == "游戏开始":
                parse_game_start(parts, player_states)

                # 更稳一点：如果第一位玩家还没记录，这里再补一次
                if first_player_id is None and len(parts) > 2:
                    raw_pid = parts[2].strip()
                    if raw_pid.startswith("玩家"):
                        first_player_id = normalize_player_id(raw_pid)

            elif log_type == "玩家基础信息":
                parse_player_base_info(parts, player_states)

                if first_player_id is None and len(parts) > 2:
                    raw_pid = parts[2].strip()
                    if raw_pid.startswith("玩家"):
                        first_player_id = normalize_player_id(raw_pid)

            elif log_type == "技能生效":
                parse_skill_start(parts, player_states)

            elif log_type == "技能结束":
                parse_skill_end(parts, player_states)

            # 测试集不依赖决策日志来截帧，直接忽略
            elif "（决策）" in log_type or "(决策)" in log_type:
                continue

    if first_player_id is None:
        raise ValueError(f"文件中未能识别主玩家（第一位玩家）: {txt_path}")

    if first_player_id not in player_states:
        raise ValueError(f"主玩家 {first_player_id} 不在 player_states 中: {txt_path}")

    # 从文件名提取题目序号
    qid = extract_question_id_from_filename(txt_path)
    if qid is None:
        qid = fallback_qid

    # 测试集没有真实标签，这里构造一个“空决策”占位
    dummy_decision_info = {
        "time": last_event_time,
        "player_id": first_player_id,
        "decision_full": "",
        "decision_name": "",
        "decision_param_1": None,
        "decision_param_2": None,
    }

    row = build_snapshot_row(
        sample_id=str(qid),
        source_file=txt_path.replace("\\", "/"),
        decision_info=dummy_decision_info,
        player_states=player_states
    )

    if row is None:
        raise ValueError(f"无法为文件生成快照: {txt_path}")

    # 测试集提交更适合保留数字题号
    row["样本编号"] = qid
    row["来源文件"] = txt_path.replace("\\", "/")

    # 测试集没有真实标签，这些字段留空
    row["决策日志"] = ""
    row["决策内容"] = ""
    row["决策参数1"] = None
    row["决策参数2"] = None

    return row

def build_task_raw_csv(task_dir: str):
    txt_files = [
        os.path.join(task_dir, x)
        for x in os.listdir(task_dir)
        if x.lower().endswith(".txt")
    ]
    txt_files = sorted(txt_files)

    if len(txt_files) == 0:
        raise ValueError(f"{task_dir} 下没有找到 txt 文件")

    rows = []
    for i, txt_path in enumerate(txt_files, start=1):
        row = extract_task_snapshot_from_txt(txt_path, fallback_qid=i)
        rows.append(row)

    raw_df = pd.DataFrame(rows)

    # 保证列齐全且顺序一致
    for col in RAW_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = np.nan
    raw_df = raw_df[RAW_COLUMNS]

    return raw_df


# =========================================================
# 2) 原始快照 -> 清洗 + 几何增强
# =========================================================
def preprocess_task_df(raw_df: pd.DataFrame, train_feature_df: pd.DataFrame):
    df = raw_df.copy()

    # 缺失指示
    for new_col, cols in MISSING_GROUPS.items():
        df = build_missing_indicator(df, cols, new_col)

    # 统一开镜状态
    for col in OPEN_SCOPE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_scope_value)

    # 统一角色
    for col in ROLE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_role_value)

    # 删除训练时不用的列，但保留 样本编号
    drop_cols = [c for c in DROP_COLUMNS_FOR_MODEL if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 删除标签列（测试集不应该参与）
    for c in ["决策内容", "label"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # ============== 几何增强 ==============
    enemy_rel_cols = ["最近敌人相对位置X", "最近敌人相对位置Y", "最近敌人相对位置Z"]
    if all(c in df.columns for c in enemy_rel_cols):
        dx = pd.to_numeric(df["最近敌人相对位置X"], errors="coerce")
        dy = pd.to_numeric(df["最近敌人相对位置Y"], errors="coerce")
        dz = pd.to_numeric(df["最近敌人相对位置Z"], errors="coerce")

        df["主到敌人水平距离"] = np.sqrt(dx ** 2 + dy ** 2)
        df["主到敌人三维距离_重算"] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        df["主到敌人高度差"] = dz

        if "主玩家武器偏航角" in df.columns:
            yaw = pd.to_numeric(df["主玩家武器偏航角"], errors="coerce")
            target_yaw = compute_target_yaw_deg(dx, dy)
            df["主玩家朝向_敌人夹角"] = angle_diff_deg(yaw, target_yaw)

        if "主玩家武器俯仰角" in df.columns:
            pitch = pd.to_numeric(df["主玩家武器俯仰角"], errors="coerce")
            horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
            target_pitch = np.degrees(np.arctan2(dz, np.maximum(horizontal_dist, 1e-8)))
            df["主玩家俯仰_敌人夹角"] = np.abs(pitch - target_pitch)

    mate_rel_cols = ["最近队友相对位置X", "最近队友相对位置Y", "最近队友相对位置Z"]
    if all(c in df.columns for c in mate_rel_cols):
        dx = pd.to_numeric(df["最近队友相对位置X"], errors="coerce")
        dy = pd.to_numeric(df["最近队友相对位置Y"], errors="coerce")
        dz = pd.to_numeric(df["最近队友相对位置Z"], errors="coerce")

        df["主到队友水平距离"] = np.sqrt(dx ** 2 + dy ** 2)
        df["主到队友三维距离_重算"] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        df["主到队友高度差"] = dz

        if "主玩家武器偏航角" in df.columns:
            yaw = pd.to_numeric(df["主玩家武器偏航角"], errors="coerce")
            target_yaw = compute_target_yaw_deg(dx, dy)
            df["主玩家朝向_队友夹角"] = angle_diff_deg(yaw, target_yaw)

        if "主玩家武器俯仰角" in df.columns:
            pitch = pd.to_numeric(df["主玩家武器俯仰角"], errors="coerce")
            horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
            target_pitch = np.degrees(np.arctan2(dz, np.maximum(horizontal_dist, 1e-8)))
            df["主玩家俯仰_队友夹角"] = np.abs(pitch - target_pitch)

    if "最近敌人距离" in df.columns and "敌人平均距离" in df.columns:
        a = pd.to_numeric(df["最近敌人距离"], errors="coerce")
        b = pd.to_numeric(df["敌人平均距离"], errors="coerce")
        df["最近敌人距离_除以_敌人平均距离"] = safe_divide(a, b)

    if "最近队友距离" in df.columns and "队友平均距离" in df.columns:
        a = pd.to_numeric(df["最近队友距离"], errors="coerce")
        b = pd.to_numeric(df["队友平均距离"], errors="coerce")
        df["最近队友距离_除以_队友平均距离"] = safe_divide(a, b)

    if "10米内敌人数" in df.columns and "10米内队友数" in df.columns:
        a = pd.to_numeric(df["10米内敌人数"], errors="coerce")
        b = pd.to_numeric(df["10米内队友数"], errors="coerce")
        df["10米压力差_敌减队"] = a - b

    if "20米内敌人数" in df.columns and "20米内队友数" in df.columns:
        a = pd.to_numeric(df["20米内敌人数"], errors="coerce")
        b = pd.to_numeric(df["20米内队友数"], errors="coerce")
        df["20米压力差_敌减队"] = a - b

    if "敌方玩家数量" in df.columns and "队友玩家数量" in df.columns:
        a = pd.to_numeric(df["敌方玩家数量"], errors="coerce")
        b = pd.to_numeric(df["队友玩家数量"], errors="coerce")
        df["敌我数量差"] = a - b
        df["敌我数量比"] = safe_divide(a, b + 1)

    if "主玩家速度大小" in df.columns:
        s = pd.to_numeric(df["主玩家速度大小"], errors="coerce")
        df["主玩家是否静止"] = (s < 0.5).astype("float")

    if "最近敌人速度大小" in df.columns:
        s = pd.to_numeric(df["最近敌人速度大小"], errors="coerce")
        df["最近敌人是否静止"] = (s < 0.5).astype("float")

    if "最近队友速度大小" in df.columns:
        s = pd.to_numeric(df["最近队友速度大小"], errors="coerce")
        df["最近队友是否静止"] = (s < 0.5).astype("float")

    if "主玩家速度大小" in df.columns and "最近敌人速度大小" in df.columns:
        a = pd.to_numeric(df["主玩家速度大小"], errors="coerce")
        b = pd.to_numeric(df["最近敌人速度大小"], errors="coerce")
        df["主敌速度差"] = a - b

    if "主玩家速度大小" in df.columns and "最近队友速度大小" in df.columns:
        a = pd.to_numeric(df["主玩家速度大小"], errors="coerce")
        b = pd.to_numeric(df["最近队友速度大小"], errors="coerce")
        df["主队速度差"] = a - b

    if "主玩家视野范围" in df.columns and "最近敌人视野范围" in df.columns:
        a = pd.to_numeric(df["主玩家视野范围"], errors="coerce")
        b = pd.to_numeric(df["最近敌人视野范围"], errors="coerce")
        df["主敌视野差"] = a - b

    if "主玩家Buff数量" in df.columns and "敌方玩家数量" in df.columns:
        a = pd.to_numeric(df["主玩家Buff数量"], errors="coerce")
        b = pd.to_numeric(df["敌方玩家数量"], errors="coerce")
        df["主玩家Buff数量_x_敌方玩家数量"] = a * b

    if "主玩家Buff数量" in df.columns and "10米内敌人数" in df.columns:
        a = pd.to_numeric(df["主玩家Buff数量"], errors="coerce")
        b = pd.to_numeric(df["10米内敌人数"], errors="coerce")
        df["主玩家Buff数量_x_10米内敌人数"] = a * b

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # =============================
    # 对齐训练特征
    # =============================
    train_feature_cols = [c for c in train_feature_df.columns if c != TARGET_COL]

    # 保留样本编号，不进模型
    test_ids = df[ID_COL].copy() if ID_COL in df.columns else pd.Series(np.arange(1, len(df) + 1))
    if ID_COL in df.columns and ID_COL not in train_feature_cols:
        df = df.drop(columns=[ID_COL])

    # 补缺失列
    for col in train_feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 去掉多余列
    df = df[train_feature_cols]

    # 用训练集统计值填补
    train_X = train_feature_df[train_feature_cols].copy()

    categorical_cols = detect_categorical_columns(train_X, target_col=None)
    numeric_cols = [c for c in train_feature_cols if c not in categorical_cols]

    for col in numeric_cols:
        median_val = pd.to_numeric(train_X[col], errors="coerce").median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)

    for col in categorical_cols:
        df[col] = df[col].fillna("MISSING").astype(str)

    return test_ids, df


# =========================================================
# 3) 全量训练 + 预测 + 生成提交
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    # 读取训练集
    print(f"读取训练集: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"训练集形状: {train_df.shape}")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"训练集缺少标签列: {TARGET_COL}")

    label2id, id2label = load_label_mapping(LABEL_MAPPING_PATH)
    print("标签映射如下：")
    print(label2id)

    # 1. task/*.txt -> 原始快照 CSV
    print(f"\n开始解析 task 文件夹: {TASK_DIR}")
    task_raw_df = build_task_raw_csv(TASK_DIR)
    task_raw_df.to_csv(RAW_TASK_CSV, index=False, encoding="utf-8-sig")
    print(f"原始测试快照已保存: {RAW_TASK_CSV}")
    print(f"原始测试快照形状: {task_raw_df.shape}")

    # 2. 清洗 + 几何增强
    print("\n开始清洗并构造测试特征...")
    test_ids, task_feature_df = preprocess_task_df(task_raw_df, train_df)
    task_feature_df.to_csv(PROCESSED_TASK_CSV, index=False, encoding="utf-8-sig")
    print(f"清洗增强后的测试特征已保存: {PROCESSED_TASK_CSV}")
    print(f"测试特征形状: {task_feature_df.shape}")

    # 3. 训练最终模型
    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL].copy().astype(int)

    categorical_cols = detect_categorical_columns(X_train, target_col=None)
    cat_feature_indices = [X_train.columns.get_loc(c) for c in categorical_cols]

    print("\n开始在全部训练数据上训练最终模型...")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices
    )
    model.save_model(MODEL_PATH)
    print(f"最终模型已保存: {MODEL_PATH}")

    # 4. 预测
    print("\n开始预测测试集...")
    pred_label = model.predict(task_feature_df)
    pred_label = np.array(pred_label).reshape(-1).astype(int)

    pred_decision = pd.Series(pred_label).map(id2label)
    pred_action = pred_decision.map(map_action_behavior)

    # 题目序号优先用样本编号
    question_ids = pd.to_numeric(test_ids, errors="coerce")
    if question_ids.isna().any():
        question_ids = pd.Series(np.arange(1, len(test_ids) + 1))

    submit_df = pd.DataFrame({
        SUBMIT_COLS[0]: question_ids.astype(int),
        SUBMIT_COLS[1]: pred_decision,
        SUBMIT_COLS[2]: pred_action
    })

    submit_df.to_csv(SUBMIT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n提交文件已保存: {SUBMIT_PATH}")

    print("\n提交文件前5行：")
    print(submit_df.head())

    print("\n意图决策分布：")
    print(pred_decision.value_counts(dropna=False))

    print("\n动作行为分布：")
    print(pred_action.value_counts(dropna=False))


if __name__ == "__main__":
    main()