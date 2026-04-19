import csv
import math
import tarfile
from typing import Dict, List, Optional, Set, Any

from tqdm import tqdm


# =========================
# 可配置项
# =========================

INPUT_TAR_GZ = r"./train_data/初赛训练数据.tar.gz"
OUTPUT_CSV = r"./data.csv"

# None 表示处理全部；如果想调试可设为 100、1000 等
MAX_FILES = None


# =========================
# 工具函数
# =========================

def safe_float(value: str, default: Optional[float] = None) -> Optional[float]:
    """安全转浮点"""
    try:
        return float(value)
    except Exception:
        return default


def euclidean_distance(
    x1: Optional[float], y1: Optional[float], z1: Optional[float],
    x2: Optional[float], y2: Optional[float], z2: Optional[float]
) -> Optional[float]:
    """计算三维欧氏距离"""
    values = [x1, y1, z1, x2, y2, z2]
    if any(v is None for v in values):
        return None
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def calc_speed_norm(vx: Optional[float], vy: Optional[float], vz: Optional[float]) -> Optional[float]:
    """计算速度大小"""
    values = [vx, vy, vz]
    if any(v is None for v in values):
        return None
    return math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def normalize_player_id(raw: str) -> str:
    """将“玩家2982”标准化成“2982”"""
    raw = raw.strip()
    if raw.startswith("玩家"):
        return raw.replace("玩家", "", 1)
    return raw


def buff_set_to_text(buff_set: Set[str]) -> str:
    """将 buff 集合转成字符串"""
    if not buff_set:
        return ""
    return "；".join(sorted(buff_set))


# =========================
# 玩家状态
# =========================

def new_player_state() -> Dict[str, Any]:
    """初始化玩家状态"""
    return {
        "player_id": "",
        "team_id": "",
        "role": "",
        "pos_x": None,
        "pos_y": None,
        "pos_z": None,
        "weapon_yaw": None,
        "weapon_pitch": None,
        "vel_x": None,
        "vel_y": None,
        "vel_z": None,
        "speed_norm": None,
        "camera_x": None,
        "camera_y": None,
        "camera_z": None,
        "fov": None,
        "scope_state": "",
        "buffs": set(),
        "last_update_time": None,
    }


# =========================
# 日志解析
# =========================

def parse_game_start(parts: List[str], player_states: Dict[str, Dict[str, Any]]) -> None:
    """解析游戏开始"""
    if len(parts) < 5:
        return

    player_id = normalize_player_id(parts[2])
    team_id = parts[3].strip()
    role = parts[4].strip()

    if player_id not in player_states:
        player_states[player_id] = new_player_state()

    player_states[player_id]["player_id"] = player_id
    player_states[player_id]["team_id"] = team_id
    player_states[player_id]["role"] = role


def parse_player_base_info(parts: List[str], player_states: Dict[str, Dict[str, Any]]) -> None:
    """解析玩家基础信息"""
    if len(parts) < 17:
        return

    event_time = safe_float(parts[0], None)
    player_id = normalize_player_id(parts[2])

    if player_id not in player_states:
        player_states[player_id] = new_player_state()
        player_states[player_id]["player_id"] = player_id

    state = player_states[player_id]

    state["pos_x"] = safe_float(parts[3], state["pos_x"])
    state["pos_y"] = safe_float(parts[4], state["pos_y"])
    state["pos_z"] = safe_float(parts[5], state["pos_z"])

    state["weapon_yaw"] = safe_float(parts[6], state["weapon_yaw"])
    state["weapon_pitch"] = safe_float(parts[7], state["weapon_pitch"])

    state["vel_x"] = safe_float(parts[8], state["vel_x"])
    state["vel_y"] = safe_float(parts[9], state["vel_y"])
    state["vel_z"] = safe_float(parts[10], state["vel_z"])
    state["speed_norm"] = calc_speed_norm(state["vel_x"], state["vel_y"], state["vel_z"])

    state["camera_x"] = safe_float(parts[11], state["camera_x"])
    state["camera_y"] = safe_float(parts[12], state["camera_y"])
    state["camera_z"] = safe_float(parts[13], state["camera_z"])
    state["fov"] = safe_float(parts[14], state["fov"])

    state["scope_state"] = parts[16].strip()
    state["last_update_time"] = event_time


def parse_skill_start(parts: List[str], player_states: Dict[str, Dict[str, Any]]) -> None:
    """解析技能生效"""
    if len(parts) < 4:
        return

    player_id = None
    for item in parts:
        if item.startswith("玩家"):
            player_id = normalize_player_id(item)
            break

    if player_id is None:
        return

    if player_id not in player_states:
        player_states[player_id] = new_player_state()
        player_states[player_id]["player_id"] = player_id

    skill_name = parts[-1].strip()
    if skill_name:
        player_states[player_id]["buffs"].add(skill_name)


def parse_skill_end(parts: List[str], player_states: Dict[str, Dict[str, Any]]) -> None:
    """解析技能结束"""
    if len(parts) < 4:
        return

    player_id = None
    for item in parts:
        if item.startswith("玩家"):
            player_id = normalize_player_id(item)
            break

    if player_id is None or player_id not in player_states:
        return

    skill_name = parts[-1].strip()
    if skill_name and skill_name in player_states[player_id]["buffs"]:
        player_states[player_id]["buffs"].remove(skill_name)


def parse_decision(parts: List[str]) -> Optional[Dict[str, Any]]:
    """解析决策日志"""
    if len(parts) < 3:
        return None

    log_type = parts[1].strip()
    if "（决策）" not in log_type and "(决策)" not in log_type:
        return None

    event_time = safe_float(parts[0], None)
    player_id = normalize_player_id(parts[2])

    decision_name = (
        log_type.replace("（决策）", "")
        .replace("(决策)", "")
        .strip()
    )

    decision_param_1 = safe_float(parts[3], None) if len(parts) > 3 else None
    decision_param_2 = safe_float(parts[4], None) if len(parts) > 4 else None

    return {
        "time": event_time,
        "player_id": player_id,
        "decision_full": log_type,
        "decision_name": decision_name,
        "decision_param_1": decision_param_1,
        "decision_param_2": decision_param_2,
    }


# =========================
# 输出字段
# =========================

def build_empty_detail_fields(prefix_cn: str) -> Dict[str, Any]:
    """构造最近敌人/最近队友空字段"""
    return {
        f"{prefix_cn}ID": "",
        f"{prefix_cn}队伍": "",
        f"{prefix_cn}角色": "",
        f"{prefix_cn}距离": None,
        f"{prefix_cn}相对位置X": None,
        f"{prefix_cn}相对位置Y": None,
        f"{prefix_cn}相对位置Z": None,
        f"{prefix_cn}位置X": None,
        f"{prefix_cn}位置Y": None,
        f"{prefix_cn}位置Z": None,
        f"{prefix_cn}武器偏航角": None,
        f"{prefix_cn}武器俯仰角": None,
        f"{prefix_cn}速度X": None,
        f"{prefix_cn}速度Y": None,
        f"{prefix_cn}速度Z": None,
        f"{prefix_cn}速度大小": None,
        f"{prefix_cn}相机X坐标": None,
        f"{prefix_cn}相机Y坐标": None,
        f"{prefix_cn}相机Z坐标": None,
        f"{prefix_cn}视野范围": None,
        f"{prefix_cn}开镜状态": "",
        f"{prefix_cn}Buff数量": 0,
        f"{prefix_cn}Buff列表": "",
    }


def fill_detail_fields(prefix_cn: str, target_state: Optional[Dict[str, Any]], main_state: Dict[str, Any]) -> Dict[str, Any]:
    """填充最近敌人/最近队友详细字段"""
    result = build_empty_detail_fields(prefix_cn)
    if not target_state:
        return result

    rel_x = None if main_state["pos_x"] is None or target_state["pos_x"] is None else target_state["pos_x"] - main_state["pos_x"]
    rel_y = None if main_state["pos_y"] is None or target_state["pos_y"] is None else target_state["pos_y"] - main_state["pos_y"]
    rel_z = None if main_state["pos_z"] is None or target_state["pos_z"] is None else target_state["pos_z"] - main_state["pos_z"]

    dist = euclidean_distance(
        main_state["pos_x"], main_state["pos_y"], main_state["pos_z"],
        target_state["pos_x"], target_state["pos_y"], target_state["pos_z"]
    )

    result.update({
        f"{prefix_cn}ID": target_state.get("player_id", ""),
        f"{prefix_cn}队伍": target_state.get("team_id", ""),
        f"{prefix_cn}角色": target_state.get("role", ""),
        f"{prefix_cn}距离": dist,
        f"{prefix_cn}相对位置X": rel_x,
        f"{prefix_cn}相对位置Y": rel_y,
        f"{prefix_cn}相对位置Z": rel_z,
        f"{prefix_cn}位置X": target_state.get("pos_x"),
        f"{prefix_cn}位置Y": target_state.get("pos_y"),
        f"{prefix_cn}位置Z": target_state.get("pos_z"),
        f"{prefix_cn}武器偏航角": target_state.get("weapon_yaw"),
        f"{prefix_cn}武器俯仰角": target_state.get("weapon_pitch"),
        f"{prefix_cn}速度X": target_state.get("vel_x"),
        f"{prefix_cn}速度Y": target_state.get("vel_y"),
        f"{prefix_cn}速度Z": target_state.get("vel_z"),
        f"{prefix_cn}速度大小": target_state.get("speed_norm"),
        f"{prefix_cn}相机X坐标": target_state.get("camera_x"),
        f"{prefix_cn}相机Y坐标": target_state.get("camera_y"),
        f"{prefix_cn}相机Z坐标": target_state.get("camera_z"),
        f"{prefix_cn}视野范围": target_state.get("fov"),
        f"{prefix_cn}开镜状态": target_state.get("scope_state", ""),
        f"{prefix_cn}Buff数量": len(target_state.get("buffs", set())),
        f"{prefix_cn}Buff列表": buff_set_to_text(target_state.get("buffs", set())),
    })
    return result


def build_snapshot_row(
    sample_id: str,
    source_file: str,
    decision_info: Dict[str, Any],
    player_states: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """冻结静态快照并生成一行输出"""
    main_player_id = decision_info["player_id"]
    if main_player_id not in player_states:
        return None

    main_state = player_states[main_player_id]
    main_team = main_state.get("team_id", "")

    enemy_players = []
    teammate_players = []

    for pid, state in player_states.items():
        if pid == main_player_id:
            continue

        dist = euclidean_distance(
            main_state["pos_x"], main_state["pos_y"], main_state["pos_z"],
            state["pos_x"], state["pos_y"], state["pos_z"]
        )
        if dist is None:
            continue

        item = {"player_id": pid, "distance": dist, "state": state}
        if state.get("team_id", "") == main_team:
            teammate_players.append(item)
        else:
            enemy_players.append(item)

    enemy_players.sort(key=lambda x: x["distance"])
    teammate_players.sort(key=lambda x: x["distance"])

    nearest_enemy = enemy_players[0]["state"] if enemy_players else None
    nearest_teammate = teammate_players[0]["state"] if teammate_players else None

    enemy_distances = [x["distance"] for x in enemy_players]
    teammate_distances = [x["distance"] for x in teammate_players]

    def mean_or_none(values: List[float]) -> Optional[float]:
        return None if not values else sum(values) / len(values)

    row = {
        "样本编号": sample_id,
        "来源文件": source_file,
        "决策时刻": decision_info["time"],
        "主玩家ID": main_player_id,
        "主玩家队伍": main_state.get("team_id", ""),
        "主玩家角色": main_state.get("role", ""),
        "决策日志": decision_info["decision_full"],
        "决策内容": decision_info["decision_name"],
        "决策参数1": decision_info["decision_param_1"],
        "决策参数2": decision_info["decision_param_2"],

        "主玩家位置X": main_state.get("pos_x"),
        "主玩家位置Y": main_state.get("pos_y"),
        "主玩家位置Z": main_state.get("pos_z"),
        "主玩家武器偏航角": main_state.get("weapon_yaw"),
        "主玩家武器俯仰角": main_state.get("weapon_pitch"),
        "主玩家速度X": main_state.get("vel_x"),
        "主玩家速度Y": main_state.get("vel_y"),
        "主玩家速度Z": main_state.get("vel_z"),
        "主玩家速度大小": main_state.get("speed_norm"),
        "主玩家相机X坐标": main_state.get("camera_x"),
        "主玩家相机Y坐标": main_state.get("camera_y"),
        "主玩家相机Z坐标": main_state.get("camera_z"),
        "主玩家视野范围": main_state.get("fov"),
        "主玩家开镜状态": main_state.get("scope_state", ""),
        "主玩家Buff数量": len(main_state.get("buffs", set())),
        "主玩家Buff列表": buff_set_to_text(main_state.get("buffs", set())),

        "当前已追踪玩家数量": len(player_states),
        "敌方玩家数量": len(enemy_players),
        "队友玩家数量": len(teammate_players),
        "敌人平均距离": mean_or_none(enemy_distances),
        "队友平均距离": mean_or_none(teammate_distances),
        "10米内敌人数": sum(1 for d in enemy_distances if d <= 10),
        "20米内敌人数": sum(1 for d in enemy_distances if d <= 20),
        "10米内队友数": sum(1 for d in teammate_distances if d <= 10),
        "20米内队友数": sum(1 for d in teammate_distances if d <= 20),
    }

    row.update(fill_detail_fields("最近敌人", nearest_enemy, main_state))
    row.update(fill_detail_fields("最近队友", nearest_teammate, main_state))
    return row


# =========================
# 单个 txt 字节流处理
# =========================

def process_txt_stream(file_obj, member_name: str, sample_index: int) -> Optional[Dict[str, Any]]:
    """处理单个 txt 的字节流"""
    player_states: Dict[str, Dict[str, Any]] = {}
    sample_id = f"{sample_index:06d}"

    for raw_line in file_obj:
        try:
            line = raw_line.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue

        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        log_type = parts[1].strip()

        if log_type == "游戏开始":
            parse_game_start(parts, player_states)

        elif log_type == "玩家基础信息":
            parse_player_base_info(parts, player_states)

        elif log_type == "技能生效":
            parse_skill_start(parts, player_states)

        elif log_type == "技能结束":
            parse_skill_end(parts, player_states)

        elif "（决策）" in log_type or "(决策)" in log_type:
            decision_info = parse_decision(parts)
            if decision_info is not None:
                return build_snapshot_row(
                    sample_id=sample_id,
                    source_file=member_name,
                    decision_info=decision_info,
                    player_states=player_states
                )

    return None


# =========================
# 全量流式处理并直接写 CSV
# =========================

def process_tar_stream_to_csv(tar_gz_path: str, output_csv: str, max_files: Optional[int] = None) -> None:
    """
    流式扫描 tar.gz，遇到 txt 就直接处理；
    每生成一条样本，立刻写入 CSV。
    """
    processed_count = 0
    written_count = 0
    writer = None

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as fout:
        with tarfile.open(tar_gz_path, "r|gz") as tar:
            for member in tqdm(tar, desc="流式扫描压缩包", unit="member"):
                if not member.isfile():
                    continue

                member_name = member.name.replace("\\", "/").strip()
                if not member_name.lower().endswith(".txt"):
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    continue

                processed_count += 1
                row = process_txt_stream(extracted, member_name, processed_count)

                if row is None:
                    continue

                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(row)
                written_count += 1

                if written_count % 1000 == 0:
                    fout.flush()

                if max_files is not None and written_count >= max_files:
                    break

    print(f"处理完成。")
    print(f"成功写入样本数: {written_count}")
    print(f"输出文件: {output_csv}")


def main():
    process_tar_stream_to_csv(INPUT_TAR_GZ, OUTPUT_CSV, MAX_FILES)


if __name__ == "__main__":
    main()