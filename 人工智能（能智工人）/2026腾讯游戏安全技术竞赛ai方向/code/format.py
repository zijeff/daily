import pandas as pd

# 用于对输出的答案格式化，使其符合答题卡示例

TEMPLATE_XLSX = "初赛答题卡（示例）.xlsx"
SUBMISSION_CSV = "submission.csv"

OUTPUT_CSV = "final_submission.csv"
OUTPUT_XLSX = "final_submission.xlsx"

REQUIRED_TEMPLATE_COLS = ["题目序号", "意图决策", "动作行为"]

VALID_INTENTS = {"交战", "避战"}
VALID_ACTIONS = {"开火", "放技能", "丢雷", "搜", "救援"}


def map_intent(action: str) -> str:
    if pd.isna(action):
        return ""
    action = str(action).strip()
    if action in ["搜", "救援"]:
        return "避战"
    return "交战"


def main():
    print(f"读取模板: {TEMPLATE_XLSX}")
    template_df = pd.read_excel(TEMPLATE_XLSX)

    print(f"读取提交文件: {SUBMISSION_CSV}")
    sub_df = pd.read_csv(SUBMISSION_CSV)

    if list(template_df.columns) != REQUIRED_TEMPLATE_COLS:
        raise ValueError(
            f"模板列名异常。\n当前模板列名: {list(template_df.columns)}\n"
            f"期望列名: {REQUIRED_TEMPLATE_COLS}"
        )

    if "题目序号" not in sub_df.columns:
        raise ValueError("提交文件缺少列: 题目序号")
    if "意图决策" not in sub_df.columns:
        raise ValueError("提交文件缺少列: 意图决策")
    if "动作行为" not in sub_df.columns:
        raise ValueError("提交文件缺少列: 动作行为")

    sub_df = sub_df[["题目序号", "意图决策", "动作行为"]].copy()

    intent_values = set(sub_df["意图决策"].dropna().astype(str).str.strip().unique())
    action_values = set(sub_df["动作行为"].dropna().astype(str).str.strip().unique())

    # 情况1：已经是正确格式
    if intent_values.issubset(VALID_INTENTS) and action_values.issubset(VALID_ACTIONS):
        print("检测到 submission.csv 已是正确字段语义，直接按官方模板整理。")
        final_df = sub_df.copy()

    # 情况2：两列写反了
    elif intent_values.issubset(VALID_ACTIONS) and action_values.issubset(VALID_INTENTS):
        print("检测到 submission.csv 中“意图决策”和“动作行为”写反，正在自动纠正。")
        final_df = pd.DataFrame({
            "题目序号": sub_df["题目序号"],
            "动作行为": sub_df["意图决策"].astype(str).str.strip(),
        })
        final_df["意图决策"] = final_df["动作行为"].apply(map_intent)
        final_df = final_df[["题目序号", "意图决策", "动作行为"]]

    # 情况3：只有动作行为列是具体动作，没有意图列
    elif action_values.issubset(VALID_ACTIONS):
        print("检测到“动作行为”列是具体动作，将自动生成“意图决策”。")
        final_df = sub_df.copy()
        final_df["动作行为"] = final_df["动作行为"].astype(str).str.strip()
        final_df["意图决策"] = final_df["动作行为"].apply(map_intent)
        final_df = final_df[["题目序号", "意图决策", "动作行为"]]

    else:
        raise ValueError(
            f"无法识别 submission.csv 的字段语义。\n"
            f"意图决策列唯一值示例: {sorted(list(intent_values))[:10]}\n"
            f"动作行为列唯一值示例: {sorted(list(action_values))[:10]}"
        )

    # 检查重复题号
    if final_df["题目序号"].duplicated().any():
        dup_ids = final_df.loc[final_df["题目序号"].duplicated(), "题目序号"].tolist()
        raise ValueError(f"提交文件中存在重复题目序号: {dup_ids[:10]}")

    # 按模板顺序对齐
    final_df = template_df[["题目序号"]].merge(final_df, on="题目序号", how="left")

    if final_df["意图决策"].isna().any():
        bad_ids = final_df.loc[final_df["意图决策"].isna(), "题目序号"].tolist()
        raise ValueError(f"这些题目序号缺少意图决策: {bad_ids[:20]}")

    if final_df["动作行为"].isna().any():
        bad_ids = final_df.loc[final_df["动作行为"].isna(), "题目序号"].tolist()
        raise ValueError(f"这些题目序号缺少动作行为: {bad_ids[:20]}")

    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    final_df.to_excel(OUTPUT_XLSX, index=False)

    print("格式修正完成")
    print(f"输出 CSV : {OUTPUT_CSV}")
    print(f"输出 XLSX: {OUTPUT_XLSX}")
    print("\n前5行预览：")
    print(final_df.head())


if __name__ == "__main__":
    main()