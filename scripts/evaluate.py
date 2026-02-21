"""
XSCT Bench Dataset — 评测调用示例脚本
======================================

用法：
    python scripts/evaluate.py --model YOUR_MODEL_ID --type xsct-l --api-key YOUR_KEY

依赖：
    pip install openai httpx

支持任何兼容 OpenAI 接口的服务，包括：
- OpenRouter（推荐，统一接口覆盖几乎所有模型）
- 各家官方 API（OpenAI、Anthropic、阿里云百炼、智谱、月之暗面……）

完整评测流程：
    1. 读取测试用例
    2. 调用被测模型，获取输出
    3. 调用 AI 裁判（默认 Gemini 3 Flash），对每个维度独立打分
    4. 按权重计算最终分数
    5. 输出评测结果

评分说明见 METHODOLOGY.md
"""

import json
import argparse
import time
import re
import sys
from pathlib import Path
from openai import OpenAI


# ─── 配置 ────────────────────────────────────────────────────────────────────

JUDGE_MODEL = "google/gemini-3-flash-preview"  # 裁判模型，与 XSCT LM Arena 保持一致
PASS_THRESHOLD = 60                             # ≥60 视为通过

EVAL_PROMPT_TEMPLATE = """你是专业的 AI 评测专家。请严格按照指定的评分维度进行评分。

【任务类型】{eval_type}
【难度级别】{difficulty}

【任务描述】
{task_description}

【模型收到的完整输入】
{prompt_text}

【模型生成结果】
{output}

【评分维度】（必须且只能使用以下维度，禁止自创其他维度）
{dimensions_text}

【关键要求检查】
{requirements_text}

【评分原则】
1. 只评估原始提示词明确要求的内容
2. 核心原则：模型完成了提示词的要求就应该得高分
3. 对每个维度单独打分（0-100 分）
4. 必须引用模型输出中的具体内容作为评分依据

【评分标准】
- 90-100：完全满足要求，表现优秀
- 70-89：基本满足，有少量瑕疵
- 60-69：部分满足，存在明显问题
- 0-59：未满足核心要求

【输出格式】必须严格输出 JSON，不要有任何其他内容：
{{
  "dimension_scores": {{
    "维度名称": {{"score": 分数, "reason": "评分依据，引用具体内容"}}
  }},
  "overall_comment": "整体评价（1-2句）"
}}"""


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def load_testcases(test_type: str, level: str = "basic") -> list[dict]:
    """
    加载测试用例。

    Args:
        test_type: "xsct-l" | "xsct-vg" | "xsct-w"
        level:     "basic" | "medium" | "hard"

    Returns:
        用例列表，每条包含 id、title、dimension、messages、requirements、criteria
    """
    path = Path(__file__).parent.parent / "data" / test_type / "testcases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"用例文件不存在：{path}")

    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tc = json.loads(line)
            level_data = tc.get("levels", {}).get(level)
            if not level_data:
                continue
            cases.append({
                "id": tc["id"],
                "title": tc.get("title", ""),
                "dimension": tc.get("dimension", ""),
                "test_type": tc.get("test_type", test_type),
                "messages": level_data.get("messages", []),
                "requirements": level_data.get("requirements", []),
                "criteria": level_data.get("criteria", {}),
            })
    return cases


def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    """调用被测模型，返回输出文本。"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def format_messages(messages: list[dict]) -> str:
    """将 messages 格式化为可读文本，传给裁判。"""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System Prompt]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant - 预设上下文]\n{content}")
    return "\n\n".join(parts)


def build_dimensions_text(criteria: dict) -> str:
    """将 criteria 格式化为裁判提示词中的维度描述。"""
    lines = []
    for dim, info in criteria.items():
        weight = info.get("weight", 0)
        desc = info.get("desc", "")
        lines.append(f"- {dim}（权重 {weight}%）：{desc}")
    return "\n".join(lines)


def judge_output(
    judge_client: OpenAI,
    judge_model: str,
    testcase: dict,
    model_output: str,
    level: str,
) -> dict:
    """
    调用裁判模型对输出打分。

    Returns:
        {
            "dimension_scores": {"维度": {"score": int, "reason": str}},
            "weighted_score": float,   # 加权总分
            "passed": bool,
            "overall_comment": str,
        }
    """
    criteria = testcase["criteria"]
    dimensions_text = build_dimensions_text(criteria)
    requirements_text = "\n".join(
        f"{i+1}. {r}" for i, r in enumerate(testcase.get("requirements", []))
    )
    prompt_text = format_messages(testcase["messages"])

    prompt = EVAL_PROMPT_TEMPLATE.format(
        eval_type=testcase["test_type"],
        difficulty=level,
        task_description=testcase.get("title", ""),
        prompt_text=prompt_text,
        output=model_output,
        dimensions_text=dimensions_text,
        requirements_text=requirements_text or "（无特殊要求）",
    )

    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
        extra_body={"reasoning": {"enabled": False}},  # Gemini 3 Flash 关闭 Reasoning
    )

    raw = response.choices[0].message.content or "{}"

    # 提取 JSON（模型可能在前后加了 markdown 代码块）
    json_match = re.search(r'\{[\s\S]*\}', raw)
    result = json.loads(json_match.group(0)) if json_match else {}

    dimension_scores = result.get("dimension_scores", {})

    # 按权重计算加权总分
    total_weight = sum(criteria[d]["weight"] for d in dimension_scores if d in criteria)
    weighted_score = 0.0
    if total_weight > 0:
        for dim, info in dimension_scores.items():
            if dim in criteria:
                weighted_score += info["score"] * criteria[dim]["weight"]
        weighted_score /= total_weight

    return {
        "dimension_scores": dimension_scores,
        "weighted_score": round(weighted_score, 1),
        "passed": weighted_score >= PASS_THRESHOLD,
        "overall_comment": result.get("overall_comment", ""),
    }


def print_result(testcase: dict, model_output: str, score: dict, level: str):
    """打印单条评测结果。"""
    status = "✓ 通过" if score["passed"] else "✗ 未通过"
    print(f"\n{'─'*60}")
    print(f"[{testcase['id']}] {testcase['title']}  难度: {level}")
    print(f"维度: {testcase['dimension']}")
    print(f"综合得分: {score['weighted_score']}  {status}")
    print("── 各维度得分 ──")
    for dim, info in score["dimension_scores"].items():
        print(f"  {dim}: {info['score']}  |  {info['reason'][:60]}...")
    if score["overall_comment"]:
        print(f"── 整体评价 ──\n  {score['overall_comment']}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XSCT Bench Dataset 评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 用 OpenRouter 测 Gemini 3 Flash，评 xsct-l basic 前5条
  python scripts/evaluate.py \\
      --model google/gemini-3-flash-preview \\
      --type xsct-l \\
      --level basic \\
      --limit 5 \\
      --api-key sk-or-xxx \\
      --base-url https://openrouter.ai/api/v1

  # 指定单条用例
  python scripts/evaluate.py \\
      --model deepseek/deepseek-chat \\
      --type xsct-l \\
      --id l_hallucination_054 \\
      --api-key sk-xxx
        """
    )
    parser.add_argument("--model",    required=True, help="被测模型 ID")
    parser.add_argument("--type",     default="xsct-l", choices=["xsct-l", "xsct-vg", "xsct-w"], help="测试集类型")
    parser.add_argument("--level",    default="basic",  choices=["basic", "medium", "hard"], help="难度级别")
    parser.add_argument("--limit",    type=int, default=0, help="评测条数上限（0=全部）")
    parser.add_argument("--id",       default="",  help="只评测指定 id 的用例")
    parser.add_argument("--api-key",  required=True, help="API 密钥")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="API Base URL")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help=f"裁判模型（默认 {JUDGE_MODEL}）")
    parser.add_argument("--judge-api-key", default="", help="裁判模型 API 密钥（不填则和 --api-key 相同）")
    parser.add_argument("--output",   default="", help="结果输出 JSON 文件路径（可选）")
    args = parser.parse_args()

    # 初始化客户端
    model_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    judge_key = args.judge_api_key or args.api_key
    judge_client = OpenAI(api_key=judge_key, base_url=args.base_url)

    # 加载用例
    print(f"加载用例：{args.type} / {args.level}")
    testcases = load_testcases(args.type, args.level)
    if args.id:
        testcases = [tc for tc in testcases if tc["id"] == args.id]
    if args.limit > 0:
        testcases = testcases[:args.limit]
    print(f"共 {len(testcases)} 条用例，开始评测模型：{args.model}")

    results = []
    passed = 0

    for i, tc in enumerate(testcases, 1):
        print(f"\n[{i}/{len(testcases)}] {tc['id']} — {tc['title']}", end=" ", flush=True)

        # Step 1: 调用被测模型
        try:
            model_output = call_model(model_client, args.model, tc["messages"])
        except Exception as e:
            print(f"\n  被测模型调用失败：{e}")
            continue

        print("→ 评分中...", end=" ", flush=True)

        # Step 2: 调用裁判打分
        try:
            score = judge_output(judge_client, args.judge_model, tc, model_output, args.level)
        except Exception as e:
            print(f"\n  裁判调用失败：{e}")
            continue

        if score["passed"]:
            passed += 1

        print(f"{'✓' if score['passed'] else '✗'} {score['weighted_score']}")
        print_result(tc, model_output, score, args.level)

        results.append({
            "id": tc["id"],
            "title": tc["title"],
            "dimension": tc["dimension"],
            "level": args.level,
            "model": args.model,
            "model_output": model_output,
            "score": score,
        })

        # 避免 rate limit
        time.sleep(0.5)

    # 汇总
    total = len(results)
    avg_score = sum(r["score"]["weighted_score"] for r in results) / total if total else 0
    print(f"\n{'='*60}")
    print(f"评测完成")
    print(f"模型：{args.model}  |  测试集：{args.type} / {args.level}")
    print(f"用例数：{total}  |  通过：{passed}  |  通过率：{passed/total*100:.1f}%")
    print(f"平均分：{avg_score:.1f}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
