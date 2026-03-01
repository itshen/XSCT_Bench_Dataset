"""
XSCT Bench Dataset — 评测脚本
======================================

用法：
    python scripts/evaluate.py --model YOUR_MODEL --type xsct-l --api-key YOUR_KEY

依赖：
    pip install openai

本脚本的评测逻辑与 XSCT LM Arena (xsct.ai) 平台保持一致：
- 评分提示词模板对齐平台版本
- 权重计算方式相同（rubric 分段细则 + 加权求和）
- 裁判模型：google/gemini-3-flash-preview（关闭 Reasoning）
- 通过阈值：≥ 60 分

完整方法论见 METHODOLOGY.md
"""

import json
import argparse
import time
import re
import sys
from pathlib import Path
from openai import OpenAI


# ─── 常量 ─────────────────────────────────────────────────────────────────────

JUDGE_MODEL      = "google/gemini-3-flash-preview"
PASS_THRESHOLD   = 60
OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

DIFFICULTY_ZH = {"basic": "基础", "medium": "进阶", "hard": "困难"}

# ─── 评分提示词模板（与平台 EVAL_PROMPT_TEMPLATE_V2 对齐）───────────────────

EVAL_PROMPT_TEMPLATE = """你是专业的 AI 评测专家。请严格按照指定的评分维度进行评分。

【任务类型】{eval_type}
【难度级别】{difficulty}

【任务描述】
{task_description}

【原始提示词】
{prompt}
{reference_section}
【模型生成结果】
{output}

【评分维度】（必须且只能使用以下维度，禁止自创其他维度）
{dimensions_text}

【关键要求检查】
请根据原始提示词的要求逐项检查：
{requirements_text}

【评分原则】
1. 只评估原始提示词明确要求的内容，不要评估提示词未要求的方面
2. 如果提示词没有要求"光影效果"，就不要因为光影扣分
3. 如果提示词没有要求"构图美感"，就不要因为构图扣分
4. 核心原则：模型完成了提示词的要求就应该得高分

【评分规则】
1. 对每个维度单独打分（0-100 分）
2. 90-100分：完全满足提示词要求，表现优秀
3. 70-89分：基本满足要求，有少量瑕疵
4. 60-69分：部分满足要求，存在明显问题
5. 0-59分：未能满足提示词的核心要求

【输出格式】严格按以下 JSON 格式输出：
{output_example}

【重要警告】
- 你必须且只能输出上述指定的维度，禁止自创"提示词遵循"、"写实质量"、"光影合理性"、"构图美感"等其他维度
- dimension_scores 的 key 必须与【评分维度】中列出的维度名完全一致
- 不要输出 total_score，总分由系统自动计算
"""


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def load_testcases(test_type: str, level: str = "basic") -> list[dict]:
    """
    加载测试用例，返回解析好的列表。
    每条包含：id, title, dimension, test_type, messages, requirements, criteria
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
                "id":           tc["id"],
                "title":        tc.get("title", ""),
                "description":  tc.get("description", ""),
                "dimension":    tc.get("dimension", ""),
                "test_type":    tc.get("test_type", test_type),
                "messages":     level_data.get("messages", []),
                "requirements": level_data.get("requirements", []),
                "criteria":     level_data.get("criteria", {}),
                # 参考答案（部分用例有）
                "reference_answer": level_data.get("reference_answer", []),
            })
    return cases


def format_messages(messages: list[dict]) -> str:
    """将 messages 列表格式化为可读文本，传给裁判。"""
    parts = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant (预设)]\n{content}")
    return "\n\n".join(parts)


def build_dimensions_text(criteria: dict) -> str:
    """
    将 criteria 格式化为评分维度文本。
    完整输出 rubric 分段细则，与平台 format_criteria_for_eval_prompt 对齐。
    """
    lines = []
    for name, data in criteria.items():
        if not isinstance(data, dict):
            continue
        weight = data.get("weight", 0)
        desc   = data.get("desc", "")
        rubric = data.get("rubric", {})

        lines.append(f"\n**{name}（权重 {weight}%）**：{desc}")

        if isinstance(rubric, dict):
            for score_range, detail in rubric.items():
                lines.append(f"  - {score_range}分：{detail}")
        elif isinstance(rubric, list):
            for r in rubric:
                lines.append(f"  - {r}")

    return "\n".join(lines)


def build_requirements_text(requirements: list) -> str:
    if not requirements:
        return "（请根据原始提示词的要求进行检查）"
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(requirements))


def build_reference_section(reference_answer) -> str:
    """构建参考答案区段（与平台 _build_reference_answers_section 对齐）。"""
    if isinstance(reference_answer, str):
        answers = [reference_answer] if reference_answer.strip() else []
    elif isinstance(reference_answer, list):
        answers = [a for a in reference_answer if a and str(a).strip()]
    else:
        answers = []

    if not answers:
        return ""
    if len(answers) == 1:
        return f"\n【参考答案】\n{answers[0]}\n\n"

    section = f"\n【参考答案】（共 {len(answers)} 套，模型输出需完整符合其中一套，不得混用）\n\n"
    for i, ans in enumerate(answers, 1):
        section += f"--- 参考答案 {i} ---\n{ans}\n\n"
    return section


def build_eval_prompt(tc: dict, model_output: str, level: str) -> str:
    """构建完整的评分提示词。"""
    criteria   = tc["criteria"]
    dim_names  = list(criteria.keys())

    # 评分维度文本（含 rubric）
    dimensions_text = build_dimensions_text(criteria)

    # 若 criteria 为空，使用兜底维度
    if not dimensions_text.strip():
        dimensions_text = (
            "\n**准确性（权重 35%）**：是否正确理解并完成任务"
            "\n**质量（权重 35%）**：生成内容的整体质量"
            "\n**清晰度（权重 30%）**：表达是否清晰易懂"
        )
        dim_names = ["准确性", "质量", "清晰度"]

    # 输出示例（使用真实维度名，防止 AI 自创）
    scores_example = ",\n    ".join(
        f'"{name}": {{"score": 85, "reason": "评分理由..."}}'
        for name in dim_names
    )
    output_example = (
        '{\n'
        '  "dimension_scores": {\n'
        f'    {scores_example}\n'
        '  },\n'
        '  "overall_comment": "整体评价（1-2句）"\n'
        '}'
    )

    # 评测类型中文名
    dimension_type_map = {
        "L-": "文字生成", "P-": "图像生成", "VG-": "图像生成", "W-": "网页生成",
    }
    eval_type = "通用文本"
    for prefix, label in dimension_type_map.items():
        if tc["dimension"].startswith(prefix):
            eval_type = label
            break

    return EVAL_PROMPT_TEMPLATE.format(
        eval_type       = eval_type,
        difficulty      = DIFFICULTY_ZH.get(level, level),
        task_description= tc["description"],
        prompt          = format_messages(tc["messages"]),
        reference_section = build_reference_section(tc.get("reference_answer", [])),
        output          = model_output,
        dimensions_text = dimensions_text,
        requirements_text = build_requirements_text(tc["requirements"]),
        output_example  = output_example,
    )


def parse_judge_response(raw: str) -> dict:
    """解析裁判返回的 JSON，容错处理 markdown 代码块。"""
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {}


def calc_weighted_score(dimension_scores: dict, criteria: dict) -> float:
    """
    按权重计算加权总分（与平台 parse_dimension_scores 对齐）。
    支持模糊维度名匹配（大小写不敏感、子串匹配）。
    """
    if not dimension_scores:
        return 50.0

    def extract_score(dim_data) -> float:
        if isinstance(dim_data, dict):
            raw = dim_data.get("score", 0)
            if isinstance(raw, dict):
                raw = raw.get("value", raw.get("score", 0))
        else:
            raw = dim_data
        try:
            return float(raw or 0)
        except (TypeError, ValueError):
            return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for dim_name, dim_data in dimension_scores.items():
        score = extract_score(dim_data)
        # 模糊匹配权重
        weight = 0
        if criteria:
            for c_name, c_val in criteria.items():
                w = c_val.get("weight", 0) if isinstance(c_val, dict) else 0
                if (c_name.lower() == dim_name.lower()
                        or c_name in dim_name or dim_name in c_name):
                    weight = w
                    break
        if weight == 0:
            weight = 100 / len(dimension_scores)
        weighted_sum += score * weight
        total_weight += weight

    return round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0


# ─── 核心调用 ─────────────────────────────────────────────────────────────────

def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    """调用被测模型，返回输出文本。"""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return resp.choices[0].message.content or ""


def judge(judge_client: OpenAI, judge_model: str, eval_prompt: str) -> dict:
    """
    调用裁判模型打分。
    关闭 Reasoning（与平台一致），temperature=0 保证确定性。
    """
    resp = judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个专业的 AI 评测专家，负责评估大模型的生成质量。"
                    "请根据给定的评分维度和标准进行客观评分。"
                    "在所有文字描述中，引号请使用「」而非""，以确保 JSON 格式正确。"
                ),
            },
            {"role": "user", "content": eval_prompt},
        ],
        temperature=0.0,
        extra_body={"reasoning": {"enabled": False}},
    )
    return parse_judge_response(resp.choices[0].message.content or "")


def evaluate_one(
    model_client: OpenAI,
    judge_client: OpenAI,
    model: str,
    judge_model: str,
    tc: dict,
    level: str,
) -> dict | None:
    """评测单条用例，返回结果 dict，失败返回 None。"""
    # Step 1: 被测模型生成
    try:
        model_output = call_model(model_client, model, tc["messages"])
    except Exception as e:
        print(f"\n  ✗ 被测模型调用失败：{e}")
        return None

    print("→ 评分中...", end=" ", flush=True)

    # Step 2: 裁判打分
    try:
        eval_prompt = build_eval_prompt(tc, model_output, level)
        result      = judge(judge_client, judge_model, eval_prompt)
    except Exception as e:
        print(f"\n  ✗ 裁判调用失败：{e}")
        return None

    dimension_scores = result.get("dimension_scores", {})
    weighted_score   = calc_weighted_score(dimension_scores, tc["criteria"])
    passed           = weighted_score >= PASS_THRESHOLD

    return {
        "id":               tc["id"],
        "title":            tc["title"],
        "dimension":        tc["dimension"],
        "level":            level,
        "model":            model,
        "model_output":     model_output,
        "dimension_scores": dimension_scores,
        "weighted_score":   weighted_score,
        "passed":           passed,
        "overall_comment":  result.get("overall_comment", ""),
    }


# ─── 输出 ─────────────────────────────────────────────────────────────────────

def print_result(r: dict):
    status = "✓ 通过" if r["passed"] else "✗ 未通过"
    print(f"\n{'─'*60}")
    print(f"[{r['id']}] {r['title']}  难度: {r['level']}")
    print(f"维度: {r['dimension']}")
    print(f"综合得分: {r['weighted_score']}  {status}")
    if r["dimension_scores"]:
        print("── 各维度 ──")
        for dim, info in r["dimension_scores"].items():
            score  = info.get("score", "?") if isinstance(info, dict) else info
            reason = (info.get("reason", "") if isinstance(info, dict) else "")[:80]
            print(f"  {dim}: {score}  |  {reason}")
    if r["overall_comment"]:
        print(f"── 整体评价 ──\n  {r['overall_comment']}")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XSCT Bench Dataset 评测脚本（与 xsct.ai 平台评分逻辑对齐）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # Gemini 3 Flash，xsct-l basic 前 5 条
  python scripts/evaluate.py \\
      --model google/gemini-3-flash-preview \\
      --type xsct-l --level basic --limit 5 \\
      --api-key sk-or-xxx

  # Claude Sonnet 4.6，全量评测，结果写文件
  python scripts/evaluate.py \\
      --model anthropic/claude-sonnet-4-6 \\
      --type xsct-l --level basic \\
      --api-key sk-or-xxx \\
      --output results/claude-sonnet-basic.json

  # Kimi k2.5，指定单条
  python scripts/evaluate.py \\
      --model moonshot/kimi-k2-5 \\
      --type xsct-l \\
      --id l_hallucination_054 \\
      --api-key sk-or-xxx

  # 自定义裁判模型（默认 Gemini 3 Flash）
  python scripts/evaluate.py \\
      --model anthropic/claude-sonnet-4-6 \\
      --type xsct-l --level hard \\
      --api-key sk-or-xxx \\
      --judge-model anthropic/claude-sonnet-4-6 \\
      --judge-api-key sk-or-yyy
        """,
    )
    parser.add_argument("--model",          required=True, help="被测模型 ID（OpenRouter 格式）")
    parser.add_argument("--type",           default="xsct-l",
                        choices=["xsct-l", "xsct-vg", "xsct-w"], help="测试集类型")
    parser.add_argument("--level",          default="basic",
                        choices=["basic", "medium", "hard"], help="难度级别")
    parser.add_argument("--limit",          type=int, default=0, help="评测条数上限（0=全部）")
    parser.add_argument("--id",             default="",  help="只评测指定 id 的单条用例")
    parser.add_argument("--api-key",        required=True, help="OpenRouter API Key")
    parser.add_argument("--base-url",       default=OPENROUTER_BASE, help="API Base URL")
    parser.add_argument("--judge-model",    default=JUDGE_MODEL,
                        help=f"裁判模型（默认 {JUDGE_MODEL}）")
    parser.add_argument("--judge-api-key",  default="",
                        help="裁判 API Key（不填则与 --api-key 相同）")
    parser.add_argument("--judge-base-url", default="",
                        help="裁判 Base URL（不填则与 --base-url 相同）")
    parser.add_argument("--output",         default="", help="结果输出 JSON 文件路径（可选）")
    parser.add_argument("--delay",          type=float, default=0.5,
                        help="每条用例间隔秒数（默认 0.5，避免 rate limit）")
    args = parser.parse_args()

    # 初始化客户端
    model_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    judge_key    = args.judge_api_key  or args.api_key
    judge_base   = args.judge_base_url or args.base_url
    judge_client = OpenAI(api_key=judge_key, base_url=judge_base)

    # 加载用例
    print(f"加载用例：{args.type} / {args.level}")
    testcases = load_testcases(args.type, args.level)
    if args.id:
        testcases = [tc for tc in testcases if tc["id"] == args.id]
        if not testcases:
            print(f"未找到 id={args.id} 的用例", file=sys.stderr)
            sys.exit(1)
    if args.limit > 0:
        testcases = testcases[:args.limit]

    print(f"共 {len(testcases)} 条，模型：{args.model}  裁判：{args.judge_model}")
    print(f"{'─'*60}")

    results = []
    passed  = 0

    for i, tc in enumerate(testcases, 1):
        print(f"[{i}/{len(testcases)}] {tc['id']}  {tc['title'][:40]}", end="  ", flush=True)

        r = evaluate_one(
            model_client, judge_client,
            args.model, args.judge_model,
            tc, args.level,
        )
        if r is None:
            continue

        if r["passed"]:
            passed += 1
        print(f"{'✓' if r['passed'] else '✗'}  {r['weighted_score']}")
        print_result(r)
        results.append(r)

        if i < len(testcases):
            time.sleep(args.delay)

    # 汇总
    total     = len(results)
    avg_score = sum(r["weighted_score"] for r in results) / total if total else 0
    print(f"\n{'='*60}")
    print(f"评测完成  |  模型：{args.model}")
    print(f"测试集：{args.type} / {args.level}  |  总条数：{total}")
    print(f"通过：{passed}  |  通过率：{passed/total*100:.1f}%  |  平均分：{avg_score:.1f}")

    # 按维度汇总
    dim_scores: dict[str, list] = {}
    for r in results:
        dim = r["dimension"]
        dim_scores.setdefault(dim, []).append(r["weighted_score"])
    if len(dim_scores) > 1:
        print(f"\n── 各维度平均分 ──")
        for dim, scores in sorted(dim_scores.items()):
            avg = sum(scores) / len(scores)
            print(f"  {dim}: {avg:.1f}  ({len(scores)} 条)")

    # 保存结果
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model":       args.model,
            "judge_model": args.judge_model,
            "test_type":   args.type,
            "level":       args.level,
            "total":       total,
            "passed":      passed,
            "pass_rate":   round(passed / total * 100, 1) if total else 0,
            "avg_score":   round(avg_score, 1),
            "results":     results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存：{args.output}")


if __name__ == "__main__":
    main()
