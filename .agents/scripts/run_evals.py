#!/usr/bin/env python3
"""技能 evals 编排与「裁定覆盖度」门禁（`.agents/skills/*/evals/evals.json`）。

本仓 CI **没有 LLM API**，故本脚本**不做语义判断**，只做两件确定性的事：
`--pack` 把全部用例导出成**裁定任务包**（每条 expectation 一行，`verdict`/`evidence` 留空），
由**会话中的 agent（它本身就是模型）**逐条填写；`--check` 校验**裁定覆盖度**——每条必有裁定、
取值合法（`pass`/`fail`/`skip`）、`pass`/`fail` 必带 evidence、`skip` 必写理由，且结果文件的
expectation 原文与当前 `evals.json` **逐字一致**（防过期或错位）。

与 `stage_gate.py` 同构：**门禁校验证据齐备性，不校验判断本身的正确性**。
`--check` 纯确定性、可进 CI；`--pack` 产物与裁定过程属会话产物、不入库。
**细节与经验见同目录 `README.md`**（与脚本同源同改）。

用法：

```bash
python .agents/scripts/run_evals.py --list                       # 覆盖概览（技能/用例/expectation 数）
python .agents/scripts/run_evals.py --pack --out /tmp/skill_evals.json   # 生成裁定任务包
python .agents/scripts/run_evals.py --check --results /tmp/skill_evals.json  # 覆盖度门禁
python .agents/scripts/run_evals.py --selftest                   # 负样本自测（验证门禁本身有效）
```

退出码：0 = 覆盖完整；1 = 存在缺口/非法裁定；2 = 前置条件缺失（`.agents` 或 results 文件不存在）。
零网络、零模型、零 NPU、只读（`--pack` 除写出结果包外不改任何文件）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

AGENTS = ".agents"
SKILLS = "skills"
SCHEMA = 1
VERDICTS = ("pass", "fail", "skip")


class Case:
    """一条 eval 用例（含其所属技能与 expectation 原文）。"""

    __slots__ = ("case_id", "expectations", "prompt", "skill")

    def __init__(self, skill: str, case_id: object, prompt: str, expectations: list[str]) -> None:
        self.skill = skill
        self.case_id = case_id
        self.prompt = prompt
        self.expectations = expectations

    @property
    def key(self) -> str:
        return f"{self.skill}#{self.case_id}"


def load_cases(agents: Path) -> tuple[list[Case], list[str]]:
    """读取全部技能的 evals.json；返回 (用例列表, 结构性问题)。"""
    cases: list[Case] = []
    problems: list[str] = []
    skills_dir = agents / SKILLS
    for skill in sorted(d for d in skills_dir.iterdir() if d.is_dir()):
        path = skill / "evals" / "evals.json"
        if not path.is_file():
            problems.append(f"{skill.name}: 缺 evals/evals.json")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{skill.name}: evals.json 不可解析（{exc}）")
            continue
        raw_cases = data.get("evals") if isinstance(data, dict) else None
        if not isinstance(raw_cases, list):
            problems.append(f"{skill.name}: 缺 `evals` 数组")
            continue
        for idx, item in enumerate(raw_cases, start=1):
            if not isinstance(item, dict):
                problems.append(f"{skill.name}: 第 {idx} 条用例不是对象")
                continue
            expectations = item.get("expectations")
            if not isinstance(expectations, list) or not expectations:
                problems.append(f"{skill.name}#{item.get('id', idx)}: expectations 为空")
                continue
            cases.append(
                Case(
                    skill.name,
                    item.get("id", idx),
                    str(item.get("prompt") or ""),
                    [str(x) for x in expectations],
                )
            )
    return cases, problems


def build_pack(cases: list[Case]) -> dict:
    """生成裁定任务包骨架：每条 expectation 一行，裁定字段留空。"""
    verdicts: dict[str, object] = {}
    for case in cases:
        verdicts[case.key] = {
            "prompt": case.prompt,
            "checks": [{"expectation": exp, "verdict": "", "evidence": ""} for exp in case.expectations],
        }
    return {
        "schema": SCHEMA,
        "instructions": (
            "由会话内 agent 逐条填写：verdict ∈ {pass, fail, skip}；"
            "pass/fail 必须写 evidence（可复核的观察）；skip 必须写理由。"
            "expectation 原文不要改动——它是结果文件与 evals.json 的对齐锚点。"
        ),
        "verdicts": verdicts,
    }


def check_verdicts(cases: list[Case], results: dict) -> list[str]:
    """覆盖度校验：返回问题清单（空 = 通过）。"""
    errors: list[str] = []
    verdicts = results.get("verdicts")
    if not isinstance(verdicts, dict):
        return ["结果文件缺 `verdicts` 对象"]

    expected_keys = {case.key for case in cases}
    for extra in sorted(set(verdicts) - expected_keys):
        errors.append(f"{extra}: 结果文件里的用例在当前 evals.json 中不存在（结果已过期？）")

    for case in cases:
        entry = verdicts.get(case.key)
        if not isinstance(entry, dict):
            errors.append(f"{case.key}: 缺裁定条目（未覆盖）")
            continue
        checks = entry.get("checks")
        if not isinstance(checks, list):
            errors.append(f"{case.key}: 缺 `checks` 数组")
            continue
        if len(checks) != len(case.expectations):
            errors.append(f"{case.key}: 裁定条数 {len(checks)} ≠ expectation 条数 {len(case.expectations)}")
            continue
        for pos, (check, expectation) in enumerate(zip(checks, case.expectations), start=1):
            if not isinstance(check, dict):
                errors.append(f"{case.key} 第 {pos} 条: 裁定不是对象")
                continue
            if check.get("expectation") != expectation:
                errors.append(f"{case.key} 第 {pos} 条: expectation 原文与 evals.json 不一致（结果已过期或错位）")
                continue
            verdict = check.get("verdict")
            if verdict not in VERDICTS:
                errors.append(f"{case.key} 第 {pos} 条: 裁定缺失或非法「{verdict}」（可选 {list(VERDICTS)}）")
                continue
            evidence = str(check.get("evidence") or "").strip()
            if verdict in ("pass", "fail") and not evidence:
                errors.append(f"{case.key} 第 {pos} 条: 裁定为 {verdict} 但未写 evidence")
            if verdict == "skip" and not evidence:
                errors.append(f"{case.key} 第 {pos} 条: 裁定为 skip 但未写理由")
    return errors


def summarize(cases: list[Case], results: dict | None) -> list[str]:
    """输出「技能 → 用例数 / expectation 数 / 已裁定数」概览。"""
    lines = []
    per_skill: dict[str, list[int]] = {}
    for case in cases:
        slot = per_skill.setdefault(case.skill, [0, 0])
        slot[0] += 1
        slot[1] += len(case.expectations)
    total_cases = total_exps = 0
    lines.append(f"{'技能':<34}{'用例':>5}{'expectation':>14}{'已裁定':>8}")
    for skill in sorted(per_skill):
        n_cases, n_exps = per_skill[skill]
        done = 0
        if results:
            verdicts = results.get("verdicts") or {}
            for case in cases:
                if case.skill != skill:
                    continue
                entry = verdicts.get(case.key)
                if isinstance(entry, dict) and isinstance(entry.get("checks"), list):
                    done += sum(1 for c in entry["checks"] if isinstance(c, dict) and c.get("verdict") in VERDICTS)
        lines.append(f"{skill:<34}{n_cases:>5}{n_exps:>14}{done:>8}")
        total_cases += n_cases
        total_exps += n_exps
    lines.append(f"{'合计':<34}{total_cases:>5}{total_exps:>14}")
    return lines


def _write_pack(cases: list[Case], out: Path) -> int:
    pack = build_pack(cases)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(len(c.expectations) for c in cases)
    print(f"已生成裁定任务包：{out}（{len(cases)} 用例 / {total} 条 expectation 待裁定）")
    return 0


# ---------------------------------------------------------------- 自测


def _selftest(base: Path | None = None) -> int:
    """负样本自测：每类缺口都必须被 --check 抓住，完整结果必须通过。

    夹具用确定性路径（`mkdtemp` 在 Windows 上建的目录 DACL 过严，嵌套建目录会被拒）。
    """
    failures: list[str] = []
    root = (base or Path(__file__).resolve().parents[1]) / ".run_evals_selftest"
    agents = root / AGENTS
    shutil.rmtree(root, ignore_errors=True)
    try:
        for name, cases in (
            ("skill-a", [{"id": 1, "prompt": "p1", "expectations": ["e1", "e2"]}]),
            ("skill-b", [{"id": 1, "prompt": "p2", "expectations": ["e3"]}]),
        ):
            (agents / SKILLS / name / "evals").mkdir(parents=True, exist_ok=True)
            path = agents / SKILLS / name / "evals" / "evals.json"
            path.write_text(json.dumps({"skill_name": name, "evals": cases}), encoding="utf-8")

        cases, problems = load_cases(agents)
        if problems:
            failures.append(f"夹具加载出现问题：{problems}")
        if len(cases) != 2:
            failures.append(f"夹具用例数应为 2，实为 {len(cases)}")

        good = {
            "schema": SCHEMA,
            "verdicts": {
                "skill-a#1": {
                    "checks": [
                        {"expectation": "e1", "verdict": "pass", "evidence": "观察到 A"},
                        {"expectation": "e2", "verdict": "skip", "evidence": "环境不支持"},
                    ]
                },
                "skill-b#1": {"checks": [{"expectation": "e3", "verdict": "fail", "evidence": "答成了 B"}]},
            },
        }
        residue = check_verdicts(cases, good)
        for item in residue:
            failures.append(f"完整结果被误报：{item}")

        negatives = {
            "缺裁定条目": {"schema": SCHEMA, "verdicts": {"skill-a#1": good["verdicts"]["skill-a#1"]}},
            "裁定为空": {
                "schema": SCHEMA,
                "verdicts": {
                    "skill-a#1": {
                        "checks": [
                            {"expectation": "e1", "verdict": "", "evidence": ""},
                            good["verdicts"]["skill-a#1"]["checks"][1],
                        ]
                    },
                    "skill-b#1": good["verdicts"]["skill-b#1"],
                },
            },
            "pass 无证据": {
                "schema": SCHEMA,
                "verdicts": {
                    "skill-a#1": {
                        "checks": [
                            {"expectation": "e1", "verdict": "pass", "evidence": ""},
                            good["verdicts"]["skill-a#1"]["checks"][1],
                        ]
                    },
                    "skill-b#1": good["verdicts"]["skill-b#1"],
                },
            },
            "expectation 漂移": {
                "schema": SCHEMA,
                "verdicts": {
                    "skill-a#1": {
                        "checks": [
                            {"expectation": "e1-改过", "verdict": "pass", "evidence": "x"},
                            good["verdicts"]["skill-a#1"]["checks"][1],
                        ]
                    },
                    "skill-b#1": good["verdicts"]["skill-b#1"],
                },
            },
            "非法裁定值": {
                "schema": SCHEMA,
                "verdicts": {
                    "skill-a#1": {
                        "checks": [
                            {"expectation": "e1", "verdict": "maybe", "evidence": "x"},
                            good["verdicts"]["skill-a#1"]["checks"][1],
                        ]
                    },
                    "skill-b#1": good["verdicts"]["skill-b#1"],
                },
            },
            "条数不符": {
                "schema": SCHEMA,
                "verdicts": {
                    "skill-a#1": {"checks": [{"expectation": "e1", "verdict": "pass", "evidence": "x"}]},
                    "skill-b#1": good["verdicts"]["skill-b#1"],
                },
            },
        }
        for label, results in negatives.items():
            if not check_verdicts(cases, results):
                failures.append(f"负样本未触发：{label}")

        # 结构问题：expectations 为空必须被 load_cases 报出（KB005 之外的补充）
        (agents / SKILLS / "skill-c" / "evals").mkdir(parents=True, exist_ok=True)
        (agents / SKILLS / "skill-c" / "evals" / "evals.json").write_text(
            json.dumps({"skill_name": "skill-c", "evals": [{"id": 1, "prompt": "p", "expectations": []}]}),
            encoding="utf-8",
        )
        _, problems2 = load_cases(agents)
        if not any("skill-c" in p for p in problems2):
            failures.append("负样本未触发：expectations 为空未被报出")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("run_evals --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("run_evals --selftest: PASS（6 类缺口逐类触发，完整结果 0 违规）")
    return 0


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="仓库根（默认当前目录）")
    parser.add_argument("--list", action="store_true", help="列出技能/用例/expectation 覆盖概览")
    parser.add_argument("--pack", action="store_true", help="生成裁定任务包骨架")
    parser.add_argument("--check", action="store_true", help="校验裁定覆盖度（确定性，可进 CI）")
    parser.add_argument("--out", help="--pack 的输出路径")
    parser.add_argument("--results", help="--check 读取的结果文件")
    parser.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    root = Path(args.root).resolve()
    agents = root / AGENTS
    if not (agents / SKILLS).is_dir():
        print(f"run_evals: 前置条件缺失：{agents / SKILLS} 不存在", file=sys.stderr)
        return 2

    cases, problems = load_cases(agents)
    for item in problems:
        print(f"  - {item}", file=sys.stderr)
    if problems:
        print(f"run_evals: evals.json 结构问题 {len(problems)} 处（先修结构再谈裁定）", file=sys.stderr)
        return 1

    if args.pack:
        out = Path(args.out) if args.out else root / "skill_evals_pack.json"
        return _write_pack(cases, out)

    if args.check:
        if not args.results:
            print("run_evals: --check 需要 --results <结果文件>", file=sys.stderr)
            return 2
        path = Path(args.results)
        if not path.is_file():
            print(f"run_evals: 前置条件缺失：结果文件不存在（{path}）", file=sys.stderr)
            return 2
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"run_evals: 结果文件不可解析（{exc}）", file=sys.stderr)
            return 2
        errors = check_verdicts(cases, results)
        for line in summarize(cases, results):
            print(line)
        if errors:
            print(f"\nrun_evals: 裁定缺口 {len(errors)} 处")
            for item in errors:
                print(f"  - {item}")
            return 1
        total = sum(len(c.expectations) for c in cases)
        print(f"\nrun_evals: 裁定覆盖完整（{len(cases)} 用例 / {total} 条 expectation 全部有裁定且合法）")
        return 0

    # 默认（含 --list）：覆盖概览
    for line in summarize(cases, None):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
