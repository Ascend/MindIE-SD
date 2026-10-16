#!/usr/bin/env python3
"""a2a 逐腿对账分析器（口径见 `../references/a2a-integrity-audit.md`）。

读入**各 rank 的日志**，逐腿核对「我发给对端的指纹」与「对端收到我的指纹」，给出判读结论：

- 全部一致 + 自腿精确 ⇒ **传输忠实**（问题在发送侧内容或消费侧，不要再查搬运）；
- 有不一致且发送缓冲前后不等（`spre != spost`）⇒ **发送缓冲在下发后被改写**；
- 有不一致但发送缓冲前后相等 ⇒ **搬运 / 落位**问题；
- 自腿不精确 ⇒ **拼装 / 打包**自身的错。

零依赖、只读、离线（不需要 NPU）；不需要新增任何集合通信。

## 日志行格式（每个 rank 在**调用结束时**打印一次，每 chunk 一行）

```text
A2AUD rank=0 call=12 chunk=3 sent=11,22,33,44 got=5,6,7,8 spre=11,22,33,44 spost=11,22,33,44
```

- `sent[j]`：本 rank **发往 destination `j`** 的那条腿的指纹（**下发前**取，`spre` 为同一批的复取）；
- `got[r]`：本 rank **从 source `r`** 收到的指纹；
- `spre` / `spost`：发送侧指纹在下发前 / 调用结束时的两次取值（省略则跳过闭环检查）。

指纹可以是任意整数（sum、hash、逐腿校验和皆可），只要**同一条腿两侧用同一算法**。

用法：

```bash
python a2a_reconcile.py --log run_rank0.log --log run_rank1.log [--log ...]
python a2a_reconcile.py --glob 'run_rank*.log'
python a2a_reconcile.py --selftest
```

退出码：0 = 传输忠实（或仅剩已声明的旁证缺口）；1 = 发现不一致；2 = 前置条件缺失（无可用记录）。

行为特征说明：脚本**不做**"事后重读"式的判断 —— 它比较的是日志里**消费时刻**记录的指纹
（这是本审计唯一有效的证据形态，理由见参考文件 §二）。
"""

from __future__ import annotations

import argparse
import glob as globmod
import re
import sys
from pathlib import Path

PASS, FAIL, PRECOND = 0, 1, 2

# 输出编码兜底：非 UTF-8 控制台（如 Windows GBK）上打印中文/符号会抛 UnicodeEncodeError，
# 让一个纯分析脚本崩在"打印结论"这一步最不值当 —— 统一按 UTF-8 输出并对不可编码字符降级。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - 老解释器/被重定向的流
        pass

LINE_RE = re.compile(
    r"A2AUD\s+rank=(?P<rank>\d+)\s+call=(?P<call>\d+)\s+chunk=(?P<chunk>-?\d+)\s+"
    r"sent=(?P<sent>[\d,]*(?:,\s*[\d,]*)*?)\s+got=(?P<got>[\d,]*)\s*"
    r"(?:spre=(?P<spre>[\d,]*)\s+spost=(?P<spost>[\d,]*))?"
)


def _ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    text = text.strip().strip(",")
    if not text:
        return []
    try:
        return [int(x.strip()) for x in text.split(",") if x.strip() != ""]
    except ValueError:
        return None


class Rec:
    __slots__ = ("call", "chunk", "got", "rank", "sent", "spost", "spre")

    def __init__(
        self,
        rank: int,
        call: int,
        chunk: int,
        sent: list[int],
        got: list[int],
        spre: list[int] | None,
        spost: list[int] | None,
    ) -> None:
        self.rank, self.call, self.chunk = rank, call, chunk
        self.sent, self.got, self.spre, self.spost = sent, got, spre, spost


def parse(paths: list[Path]) -> tuple[list[Rec], int]:
    recs: list[Rec] = []
    bad_lines = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "A2AUD" not in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                bad_lines += 1
                continue
            sent, got = _ints(m.group("sent")), _ints(m.group("got"))
            if sent is None or got is None:
                bad_lines += 1
                continue
            recs.append(
                Rec(
                    int(m.group("rank")),
                    int(m.group("call")),
                    int(m.group("chunk")),
                    sent,
                    got,
                    _ints(m.group("spre")),
                    _ints(m.group("spost")),
                )
            )
    return recs, bad_lines


def reconcile(recs: list[Rec]) -> tuple[int, list[str]]:
    """逐腿对账。返回 (退出码, 输出行)。"""
    by_key: dict[tuple[int, int], dict[int, Rec]] = {}
    for r in recs:
        by_key.setdefault((r.call, r.chunk), {})[r.rank] = r

    pairs = mismatch = self_bad = unstable = 0
    examples: list[str] = []
    for (call, chunk), ranks in sorted(by_key.items()):
        for src, rs in sorted(ranks.items()):
            # 自腿：自己发给自己的那条腿必须等于自己收到的自腿
            if src < len(rs.sent) and src < len(rs.got):
                self_bad += int(rs.sent[src] != rs.got[src])
            if rs.spre is not None and rs.spost is not None and rs.spre != rs.spost:
                unstable += 1
                if len(examples) < 5:
                    examples.append(f"call={call} chunk={chunk} rank={src}: 发送缓冲在下发后被改写")
            for dst, rd in sorted(ranks.items()):
                if dst == src:
                    continue
                if dst >= len(rs.sent) or src >= len(rd.got):
                    continue  # 该腿未被记录（例如 rank 数不足）
                pairs += 1
                if rs.sent[dst] != rd.got[src]:
                    mismatch += 1
                    if len(examples) < 5:
                        examples.append(
                            f"call={call} chunk={chunk} leg {src}->{dst}: sent={rs.sent[dst]} != got={rd.got[src]}"
                        )

    lines = [
        (
            f"records={len(recs)}  legs_compared={pairs}  mismatch={mismatch}  "
            f"self_leg_bad={self_bad}  send_buffer_rewritten={unstable}"
        ),
    ]
    if mismatch == 0 and self_bad == 0 and unstable == 0:
        lines.append("VERDICT: 传输忠实（逐腿与自腿全一致，发送缓冲未被改写）")
        lines.append("  ⇒ 若内容仍错：查发送侧内容（量化契约/拼装）或消费侧读法，不要再查搬运。")
        return PASS, lines
    if mismatch and unstable:
        lines.append("VERDICT: 发送缓冲在下发后被改写（缓冲生命周期问题）")
    elif mismatch:
        lines.append("VERDICT: 搬运 / 落位不一致（查切分形状、对齐、每目的分片边界）")
    elif self_bad:
        lines.append("VERDICT: 自腿不精确 ⇒ 拼装 / 打包自身的错（查 rank 顺序与每来源偏移）")
    else:
        lines.append("VERDICT: 发送缓冲在下发后被改写（无逐腿不一致，但 spre != spost）")
    lines.extend(f"  - {e}" for e in examples)
    lines.append("  [!] 下结论前先按 measurement-discipline §8 做负对照（审计自身的依赖/顺序也要被证明）")
    return FAIL, lines


def selftest() -> int:
    import tempfile

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def write(name: str, rows: list[str]) -> Path:
            p = root / name
            p.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return p

        # 干净：2 rank、2 chunk。语义：rank r 的 sent[d] = 发往 d 的行，got[s] = 从 s 收到的行，
        # 因此恒有 got_r[s] == sent_s[r]（自腿即 got_r[r] == sent_r[r]）。
        clean = [
            write(
                "clean_r0.log",
                [
                    "A2AUD rank=0 call=1 chunk=0 sent=10,20 got=10,30 spre=10,20 spost=10,20",
                    "A2AUD rank=0 call=1 chunk=1 sent=11,21 got=11,31 spre=11,21 spost=11,21",
                ],
            ),
            write(
                "clean_r1.log",
                [
                    "A2AUD rank=1 call=1 chunk=0 sent=30,40 got=20,40 spre=30,40 spost=30,40",
                    "A2AUD rank=1 call=1 chunk=1 sent=31,41 got=21,41 spre=31,41 spost=31,41",
                ],
            ),
        ]
        # 逐腿不一致：rank1 说发给 rank0 的是 30，而 rank0 收到的却是 99
        bad_leg = [
            write("badleg_r0.log", ["A2AUD rank=0 call=1 chunk=0 sent=10,20 got=10,99"]),
            write("badleg_r1.log", ["A2AUD rank=1 call=1 chunk=0 sent=30,41 got=20,41"]),
        ]
        # 发送缓冲被改写：逐腿一致，但下发前/后的发送侧指纹不同
        unstable = [
            write("unstable_r0.log", ["A2AUD rank=0 call=1 chunk=0 sent=10,20 got=10,30 spre=1,2 spost=10,20"]),
            write("unstable_r1.log", ["A2AUD rank=1 call=1 chunk=0 sent=30,40 got=20,40 spre=3,4 spost=30,40"]),
        ]
        # 自腿坏：rank0 收到的自腿不等于它自己发出的那一行
        selfbad = [
            write("self_r0.log", ["A2AUD rank=0 call=1 chunk=0 sent=10,20 got=777,30"]),
            write("self_r1.log", ["A2AUD rank=1 call=1 chunk=0 sent=30,40 got=20,40"]),
        ]

        for label, paths, want in (
            ("干净运行（传输忠实）", clean, PASS),
            ("逐腿不一致", bad_leg, FAIL),
            ("发送缓冲被改写", unstable, FAIL),
            ("自腿不精确", selfbad, FAIL),
        ):
            recs, bad = parse(paths)
            if bad:
                failures.append(f"{label}: 解析失败 {bad} 行")
                continue
            code, out = reconcile(recs)
            if code != want:
                failures.append(f"{label}: 期望 rc={want}，实得 rc={code}（{out[0]}）")

        # 残缺行必须被计数而不崩
        junk = write("junk.log", ["A2AUD rank=x broken line", "not a record"])
        recs, bad = parse([junk])
        if recs or bad != 1:
            failures.append(f"残缺行处理：期望 recs=0/bad=1，实得 recs={len(recs)}/bad={bad}")

    if failures:
        print("a2a_reconcile --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return FAIL
    print("a2a_reconcile --selftest: PASS（4 类样例逐类判定正确，残缺行被计数不崩）")
    return PASS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=[], help="某个 rank 的日志（可重复）")
    ap.add_argument("--glob", help="按通配符收集日志（如 'run_rank*.log'）")
    ap.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    paths = [Path(p) for p in a.log]
    if a.glob:
        paths += [Path(p) for p in sorted(globmod.glob(a.glob))]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("a2a_reconcile: 前置条件缺失：没有可用的日志文件（用 --log/--glob）", file=sys.stderr)
        return PRECOND

    recs, bad = parse(paths)
    if not recs:
        print(f"a2a_reconcile: 前置条件缺失：{len(paths)} 个文件里没有 A2AUD 记录（残缺行 {bad}）", file=sys.stderr)
        return PRECOND
    if bad:
        print(f"（注意：{bad} 行无法解析，已跳过）")
    code, lines = reconcile(recs)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
