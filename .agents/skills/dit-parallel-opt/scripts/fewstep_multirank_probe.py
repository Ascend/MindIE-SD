#!/usr/bin/env python
"""Few-step x multi-rank parallel-configuration probe with a DiT-only caliber.

Answers: *can a few-step, multi-rank run be used to rank parallel configurations, and
when does that ranking extrapolate to full step counts?*

Mandated caliber -- **DiT-only**: every parallel comparison metric in the emitted
evidence bundle is taken from the denoise stage alone (``<Pipeline>.diffuse`` wall
clock and the per-denoise-step deltas).  VAE decode, prompt encode, model load and
warmup are recorded but never enter a comparison, because at few steps the fixed
per-request overhead dominates the request and swamps the parallel difference.

What it does, per configuration:
  1. launches one ``vllm-omni serve`` on an isolated card group (scoped PID-tree kill);
  2. verifies the *counting contract* (rank count, sp/tp/dp sizes, offload counters) so a
     silently-inert configuration cannot pass as a measured one;
  3. issues interleaved few-step / anchor-step requests so thermal and co-tenant drift is
     spread across step tiers instead of being confounded with the tier;
  4. parses the serve log (the log is authoritative: request segmentation keys off the
     API server's ``Video sampling params: steps=N`` line, which exists even when the
     response never reaches the client);
  5. emits ``summary.json`` + ``evidence.md`` with per-cell dispersion and the
     few-step-vs-anchor ordering verdict, gated at a noise threshold.

Runs on the inference host / inside the inference container; it only needs ``curl`` and
the local model weights.  Point ``--parse-only`` at an existing run directory to
re-analyse logs without touching the NPU.

Example
-------
    python fewstep_multirank_probe.py --out-dir ./runs/probe \\
        --config A:4,5:tp1,usp2,dlo --config B:4,5:tp2,usp1 \\
        --steps-few 4 --steps-anchor 10 --reps 3

Privacy: this file carries no host/IP/container/user specific value; pass site values at
run time (``--model``, ``--venv``, ``--out-dir``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

NOISE_THRESHOLD_PCT = 3.0
"""Dispersion above this marks a cell unstable and unusable for a criterion."""

DRIFT_THRESHOLD_PCT = 5.0
"""Above this per-step DiT drift between the few-step and anchor tiers, the few-step
ranking must be re-verified at the anchor (or higher) step count."""

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
REQ_RE = re.compile(r"Video sampling params: steps=(\d+)")
PROG_RE = re.compile(r"(?:^|[^0-9])(\d+)/(\d+)\s*\[\s*(?:(\d+):)?(\d+):(\d+)<")
STAGE_RE = re.compile(
    r"DiffusionPipelineProfiler\]\s*([A-Za-z0-9_]+)\.(diffuse|decode|forward|encode_prompt)"
    r"\s+took\s+([0-9.]+)s"
)
RANK_RE = re.compile(r"DiffusionWorker_SP(\d+)")
POST_RE = re.compile(r"POST /v1/videos/sync HTTP/1\.1\"\s+(\d+)")
DEAD_RE = re.compile(
    r"Traceback \(most recent call last\)|Worker failed|ERR00[0-9]|exitcode=[1-9]"
    r"|OutOfMemory|OOM killer"
)

CONTRACT_PATTERNS = (
    "Distributed layer-wise offloading enabled",
    "Applying sequence parallelism",
    "Building SP subgroups",
    "sp_size=",
    "dp_size=",
)
STEP_TIERS = ("few", "anchor")


def kill_tree(pid: int | None) -> None:
    """Kill a process and its descendants only -- never a blanket pattern match.

    Sharing one host between several inference runs is normal; a pattern-based kill
    would take down other tenants' servers.
    """
    if not pid:
        return
    with contextlib.suppress(OSError):
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False).stdout
        for child in out.split():
            kill_tree(int(child))
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)


def read_text(path: str) -> str:
    with open(path, "rb") as fh:
        return ANSI_RE.sub("", fh.read().decode("utf-8", "replace")).replace("\r", "\n")


def launch_serve(args: argparse.Namespace, cfg: dict, log_path: str) -> subprocess.Popen:
    extra: list[str] = []
    if cfg["dlo"]:
        extra.append("--enable-distributed-layerwise-offload")
    cmd = [
        os.path.join(args.venv, "bin", "vllm-omni"),
        "serve",
        args.model,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--num-gpus",
        str(cfg["world"]),
        "--tensor-parallel-size",
        str(cfg["tp"]),
        "--usp",
        str(cfg["usp"]),
        "--ring",
        "1",
        "--text-encoder-tp-size",
        str(cfg["world"]),
        "--vae-patch-parallel-size",
        str(cfg["world"]),
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--enable-diffusion-pipeline-profiler",
        *extra,
        "--init-timeout",
        str(args.init_timeout),
        "--stage-init-timeout",
        str(args.init_timeout),
    ]
    env = dict(os.environ)
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": cfg["cards"],
            "VLLM_OMNI_DISABLE_VLLM_ASCEND": "true",
            "VLLM_PLUGINS": "",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "28800",
            "TOKENIZERS_PARALLELISM": "false",
            "OMNI_MINDIE_COMPILE": "0",
            "OMNI_KPROF": "1",
            "OMNI_KPROF_AFTER": str(args.kprof_after),
            "OMNI_KPROF_DIR": os.path.join(args.out_dir, f"kprof_{cfg['name']}"),
        }
    )
    with open(log_path, "wb") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def wait_health(port: int, log_path: str, timeout_s: int) -> int:
    """Return startup wall time in seconds, or -1 when the service never came up."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        code = subprocess.run(
            ["curl", "-s", "-m", "8", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{port}/health"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if code == "200":
            return int(time.time() - t0)
        if os.path.exists(log_path) and DEAD_RE.search(read_text(log_path)[-200_000:]):
            return -1
        time.sleep(15)
    return -1


def post_video(args: argparse.Namespace, steps: int, out_mp4: str) -> dict:
    extra_params = f'{{"task":"t2va","duration":{args.duration},"audio_flow_shift":{args.audio_flow_shift}}}'
    form = [
        "-F",
        f"prompt={args.prompt}",
        "-F",
        f"width={args.width}",
        "-F",
        f"height={args.height}",
        "-F",
        "aspect_ratio=16:9",
        "-F",
        f"fps={args.fps}",
        "-F",
        f"num_inference_steps={steps}",
        "-F",
        f"flow_shift={args.flow_shift}",
        "-F",
        f"seed={args.seed}",
        "-F",
        f"extra_params={extra_params}",
    ]
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(args.request_timeout),
        "-o",
        out_mp4,
        "-w",
        "%{http_code} %{time_total} %{size_download}",
        "-X",
        "POST",
        f"http://127.0.0.1:{args.port}/v1/videos/sync",
        *form,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    parts = (res.stdout or "").split()
    rec = {"rc": res.returncode, "raw": (res.stdout or res.stderr or "").strip()[:200]}
    if len(parts) == 3:
        rec.update(http=int(parts[0]), seconds=float(parts[1]), bytes=int(parts[2]))
    else:
        rec.update(http=0, seconds=None, bytes=None)
    return rec


def parse_serve_log(path: str) -> list[dict]:
    """Segment a serve log into requests, keyed on the API server's step-count line."""
    if not os.path.exists(path):
        return []
    segments: list[dict] = []
    cur: dict | None = None
    for line in read_text(path).split("\n"):
        m = REQ_RE.search(line)
        if m:
            if cur is not None:
                segments.append(cur)
            cur = {"lines": [line], "http": None, "steps": int(m.group(1))}
            continue
        if cur is None:
            cur = {"lines": [line], "http": None, "steps": None}
            continue
        cur["lines"].append(line)
        m2 = POST_RE.search(line)
        if m2:
            cur["http"] = int(m2.group(1))
    if cur is not None:
        segments.append(cur)
    return [s for s in segments if s["steps"] is not None]


def segment_metrics(seg: dict) -> dict:
    """Extract DiT-only metrics for one request.

    ``diffuse`` is the denoise stage alone; the tqdm bar gives the per-iteration split.
    The bar's ``s/it`` field is a smoothed EMA rate and must NOT be used to rebuild
    elapsed time -- only the cumulative ``N/M [MM:SS`` is trustworthy (1 s quantisation).
    """
    stages: dict[str, list[float]] = {}
    cum: dict[int, int] = {}
    bar_total = None
    for line in seg["lines"]:
        m = STAGE_RE.search(line)
        if m:
            stages.setdefault(m.group(2), []).append(float(m.group(3)))
        m = PROG_RE.search(line)
        if m:
            n = int(m.group(1))
            if n == 0:
                continue  # tqdm prints `0/N [00:00` before the first iteration
            bar_total = int(m.group(2))
            hh = int(m.group(3)) if m.group(3) else 0
            elapsed = hh * 3600 + int(m.group(4)) * 60 + int(m.group(5))
            if n not in cum or elapsed > cum[n]:
                cum[n] = elapsed
    deltas: list[float] = []
    last = 0
    for n in sorted(cum):
        deltas.append(float(cum[n] - last))
        last = cum[n]
    dit = min(stages["diffuse"]) if stages.get("diffuse") else None
    n_iter = len(deltas) or bar_total
    return {
        "steps_requested": seg["steps"],
        "steps_bar_total": bar_total,
        "steps_measured": len(deltas),
        "step_s_first": deltas[0] if deltas else None,
        "step_s_steady_median": statistics.median(deltas[1:]) if len(deltas) > 1 else None,
        "dit_stage_s": dit,
        "dit_stage_per_iter_s": round(dit / n_iter, 4) if dit and n_iter else None,
        "decode_stage_s": min(stages["decode"]) if stages.get("decode") else None,
        "encode_stage_s": min(stages["encode_prompt"]) if stages.get("encode_prompt") else None,
        "forward_stage_s": min(stages["forward"]) if stages.get("forward") else None,
        "http": seg["http"],
    }


def contract_counters(log_path: str, cfg: dict) -> dict:
    """Counting contract: prove the configuration was really in force (anti no-op)."""
    if not os.path.exists(log_path):
        return {"error": "log missing"}
    text = read_text(log_path)
    ranks = sorted({int(r) for r in RANK_RE.findall(text)})
    counters = {
        "rank_labels": ranks,
        "rank_count": len(ranks),
        "expected_world": cfg["world"],
        "rank_count_ok": len(ranks) == cfg["world"],
        "sp_size_seen": sorted(set(re.findall(r"sp_size=(\d+)", text))),
        "dp_size_seen": sorted(set(re.findall(r"dp_size=(\d+)", text))),
        "dlo_line_seen": "Distributed layer-wise offloading enabled" in text,
        "lines": [],
    }
    for line in text.split("\n"):
        if any(p in line for p in CONTRACT_PATTERNS):
            counters["lines"].append(line.strip()[:190])
        if len(counters["lines"]) >= 6:
            break
    return counters


def median_or_none(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.median(vals), 4) if vals else None


def spread_pct(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    return round((max(vals) - min(vals)) / med * 100.0, 2) if med else None


def build_cells(requests: list[dict], steps_few: int, steps_anchor: int) -> list[dict]:
    cells = []
    for tier, steps in zip(STEP_TIERS, (steps_few, steps_anchor), strict=True):
        sel = [r for r in requests if r["steps_requested"] == steps]
        if not sel:
            continue
        cells.append(
            {
                "tier": tier,
                "steps_requested": steps,
                "reps": len(sel),
                "dit_stage_s_median": median_or_none([r["dit_stage_s"] for r in sel]),
                "dit_stage_s_all": [r["dit_stage_s"] for r in sel],
                "dit_stage_dispersion_pct": spread_pct([r["dit_stage_s"] for r in sel]),
                "dit_per_iter_s_median": median_or_none([r["dit_stage_per_iter_s"] for r in sel]),
                "step_steady_s_median": median_or_none([r["step_s_steady_median"] for r in sel]),
                "step_first_s_median": median_or_none([r["step_s_first"] for r in sel]),
                "decode_stage_s_median": median_or_none([r["decode_stage_s"] for r in sel]),
            }
        )
    return cells


def verdict_for_config(cells: list[dict]) -> dict:
    """Apply the extrapolation criteria to one configuration."""
    by_tier = {c["tier"]: c for c in cells}
    few, anchor = by_tier.get("few"), by_tier.get("anchor")
    if not few or not anchor:
        return {"status": "incomplete", "reason": "missing few-step or anchor tier data"}
    few_pi, anchor_pi = few["dit_per_iter_s_median"], anchor["dit_per_iter_s_median"]
    if not few_pi or not anchor_pi:
        return {"status": "incomplete", "reason": "per-iteration DiT time unavailable"}
    ratio = anchor_pi / few_pi
    drift = abs(ratio - 1.0) * 100.0
    dispersions = [c["dit_stage_dispersion_pct"] for c in cells if c["dit_stage_dispersion_pct"] is not None]
    worst = max(dispersions) if dispersions else None
    base = {
        "per_iter_ratio_anchor_over_few": round(ratio, 4),
        "step_time_drift_pct": round(drift, 2),
        "worst_dispersion_pct": worst,
    }
    if worst is not None and worst > NOISE_THRESHOLD_PCT:
        return {
            **base,
            "status": "unstable",
            "reason": "within-cell dispersion exceeds the noise threshold; not usable for any ranking claim",
        }
    if drift > DRIFT_THRESHOLD_PCT:
        return {
            **base,
            "status": "reverify-at-anchor",
            "reason": "per-step DiT time is not stable across step tiers; the few-step "
            "ranking must be re-measured at the anchor step count",
        }
    return {
        **base,
        "status": "extrapolates",
        "reason": "per-step DiT time is stable across step tiers within the noise "
        "threshold and the cells are repeatable",
    }


def ordering_stability(per_config_cells: dict[str, list[dict]]) -> dict:
    few: dict[str, float] = {}
    anchor: dict[str, float] = {}
    for cfg, cells in per_config_cells.items():
        for c in cells:
            if not c["dit_per_iter_s_median"]:
                continue
            (few if c["tier"] == "few" else anchor)[cfg] = c["dit_per_iter_s_median"]
    common = sorted(set(few) & set(anchor))
    if len(common) < 2:
        return {"status": "insufficient-configs", "configs": common}
    rank_few = sorted(common, key=lambda c: few[c])
    rank_anchor = sorted(common, key=lambda c: anchor[c])
    return {
        "status": "ok",
        "configs": common,
        "rank_by_few_step_s_per_iter": rank_few,
        "rank_by_anchor_step_s_per_iter": rank_anchor,
        "order_identical": rank_few == rank_anchor,
        "few_step_per_iter_s": {c: few[c] for c in common},
        "anchor_step_per_iter_s": {c: anchor[c] for c in common},
    }


def parse_config_spec(spec: str) -> dict:
    """Parse ``NAME:CARDS:tpX,uspY[,dlo]``."""
    parts = spec.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(f"config spec must be NAME:CARDS:tpX,uspY[,dlo] -- got {spec!r}")
    name, cards, opts = parts[0], parts[1], parts[2]
    tp = usp = None
    dlo = False
    for opt in opts.split(","):
        opt = opt.strip()
        if opt.startswith("tp"):
            tp = int(opt[2:])
        elif opt.startswith("usp"):
            usp = int(opt[3:])
        elif opt == "dlo":
            dlo = True
    if tp is None or usp is None:
        raise argparse.ArgumentTypeError(f"config spec needs tpN and uspN: {spec!r}")
    world = tp * usp
    n_cards = len(cards.split(","))
    if n_cards != world:
        raise argparse.ArgumentTypeError(f"cards ({n_cards}) must match tp*usp ({world}) for {name}")
    return {"name": name, "cards": cards, "tp": tp, "usp": usp, "world": world, "dlo": dlo}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        type=parse_config_spec,
        help="NAME:CARDS:tpX,uspY[,dlo]; repeat for each configuration",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--model", default=os.environ.get("S32_MODEL_DIR", ""), help="model directory on the inference host"
    )
    ap.add_argument(
        "--venv",
        default=sys.prefix,
        help="environment prefix providing the vllm-omni CLI; defaults to the "
        "running interpreter's prefix, so run this script with the "
        "inference environment's python",
    )
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--steps-few", type=int, default=4)
    ap.add_argument("--steps-anchor", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--warm-steps", type=int, default=6, help="excluded warmup request; also carries the profiler capture"
    )
    ap.add_argument(
        "--kprof-after", type=int, default=5, help="n-th DiT forward to profile (lands inside the warmup request)"
    )
    ap.add_argument("--init-timeout", type=int, default=1800)
    ap.add_argument("--request-timeout", type=int, default=28800)
    ap.add_argument(
        "--prompt", default="A cinematic shot of a glowing robot walking through a rainy neon city street at night."
    )
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--duration", default="15")
    ap.add_argument("--seed", type=int, default=1101)
    ap.add_argument("--flow-shift", default="12")
    ap.add_argument("--audio-flow-shift", default="3.0")
    ap.add_argument(
        "--parse-only", action="store_true", help="skip all NPU work; re-analyse existing serve logs in --out-dir"
    )
    return ap.parse_args(argv)


def run_config(args: argparse.Namespace, cfg: dict, all_requests: dict) -> None:
    log_path = os.path.join(args.out_dir, f"{cfg['name']}_serve.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    kprof_dir = os.path.join(args.out_dir, f"kprof_{cfg['name']}")
    if os.path.isdir(kprof_dir):
        shutil.rmtree(kprof_dir)
    print(
        f"[{cfg['name']}] launching world={cfg['world']} tp={cfg['tp']} usp={cfg['usp']} "
        f"dlo={cfg['dlo']} cards={cfg['cards']}"
    )
    proc = launch_serve(args, cfg, log_path)
    try:
        startup = wait_health(args.port, log_path, args.init_timeout)
        if startup < 0:
            print(f"[{cfg['name']}] SERVICE_NOT_READY")
            return
        print(f"[{cfg['name']}] startup_s={startup}")
        plan = [("warm", args.warm_steps)]
        for rep in range(1, args.reps + 1):
            plan.append((f"few_r{rep}", args.steps_few))
            plan.append((f"anchor_r{rep}", args.steps_anchor))
        for tag, steps in plan:
            out_mp4 = os.path.join(args.out_dir, f"{cfg['name']}_{tag}.mp4")
            rec = post_video(args, steps, out_mp4)
            print(f"[{cfg['name']}] req {tag} steps={steps} http={rec['http']} t={rec['seconds']}")
    finally:
        kill_tree(proc.pid)
        time.sleep(10)
        all_requests[cfg["name"]] = parse_serve_log(log_path)


def write_evidence(out_dir: str, bundle: dict) -> None:
    lines = [
        "# DiT-only few-step x multi-rank probe",
        "",
        "> Every parallel-comparison number below comes from the **DiT denoise stage only**.",
        "> VAE decode / prompt encode / load / warmup are excluded by construction.",
        "",
        "| config | tier | steps | reps | DiT per-iter (s) | DiT stage (s) | disp % | decode (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in bundle["configs"].items():
        for cell in c.get("cells", []):
            lines.append(
                f"| {name} | {cell['tier']} | {cell['steps_requested']} | {cell['reps']} "
                f"| {cell['dit_per_iter_s_median']} | {cell['dit_stage_s_median']} "
                f"| {cell['dit_stage_dispersion_pct']} | {cell['decode_stage_s_median']} |"
            )
    lines += ["", "## Per-config verdict", ""]
    for name, c in bundle["configs"].items():
        v = c.get("verdict")
        if v:
            lines.append(
                f"- **{name}**: `{v['status']}` -- {v['reason']} "
                f"(drift {v.get('step_time_drift_pct')}%, "
                f"worst dispersion {v.get('worst_dispersion_pct')}%)"
            )
    lines += [
        "",
        "## Ordering stability across step tiers",
        "",
        "```json",
        json.dumps(bundle["ordering_stability"], indent=2),
        "```",
        "",
    ]
    with open(os.path.join(out_dir, "evidence.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.config and not args.parse_only:
        print("nothing to do: pass --config (or --parse-only)", file=sys.stderr)
        return 2
    if not args.parse_only and not args.model:
        print("--model is required unless --parse-only", file=sys.stderr)
        return 2

    all_requests: dict[str, list[dict]] = {}
    if not args.parse_only:
        for cfg in args.config:
            run_config(args, cfg, all_requests)
    else:
        for fn in sorted(os.listdir(args.out_dir)):
            if fn.endswith("_serve.log"):
                name = fn[: -len("_serve.log")]
                all_requests[name] = parse_serve_log(os.path.join(args.out_dir, fn))

    per_cfg: dict[str, list[dict]] = {}
    bundle: dict = {
        "caliber": {
            "primary": "DiT denoise stage only: <Pipeline>.diffuse wall clock",
            "derived": "per-denoise-iteration DiT time = diffuse / measured iterations",
            "excluded": ["VAE decode", "prompt encode", "model load", "warmup", "response encoding"],
            "request_delimiter": "API server 'Video sampling params: steps=N' line",
            "noise_threshold_pct": NOISE_THRESHOLD_PCT,
            "drift_threshold_pct": DRIFT_THRESHOLD_PCT,
        },
        "args": {k: v for k, v in vars(args).items() if k != "config"},
        "configs": {},
    }
    specs = {c["name"]: c for c in args.config}
    for name, segments in all_requests.items():
        reqs = [segment_metrics(s) for s in segments]
        cells = build_cells(reqs, args.steps_few, args.steps_anchor)
        per_cfg[name] = cells
        entry: dict = {"spec": specs.get(name), "requests": reqs, "cells": cells}
        if specs.get(name):
            entry["contract"] = contract_counters(os.path.join(args.out_dir, f"{name}_serve.log"), specs[name])
        entry["verdict"] = verdict_for_config(cells)
        bundle["configs"][name] = entry
    bundle["ordering_stability"] = ordering_stability(per_cfg)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    write_evidence(args.out_dir, bundle)
    print(json.dumps(bundle["ordering_stability"], indent=2))
    print(f"WROTE {os.path.join(args.out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
