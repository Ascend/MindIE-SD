#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Large-shape accuracy and fixed-input stability stress for rms_norm_indexed_scale_shift."""

import argparse
import hashlib
from pathlib import Path
import tempfile

# Support namespace-package imports and direct script execution.
# pylint: disable=relative-beyond-top-level
if __package__:
    from .index_modulation_test_config import (
        EPS,
        SEED,
        add_scene_arguments,
        iter_scenes,
        print_scene_configuration,
        scene_distributions,
        scene_label,
        validate_scene_arguments,
    )
else:
    from index_modulation_test_config import (
        EPS,
        SEED,
        add_scene_arguments,
        iter_scenes,
        print_scene_configuration,
        scene_distributions,
        scene_label,
        validate_scene_arguments,
    )


def input_fingerprints(inputs):
    import torch

    return tuple(
        hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in inputs
    )


def report_stress_failure(baseline, current, iteration, distribution, seed=SEED):
    import torch

    if __package__:
        from .index_modulation_accuracy_common import _coordinate
    else:
        from index_modulation_accuracy_common import _coordinate

    changed = baseline.view(torch.int16) != current.view(torch.int16)
    a = current.float().flatten()
    g = baseline.float().flatten()
    finite = torch.isfinite(a) & torch.isfinite(g)
    diff = (a - g).abs()
    finite_diff = torch.where(finite, diff, torch.zeros_like(diff))
    denominator = torch.where(finite, g.abs().clamp_min(1e-12), torch.ones_like(g))
    relative_diff = finite_diff / denominator
    abs_index = int(finite_diff.argmax().item())
    rel_index = int(relative_diff.argmax().item())
    max_abs = float(finite_diff[abs_index].item())
    max_rel = float(relative_diff[rel_index].item())
    count = int(changed.sum().item())
    print(
        f"FAIL stress shape={tuple(baseline.shape)} dtype={baseline.dtype} seed={seed} "
        f"distribution={distribution} iteration={iteration} output=output "
        f"changed={count}/{baseline.numel()} changed_ratio={count / baseline.numel():.9g} "
        f"finite_max_abs={max_abs} index={_coordinate(abs_index, baseline.shape)} "
        f"finite_max_rel={max_rel} index={_coordinate(rel_index, baseline.shape)} "
        f"nonfinite={int((~finite).sum().item())}",
        flush=True,
    )
    scores = torch.where(finite, diff, torch.full_like(diff, float('inf')))
    scores.masked_fill_(~changed.flatten(), float('-inf'))
    for index in torch.topk(scores, min(8, count)).indices.tolist():
        print(f"  index={_coordinate(index, baseline.shape)} baseline={float(g[index])} current={float(a[index])}")
    log_root = Path(__file__).resolve().parents[5] / "log" / "rms_norm_indexed_scale_shift"
    log_root.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(tempfile.mkdtemp(prefix=f"stress-{distribution}-{iteration}-", dir=log_root))
    torch.save(
        {"iteration": iteration, "distribution": distribution, "seed": seed, "baseline": baseline, "current": current},
        dump_dir / "mismatch.pt",
    )
    print(f"Stress mismatch saved to {dump_dir / 'mismatch.pt'}", flush=True)
    raise AssertionError(f"output: stress output is not bit-exact at iteration {iteration}")


def _run_stress(inputs, runs, distribution, before, eps=EPS, seed=SEED):
    import torch
    from mindiesd import rms_norm_indexed_scale_shift

    if __package__:
        from .index_modulation_accuracy_common import _coordinate, check_output_contract
    else:
        from index_modulation_accuracy_common import _coordinate, check_output_contract

    output = rms_norm_indexed_scale_shift(*inputs, eps=eps)
    torch.npu.synchronize()
    check_output_contract(output, inputs)
    baseline = output.detach().cpu().clone()
    del output
    invalid = ~torch.isfinite(baseline)
    if bool(invalid.any()):
        count = int(invalid.sum().item())
        first = int(invalid.flatten().to(torch.int32).argmax().item())
        raise AssertionError(
            f"output: nonfinite stress baseline iteration=0 count={count}/{baseline.numel()} "
            f"first_index={_coordinate(first, baseline.shape)}"
        )
    del invalid
    if input_fingerprints(inputs) != before:
        raise AssertionError("Stress inputs changed during the baseline call")
    print(
        f"Stress shape={tuple(inputs[0].shape)} dtype={inputs[0].dtype} eps={eps} seed={seed} "
        f"distribution={distribution} runs={runs} baseline_calls=1 policy=bit_exact",
        flush=True,
    )
    for iteration in range(1, runs + 1):
        output = rms_norm_indexed_scale_shift(*inputs, eps=eps)
        torch.npu.synchronize()
        try:
            check_output_contract(output, inputs)
        except AssertionError as error:
            raise AssertionError(f"Stress iteration={iteration}: {error}") from error
        current = output.detach().cpu()
        if not torch.equal(current.view(torch.int16), baseline.view(torch.int16)):
            if input_fingerprints(inputs) != before:
                raise AssertionError(f"Stress inputs changed at iteration {iteration}")
            report_stress_failure(baseline, current, iteration, distribution, seed)
        del current, output
        if iteration % 100 == 0 or iteration == runs:
            print(f"Stress distribution={distribution} completed={iteration}/{runs}", flush=True)
    if input_fingerprints(inputs) != before:
        raise AssertionError("Stress inputs changed; output stability result is invalid")
    print(f"PASS stress distribution={distribution} runs={runs} output=output bit_exact", flush=True)


def run_stress(inputs, runs, distribution, eps=EPS, seed=SEED):
    if __package__:
        from .index_modulation_accuracy_common import log_inputs
    else:
        from index_modulation_accuracy_common import log_inputs

    print("DUT=rms_norm_indexed_scale_shift; stress baseline is independent of golden")
    log_inputs(*inputs, eps=eps)
    before = input_fingerprints(inputs)
    try:
        _run_stress(inputs, runs, distribution, before, eps, seed)
    except (AssertionError, RuntimeError) as error:
        print(
            f"FAIL stress shape={tuple(inputs[0].shape)} dtype={inputs[0].dtype} "
            f"device={inputs[0].device} eps={eps} seed={seed} distribution={distribution}: {error}",
            flush=True,
        )
        if input_fingerprints(inputs) != before:
            raise AssertionError("Stress inputs changed; output stability result is invalid") from error
        raise


def main():
    parser = argparse.ArgumentParser()
    add_scene_arguments(parser)
    parser.add_argument("--mode", choices=("all", "accuracy", "stress"), default="all")
    parser.add_argument("--distribution", choices=("all", "uniform", "normal"), default="all")
    parser.add_argument(
        "--stress-runs",
        type=int,
        default=1000,
        help="Repeated DUT calls per distribution, excluding the baseline (default: 1000)",
    )
    args = parser.parse_args()
    if args.stress_runs <= 0:
        parser.error("--stress-runs must be a positive integer")
    validate_scene_arguments(parser, args)
    if print_scene_configuration(args, accuracy=True):
        return
    if __package__:
        from .index_modulation_accuracy_common import (
            PROD_DTYPE,
            call_dut,
            compare,
            golden,
            init_device,
            make_inputs,
        )
    else:
        from index_modulation_accuracy_common import (
            PROD_DTYPE,
            call_dut,
            compare,
            golden,
            init_device,
            make_inputs,
        )

    device = init_device(args.device_id)
    stages = ("accuracy", "stress") if args.mode == "all" else (args.mode,)
    for stage in stages:
        for scene in iter_scenes(args):
            for distribution in scene_distributions(args, scene):
                print(f"stage={stage} scene={scene_label(scene)} distribution={distribution}", flush=True)
                inputs = make_inputs(
                    scene.rows, scene.hidden_size, scene.num_groups, PROD_DTYPE, device, distribution, seed=scene.seed
                )
                if stage == "stress":
                    run_stress(inputs, scene.stress_runs, distribution, scene.eps, scene.seed)
                else:
                    before = input_fingerprints(inputs)
                    actual = call_dut(*inputs, eps=scene.eps)
                    assert input_fingerprints(inputs) == before, "Accuracy inputs changed during DUT call"
                    expected = golden(*inputs, eps=scene.eps)  # Reference stays on the selected NPU.
                    compare(actual, expected, large=True)
                    print(
                        f"PASS production {scene_label(scene)} distribution={distribution}; "
                        "mixed tolerance is diagnostic only"
                    )
                    del actual, expected
                del inputs


if __name__ == "__main__":
    main()
