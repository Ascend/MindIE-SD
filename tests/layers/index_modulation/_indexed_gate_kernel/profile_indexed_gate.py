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
"""One real indexed_gate call; numerical validation belongs to the accuracy entry."""

import argparse
import json
import os

import indexed_gate_cases as cases
import indexed_gate_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-id', type=int, required=True)
    parser.add_argument('--scenario', choices=[case.name for case in cases.ALL_PERFORMANCE], required=True)
    parser.add_argument('--msprof-mode', action='store_true')
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error('device-id must be nonnegative')
    dut, device = common.initialize(args.device_id)
    print(json.dumps(dict(stage='CPU input generation', pid=os.getpid(), device_id=args.device_id)), flush=True)
    case = cases.production_case(args.scenario)
    _, tensors, _, metadata = common.make_inputs(case, device)
    print(json.dumps(dict(stage='before_dut', wrapper_calls=1, **metadata)), flush=True)
    returned = common.call(dut, tensors)
    output_contract = common.contract(returned, tensors)
    assert all(output_contract.values()), output_contract
    print(json.dumps(dict(stage='call returned', output_contract=output_contract)), flush=True)
    common.torch.npu.synchronize()
    print(json.dumps(dict(stage='synchronization completed', source_sha256_after=common.sha(common.DUT))), flush=True)


if __name__ == '__main__':
    main()
