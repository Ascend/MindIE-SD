# tests/tools

给各算子测试 / `msprof` 采集脚本共用的宿主机工具。不要放进 `tests/utils/`：那是 `mindiesd.utils` 的单测和 pytest 夹具（`from utils.utils...`）。

| 文件 | 用途 |
|------|------|
| `select_npu_device.py` | 解析 `npu-smi info`，打印物理 NPU ID 供 `--device-id` 使用 |

选卡顺序：先选没有执行任务的卡（进程数为 0；多张则 HBM 最小）。没有无任务卡时，选 NPU Util 为 0 且 HBM 最小的卡，并在终端打印 `没有选到没有任务的卡，选了 NPU x`。仍没有 Util=0 的卡则在剩余卡里选 HBM 最小，同样打这条提示。`--format=id` 的 stdout 只有卡号，提示走 stderr。

```bash
python3 tests/tools/select_npu_device.py --format=report
python3 tests/tools/select_npu_device.py --format=id
```

不要设置 `ASCEND_RT_VISIBLE_DEVICES`（与 `msprof` / `msprof op` 不兼容）。
