# QuantFlashAttn 单算子测试

测试直接调用 `torch.ops.mindiesd.quant_flash_attn`，覆盖 MXFP4 E2M1、32 元素量化组、E8M0 scale、非因果注意力和 BF16 输出。运行环境需提供支持该算子的 Ascend NPU、匹配的 CANN / torch_npu 以及由当前源码构建的 MindIE-SD 算子与 plugin。

## 场景和参考结果

| 测试 | 场景 | 对照 |
|------|------|------|
| 小 shape | B=2，Q/KV 头数=4/2，D=128；物理长度与有效长度组合见下表；BNSD 和 BSND | 同量化输入的 CPU 阶段模拟 |
| 大 shape BNSD | Q=[1,32,2048,128]，K/V=[1,4,32768,128] | NPU 解码相同量化输入，以 FP32 小算子拼接 online attention，最终输出 BF16 |
| 大 shape BSND | Q=[1,2048,32,128]，K/V=[1,32768,4,128] | 同上，输入直接按 BSND 造数 |

小 shape 的每行分别覆盖两种 layout 和 uniform / normal，共 32 条真实算子用例。全部显式传入 `seqused_q` / `seqused_kv`：第一个 batch 使用表中有效长度，第二个 batch 的有效 Q/KV 长度为 65/129。仅传 `max_seqlen`、省略 `seqused` 的路径不属于本测试套件的验收范围。

| 用例 ID | 物理 Q/KV 长度 | 有效 Q/KV 长度（第一个 batch） | 覆盖点 |
|---------|----------------|---------------------------------|--------|
| native-q128-kv256 | 128/256 | 128/256 | 整 tile |
| native-q129-kv257 | 129/257 | 129/257 | 非对齐物理长度、Q/KV 尾块 |
| native-q129-kv513 | 129/513 | 129/513 | 非对齐物理长度、多 KV tile |
| pad512-q128-kv256 | 512/512 | 128/256 | 物理补齐，整 tile 有效长度 |
| pad512-q129-kv257 | 512/512 | 129/257 | 物理补齐，非对齐有效长度 |
| pad512-q511-kv511 | 512/512 | 511/511 | 512 边界前一元素 |
| pad512-q512-kv512 | 512/512 | 512/512 | 完整 512 边界 |
| pad512-q513-kv513 | 1024/1024 | 513/513 | 跨 512 边界，物理补齐至 1024 |

输入直接按物理 shape 和 layout 生成，未使用区域也保留同分布随机值，避免错误读取 padding 被零值掩盖；有效长度仅由接口属性指定。MXFP4 编码所需的 scale 尾组补齐不改变逻辑输入长度。对每个 batch 的有效 Q 行单独检查精度，再检查完整原始输出；无效 Q 行必须为零，避免大量零 padding 稀释精度指标。测试范围为 BNSD/BSND，不包含 TND；这是底层算子测试，不经过自动 padding 的 Python layer。

每个场景分别运行 uniform 和 normal 两条用例。uniform 为 U[-5,5]；normal 为每个输入独立采样均值 [-5,5]、标准差 [0.1,2] 后生成，不钳位。默认 seed=21，日志打印实际分布参数、输入 shape、dtype、stride、device 和算子属性。

CPU reference 模拟 FP16 分数、概率 E2M1 量化与 E8M0 scale；硬件指数、归约顺序及分块可能产生差异。大场景对照不模拟内部概率 FP4 量化，因此比较包含该阶段的量化损失。

## 精度与稳定性判据

`attention_out` 约定包含内部 E2M1 概率量化，采用本地 FLOAT4 E2M1 暂定档：atol=rtol=0.5，匹配率至少 99%，最大绝对误差不超过 `max(1.0, 32×ULP)`。ULP 取最大误差点 golden 在实际 BF16 输出格式中的间距。参数集中在 `qfa_tolerance.py`；此档位适用性仍需上板验证。

小 shape 的混合容差、cosine≥0.99、模长比 [0.9,1.1] 均为硬门禁。大 shape 仅 cosine 和模长比为硬门禁，混合容差计算并打印为诊断项。第二个输出 `softmax_lse` 按关闭 LSE 的接口契约检查为空 FP32；所有输出检查 shape、dtype、设备、连续性和有限值。

压测在独立基线调用之后，对每种 layout / 分布组合额外重复 1000 轮，逐位比较全部输出（含正负零），并检查输入摘要未改变。单独压测通过只证明重复执行稳定性。

在仓库根目录运行小 shape（固定物理卡 0）：

```bash
unset ASCEND_RT_VISIBLE_DEVICES
python3 -m pytest -s tests/ops/quant_flash_attn/test_qfa_accuracy.py
```

只运行物理长度按 512 对齐的 BSND 回归，可在上述 pytest 命令后加 `-k "pad512 and BSND"`；非 512 对齐物理长度的场景可用 `-k "native"` 选择。所有场景的长度传参标识均为 `used`。

大 shape 默认 `--layout all`，先完成 BNSD、BSND 各两种分布的 4 条单次精度，再执行 4 组压测。两种 layout 的有效长度均等于物理长度，显式传入 `seqused_q/kv`；量化格式、门禁和造数范围相同。可用 `--layout BNSD` 或 `--layout BSND` 单选：

```bash
bash tests/ops/quant_flash_attn/run_qfa_prod_accuracy.sh --device-id 1
bash tests/ops/quant_flash_attn/run_qfa_prod_accuracy.sh --device-id 1 --mode accuracy
bash tests/ops/quant_flash_attn/run_qfa_prod_accuracy.sh --device-id 1 --mode stress --stress-runs 1000
bash tests/ops/quant_flash_attn/run_qfa_prod_accuracy.sh --device-id 1 --layout BSND --mode accuracy
```

省略 `--device-id` 时使用仓内选卡工具；shell 清除 `ASCEND_RT_VISIBLE_DEVICES`，Python 在导入 MindIE-SD 前设置物理设备。直接调用 Python 时须自行清除该环境变量。

大 shape 报告默认保存到仓库 `log/quant_flash_attn/accuracy_<timestamp>/`，可用 `--output-dir` 指定新的目录。各场景报告分别位于 `<场景名>/accuracy_uniform.json` 和 `<场景名>/accuracy_normal.json`，场景名包含 layout、shape 和量化格式。全部所选项目成功后才输出逐项 PASS 和根目录的 `final_summary.txt`；任一失败以非零码退出。

## 性能采集

```bash
bash tests/ops/quant_flash_attn/run_qfa_msprof_op.sh --device-id 1 --path all
bash tests/ops/quant_flash_attn/run_qfa_msprof_op.sh --device-id 1 --layout BSND
```

只有 QuantFlashAttn 一条路径，`--path qfa` 与 `all` 等效。默认 `--layout all`，场景与大 shape 精度相同，BNSD、BSND 各自采集 uniform、normal。每轮 `msprof op` 默认预热 20 次，仅采集一次目标 kernel，开启 `--kill=on`；每种 layout / 分布组合独立采集 20 轮，默认共 80 轮。可通过 `MSPROF_WARMUP` 和 `TEST_COUNT` 配置次数。

其他场景参数放在 `--` 后透传，例如 `-- --q-seq 2048 --kv-seq 32768`；原有 `-- --layout BSND` 写法仍可使用。单次 workload `profile_qfa_prod.py` 仅运行一种 layout，默认 BNSD；两种 layout 的批量采集由上述 shell 入口执行。

产物默认保存到仓库 `log/quant_flash_attn/perf_<timestamp>/`，可用 `--output-dir` 指定新目录。每轮日志位于 `<场景名>/<uniform或normal>/run_NN/msprof.log`，每份日志必须恰好包含一个正且有限的 `Task Duration(us)`。末尾先连续输出所有所选场景的逐轮数据，再集中输出各场景最小值及波动说明，同时在根目录保存 `final_summary.txt` 和 `task_duration_summary.txt`。单选 layout 时不读取另一 layout 的日志；任一采集失败不发布完整成功汇总。

最小值代表当前测试条件下观测到的最佳性能，不代表波动已消除。若波动超过 10%，考虑将预热提高到 50 或 100，在相同卡和输入下使用新目录完整重采 20 轮；不能混合不同预热配置的结果。
