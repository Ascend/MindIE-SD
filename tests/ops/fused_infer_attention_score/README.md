# fused_infer_attention_score 测试

本目录按“小 shape CPU Golden、大 shape NPU Golden、大 shape `msprof op`”三层验证 FIA FP8 量化路径。四条被测路径共用 `fia_quant_common.py`，防止精度与性能脚本的 quant mode、block 或 `inner_precise` 不一致。

## 被测路径

| 路径 | Q/K/V quant mode | `inner_precise` | V 量化块 | 说明 |
|------|------------------|-----------------|----------|------|
| `original` | `7/7/7` | `0` | `256x128` | K256/V256 原始路径，性能基线 |
| `c8v16` | `7/7/7` | `4` | `256x128` | K256/V256 C8V16 |
| `v512` | `7/7/11` | `4` | `512x128` | K256/V512x128 C8V16 |
| `v512_d64` | `7/7/12` | `4` | `512x64` | K256/V512x64 C8V16 |

V512 两条路径必须使用 `inner_precise=4`。

## 常用测试入口

| 文件 | 用途 |
|------|------|
| `test_fused_infer_attention_score_v2.py` | NPU 小 shape 输出和四路径 CPU Golden 对比 |
| `run_fused_infer_attention_score_prod_accuracy.sh` | 大 shape 精度、固定输入压测及续跑 |
| `run_fused_infer_attention_score_msprof_op.sh` | 两套 shape、四条路径分别启动 msprof op |

两个大 shape 入口调用 `fia_prod_*.py` 完成场景筛选、造数、执行和结果归档。四条量化路径的性能采集共用 `profile_fia_dit_quant.py`。

## 第一层：小 shape CPU Golden

- layout：BNSD
- 3 个零 V 用例检查 D64/D128、FP16/BF16 输出格式及全零结果。
- 9 个 CPU Golden 用例覆盖 MHA/GQA。四条量化路径均保留 Q128、KV513 的 GQA 回归；另选 5 组边界，Q/KV 长度分别为 127/255、128/256、129/257、127/511 和 129/512。
- 量化前：BF16；量化后：FP8 E4M3；输出：BF16
- `513` 同时覆盖完整 V512 tile 和尾块
- DUT 与 Golden 共用量化后的 Q/K/V 和 scale
- CPU Golden 按 C8V16 的 FP16 online-softmax、P 的 FP8 RNA 和 LastDiv 阶段计算

FP8 Cube MMA 无法由 CPU FP32 `matmul` 位级复刻，因此逐元素门禁使用 FLOAT8 E4M3 混合容差：`rtol=0.25`、`atol=0.0625`、匹配率至少 `0.99`、全张量最大绝对误差不超过 `4`。同时要求 cosine ≥ `0.99`、norm_ratio ∈ `[0.9,1.1]`。

```bash
python -m pytest --test-mode=NPU tests/ops/fused_infer_attention_score/test_fused_infer_attention_score_v2.py -q
```

## 第二层：大 shape NPU Golden

大 shape 精度与性能共用以下两套尺寸，均采用 BNSD、FP8 E4M3 Q/K/V、FP32 scale 和 BF16 输出。每套覆盖四条量化路径，共 8 个场景；小 shape 不随此集合展开。

| 场景 | Q / attention_out | K / V |
|------|-------------------|-------|
| GQA，默认顺序第一套 | [1,32,2304,128] | [1,4,30757,128] |
| MHA，长序列 | [1,40,75600,128] | [1,40,75600,128] |

输入在 CPU 按实际尺寸生成，按各路径的 block 编码为 FP8，不补齐物理 shape。精度在 8 个场景内交叉安排正态和均匀分布。正态逐输入、固定 seed 采样 mean ∈ [-5,5]、std ∈ [0.1,2]，不钳位样本；均匀分布采用 [-5,5]。性能使用正态分布，各轮保持相同 seed 和输入。

Golden 为同一份量化前 BF16 输入上的 NPU npu_fusion_attention，包含量化误差。硬门禁为 cosine ≥ 0.99、norm_ratio ∈ [0.9,1.1]；逐元素 FP8 E4M3 档的 rtol=atol=0.125、匹配率 0.99、最大误差上限 max(1,32×实际 BF16 ULP) 作为软约束打印，不改变大 shape 的通过判定。

```bash
unset ASCEND_RT_VISIBLE_DEVICES

# 无设备预览，默认两套 shape × 四路径
bash tests/ops/fused_infer_attention_score/run_fused_infer_attention_score_prod_accuracy.sh --dry-run

# 只跑 75600 场景的单次精度；可用多个物理卡按完整场景并行
bash tests/ops/fused_infer_attention_score/run_fused_infer_attention_score_prod_accuracy.sh \
  --device-id 1 3 --query-seq-len 75600 --mode accuracy

# 默认先完成全部单次精度，再固定输入各压测 1000 轮
bash tests/ops/fused_infer_attention_score/run_fused_infer_attention_score_prod_accuracy.sh --device-id 3
```

用 --path 选择单一路径，--mode accuracy|stress|all 选择阶段，--stress-runs 设置压测轮数。压测独立保存 baseline，逐轮位级比较输出；单独 stress 通过不代表 golden 精度通过。日志实时显示阶段与轮次，长设备调用期间不推算进度。输入及 baseline 的重建、校验和提交耗时属于主机时间，不作为内核性能。

## 第三层：大 shape `msprof op`

性能使用同一份尺寸、dtype 和路径定义。每轮独立启动 Python，只调用一次 EagleFusedInferAttentionScore，由 msprof op 记录 Task Duration(us)。950 默认采集 3 轮，910B 默认 20 轮；同批次固定一张物理卡。

```bash
unset ASCEND_RT_VISIBLE_DEVICES

bash tests/ops/fused_infer_attention_score/run_fused_infer_attention_score_msprof_op.sh \
  --device-id 3 --query-seq-len 75600 --path v512_d64
```

默认预热按 max(Q长度,KV长度) 分档：小于 30000 为 20 次，30000～49999 为 2 次，达到 50000 为 1 次。因此两套场景分别预热 2 次和 1 次。可显式传 --warmup（允许 0）、--runs。逐轮保留原始产物并汇总最小值，它表示当前条件下观察到的最佳性能；同场景波动超过 10% 时可增加预热，在新批次完整复测。

## 日志、中断与续跑

性能批次的 `final_summary.txt` 与终端均先列出所有逐轮耗时，再以独立表格集中列出每个所选场景的最小 Task Duration（µs）及有效轮数。部分批次也列出所选场景，尚无有效样本的场景用 `-` 表示。

默认产物位于仓根 `log/fused_infer_attention_score/accuracy_时间戳/` 或 `performance_时间戳/`。每个已完成单元立即原子保存；Ctrl+C、SIGTERM 和执行失败输出 PARTIAL 汇总，正常完成输出完整汇总。SIGKILL 后只能在下次启动恢复已提交进度。

```bash
bash tests/ops/fused_infer_attention_score/run_fused_infer_attention_score_prod_accuracy.sh \
  --resume /absolute/path/to/accuracy_batch --device-id 1 3
bash tests/ops/fused_infer_attention_score/run_fused_infer_attention_score_msprof_op.sh \
  --resume /absolute/path/to/performance_batch
```

默认启动总是新批次，显式输出目录已存在则拒绝覆盖。续跑校验原计划、代码、算子产物和软件环境，跳过完整提交的单元；已记录的 FAIL 不自动重测。精度允许显式更换卡列表，保留历史记录与 baseline；性能必须保持原卡。最终 `final_summary.txt` 包含历史及本次结果。

## 一分钟指令仿真

Host 调度分组可先独立验证，覆盖 MHA/GQA 的任务完整性和启用边界：

```bash
bash csrc/ops/fused_infer_attention_score/tests/ut/run_fia_arch35_ut.sh --schedule-only
```

此单测使用独立可执行文件，避免旧版 tiling 类与 arch35 同名结构的链接符号冲突。默认日志位于 `log/fused_infer_attention_score/host_ut/`。

先构建带 -g 的 ascend950 内核并检查调试行表，再在匹配的仿真环境运行第一套默认场景。CPU 输入编码完成后拷贝到设备，目标调用后仅检查元数据。

```bash
PYTHONPATH="$PWD" msprof op simulator --soc-version=Ascend950PR_9599 \
  --kernel-name=EagleFusedInferAttentionScore --core-id="0|1" \
  --aic-metrics=PipeUtilization,ResourceConflictRatio --timeout=1 --launch-count=1 \
  --output="$PWD/log/fused_infer_attention_score/simulator_default" \
  python3 -u tests/ops/fused_infer_attention_score/profile_fia_dit_quant.py --path original --device-id 0
```

超时后的部分流水只证明已执行区间，须检查目标指令和源码映射。默认 GQA 工作集较小，走原调度；MHA 75600 用于验证 L2 调度分支，不能用默认场景仿真代替它的精度或性能验证。

## 选卡约束

不要设置 `ASCEND_RT_VISIBLE_DEVICES`。两个 shell 入口会清除此变量，并在未传 `--device-id` 时调用 `tests/tools/select_npu_device.py` 选择物理卡；Python 内使用 `torch.npu.set_device(device_id)`。
