# Benchmark 数据异常排查

本文件记录 MindIE-SD 核心算子基准（FA/BSA/GMM/MM）实际遇到的数据异常案例与排查方法。
加载时机：benchmark 数据异常（latency 恒定、输出全零、MFU 钳位 1、偶发污染、增量合并出错）时。

## 目录

- 排查总纲：算子问题 vs benchmark 调用问题
- 案例 A：BSA 恒定 latency + 全零输出（inner_precise / mask）
- 案例 B：偶发卡顿污染（异常大 latency）
- 案例 C：MFU 钳位 1（记账与执行不符）
- 案例 D：增量合并陷阱（jsonl 覆盖）
- 通用诊断清单

---

## 1. 排查总纲：算子问题 vs benchmark 调用问题

数据异常时**第一步先跑算子自身 UT**，用结果分流：

```bash
# 远端容器内跑算子 UT（tests/plugin/test_<op>.py）
docker exec <容器> bash -lc 'cd /home/blian/code/MindIE-SD && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  python -m pytest tests/plugin/test_block_sparse_attention.py -v'
```

| UT 结果 | 结论 | 下一步 |
|---|---|---|
| 全部 PASSED | 算子正常 → **benchmark 调用问题** | 对照 UT 与 vendor 实现的调用参数（inner_precise / mask / actual_seq_lengths / shape） |
| FAIL / 崩溃 | 算子或环境问题 | 查 CANN 版本、算子编译产物、设备支持（UT 常带设备 skip 条件如 A5） |

**注意**：UT 通过不代表所有 shape 可用——UT 可能只覆盖小序列/特定参数。
若怀疑 shape 依赖，用与 benchmark 相同的 shape 直接调用算子验证（见案例 A 的验证脚本模式）。

## 2. 案例 A：BSA 恒定 latency + 全零输出（inner_precise / mask）

**现象**：BSA 所有 case latency 恒定（~60us，q_len 1024→32768 不变），MFU 爆表钳位 1。

**排查链**（按序执行）：

1. 直接调用算子，检查输出有效性：

   ```python
   out, _ = block_sparse_attention(q, k, v, block_sparse_mask=mask, ...)
   torch.npu.synchronize()
   print((out != 0).float().mean().item())   # 0.0 = kernel 空转，输出全零
   ```

2. 对照算子 UT 的调用参数，发现差异：
   - **`inner_precise`**：950 系列设备（如 Ascend950PR / A310 报 950PR 名）要求 `inner_precise=4`（UT 注释明确 "op vendor requirement"）；**用 0 会导致 kernel 输出全零、latency 恒定**（空转）。按设备名自适应：`4 if "950" in dev_name else 1`
   - **mask 构造**：`mask.view(-1)[:keep] = 1`（前 keep 块）使**后部 query 行全零**（无 attend 块）→ 崩溃。改为 **per-row uniform**：`mask[..., :per_row] = 1`（每行保留 per_row 个 kv 块），与 UT 的随机 mask 一样保证每行有 attend 块
3. 修复后验证：latency 应随 q_len/sparsity 增长（sp=0.99 远快于 sp=0.6），输出非零。

**教训**：benchmark 的 vendor 实现必须与算子 UT 的调用约定对齐（参数值、mask 布局、shape 约束）。
任何"恒定 latency"都应怀疑 kernel 未执行，先查输出有效性再查计时。

## 3. 案例 B：偶发卡顿污染（异常大 latency）

**现象**：同一 case 偶发 latency 异常（如 1024 sp=0.6 正常 49us，某次跑到 8.4 秒）。

**处理**：

1. 确认是偶发：增量重跑该 case（精确 `--config`），对比正常值
2. 正常则**只补跑该 case**，结果并入原 report-dir（同 slot 最新覆盖），不必全量重跑
3. 数据有效性守卫（代码层）：vendor 首次调用后校验输出非零，全零即抛 RuntimeError → case 标记无效不出数据；`collect_baseline` 过滤无有效测量的 entry

## 4. 案例 C：MFU 钳位 1（记账与执行不符）

**现象**：MFU 恒为 1.0（钳位值），尤其高稀疏/大序列档。

**判断**：MFU = 记账 FLOPs / 实测时间 / peak。三个来源都核对：

1. **latency 是否真实**（见案例 A：恒定 = 假数据）
2. **记账 FLOPs 是否与 kernel 实际执行一致**：BSA 的 `(1-sparsity)` 折扣是理论值，与真实块稀疏 kernel 的执行量可能不符
3. **peak_flops 是否准确**：CUBE 峰值需设备实测（如 A310 用 425/9*8≈377.78）；峰值由使用者通过 `--config` 输入，随 case arguments 进 jsonl，离线 report 从 entry arguments 读，无需设备名匹配

钳位是"兜底"：钳位发生在 `util_metrics` 公式层（MFU/MBU 先算比值再 min(≤1)），数据层保持真实比值。**钳位频繁出现 = 上面某处口径有问题，应修复而非接受**。

## 5. 案例 D：增量合并陷阱（jsonl 覆盖）

**现象**：增量补跑后 report 里某些 slot 仍显示旧值/缺失。

**陷阱**：

1. **同 report_dir 多次 run 会覆盖 jsonl**（xpu-perf 写固定路径 `NPU/<device>/<op>/NPU/<op>-NPU.jsonl`）——每次补跑用独立 report_dir
2. **mtime 决定覆盖顺序**：`load_report_entries` 按 mtime 排序，collect 时同 slot 后者覆盖——**新补的数据必须 mtime 更新**，且不能引入坏值覆盖好值（补跑前确认新值正确）
3. report 用 `--report-dir` 指向**父目录**（rglob 递归合并所有 run 子目录）

**正确流程**：主 run 一个 report_dir，每次补跑独立新目录，report 时指向父目录合并。

## 6. 通用诊断清单

- [ ] latency 是否随问题规模（q_len/M/num_tokens）增长？恒定 → 固定开销污染或 kernel 空转
- [ ] 输出是否非零/有效？全零 → kernel 未执行（查 inner_precise/mask/参数）
- [ ] 算子自身 UT 是否通过？通过 → benchmark 调用问题；失败 → 算子/环境问题
- [ ] 调用参数是否对齐 UT（inner_precise / actual_seq_lengths / block_shape / mask 布局）？
- [ ] 偶发异常值是否被增量重跑确认/覆盖？
- [ ] 长序列是否配了 `--config {timeout: 300}`（默认 5s 会超时跳过大档位）？
- [ ] MFU 钳位 1 是否因 peak 口径/记账/假 latency 导致（而非接受钳位）？
