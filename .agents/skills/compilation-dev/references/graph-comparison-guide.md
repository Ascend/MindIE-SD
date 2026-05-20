# 双图对比操作指南

Phase 5 中对比 pattern 定义图 [A] 与模型 traced 图 [B] 的操作步骤。

---

## 输出目录结构

每次 debug 采集使用独立目录，文件按来源前缀命名：

```text
.tmp/attempt-{N}/
├── wan_debug_full.log
├── PATTERN-{PatternName}.txt
├── MODEL-{model}-{op}-subgraph.txt
└── report.txt
```

**命名规则**：

- `PATTERN-` → pattern 定义图（`make_fx` trace 或 `PatternPrettyPrinter` 输出）
- `MODEL-{model}-{op}` → 模型图目标算子子图（从 `Graph before compiling` 截取）

**示例**：

- `PATTERN-WanRmsNormPattern.txt` — Wan RMSNorm pattern 的 FX graph
- `MODEL-wan-rmsnorm-subgraph.txt` — Wan 模型图中 RMSNorm 子图

---

## 步骤 1: 采集模型 traced 图 [B]

在远端 NPU 容器中运行 `--compile --debug-graph`:

```bash
docker exec <container> bash -lc "
cd /home/<user>/workspace/MindIE-SD_pattern/examples/dummy_run &&
python wan_infer.py --compile --debug-graph 2>&1 | tee /tmp/model_debug.log
"

# 下载日志到本地
sftp get /tmp/model_debug.log
```

**日志关键段**:

- `Graph before compiling` → 模型完整 FX graph（`mindie_sd_backend.py:152`）
- `PatternMatchPass replace N patterns` → 命中数（`pattern_match_pass.py:66`）
- `Graph after pattern matching` → pattern 替换后的 graph（`mindie_sd_backend.py:115`）

**提取模型 graph 子图**:
在日志中搜索目标算子名:

- RMSNorm → `rms_norm` 或 `pow` + `mean`
- AdaLN → `native_layer_norm` 或 `LayerNorm`
- RoPE → `apply_rotary_emb` 或 `unbind`

每个子图通常以一段连续的 FX node 序列出现，以 source location 注释分隔。

---

## 步骤 2: 逐节点对齐

将 pattern 的节点序列与模型 graph 中的目标子图逐节点对比。

**对齐规则**:

1. 按顺序比较节点类型（`aten.pow`, `aten.mean`, `aten.mul` 等）
2. 比较参数值（args 和 kwargs）
3. 比较 dtype 路径（每个节点的输出 dtype）
4. 第一个不匹配的节点 = 差异所在

**示例对齐** (Wan2.2 RMSNorm):

```text
节点  Pattern                   Model Graph                  匹配?
 1    in: x (fp32)              view_15 (f32)                ✅ dtype 一致
 2    pow(x, 2)                 pow(view_15, 2)              ✅
 3    mean(pow, [-1], True)     mean(pow, [2], True)         ❌ [-1] != [2]! → 类型 5
 4    add(mean, 1e-6)           add(mean, 9.9999...e-07)    ❌ 标量不等! → 类型 6
```

发现 mismatch 后，对照 `mismatch-catalog.md` 选择修复策略。
修复后在远端重新部署并验证（Phase 6）。

---

## 工具链一览

| 工具 | 用途 | 输出 |
|------|------|------|
| `--debug-graph` | 采集模型 graph + 命中数 | 完整 FX graph 文本 |
| `TORCH_COMPILE_DEBUG=1` | Dynamo 层额外信息 | Guard 检查、decomposition 详情 |
| `kernel_details.csv` | 确认融合 kernel | CANN Profiler kernel 清单 |

## 维护与更新

当 debug 工具链有新的可用选项或输出格式变化时更新此文件。
