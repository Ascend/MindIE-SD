# Ascend 多卡通信：拓扑 / 并行选型 / 带宽验证 / 环境诊断

> 来源：MiniMax-H3 × LightX2V 实测（Ascend 950PR ×8 单机，2026-09）。结论在
> 「同拓扑、同卡组、同窗口」内成立；换机型/互联需按 §3 方法重验。
> 隐私说明：本文件不含主机 IP / 口令 / 用户名；路径用占位符（`{model_dir}`、`{container}` 等）。

## 目录

1. [拓扑读取与「UB 岛 / SYS 跨岛」判定](#1-拓扑读取与判据)
2. [并行选型依赖拓扑（bulk vs head-parallel 翻转）](#2-并行选型依赖拓扑)
3. [HCCL 带宽验证（等价工具姿势与坑）](#3-hccl-带宽验证)
4. [多卡环境诊断与恢复（端口泄漏 / 卡组健康）](#4-多卡环境诊断与恢复)
5. [复现与对比纪律](#5-复现与对比纪律)

## 1. 拓扑读取与判据

```bash
npu-smi info -t topo        # 全卡互连矩阵：UB / SYS / PXB / TBNA
npu-smi info -m             # Slot/Chip 映射（同 Slot 相邻卡倾向同岛）
```

- **UB** = HCCS 高速互连（同岛）；**SYS** = 跨 PCIe/NUMA（异岛）。异岛对带宽明显更低
  （实测 2 卡 alltoall 64MB：UB 对 ≈132GB/s vs SYS 对 ≈86GB/s，约 -35%）。
- 单机常见形态：上半岛 {0..3}、下半岛 {4..7} 各自全 UB，跨岛为 SYS。
- **判据**：4 卡 a2a 实验优先选**单 UB 岛**（如 {4,5,6,7} 或 {0,1,2,3}）；若某卡 Health
  Warning/Alarm 或驱动受损，只能用跨岛组（如 {0,1,5,6}）时，预期通信带宽次优——对比必须
  同组，勿跨组比绝对值。

## 2. 并行选型依赖拓扑

**不要无条件复用历史选型结论**；同一种并行形态在不同互联拓扑下排名可能反转（实测）：

| 拓扑 | 实测结论 | 证据 |
|---|---|---|
| 单 UB 岛（4 卡全 UB） | USP4 **bulk** 最优：comm busy 占比最小（~13%/block）、同步事件最少（~4750/步）；head-parallel 单头拆分使 a2a 次数×14、同步事件 ×12 → 回退 | CANN profile + 事件统计 |
| SYS 跨岛组（两对 UB + 跨岛链路） | **head-parallel 更优**：eager 30 步同窗口 2×2 复现，rank0 clean-window bulk 4.25s vs headpar 3.72/3.81s（**-11.5%**） | 墙钟 2×2 + 配对 CANN profile |

**机制（配对 profile 解释反转）**：跨岛组里 bulk 的「单次大 alltoall」在 SYS 链路上串行暴露
（通信 500 events/942ms 未掩盖）；head-parallel 的「大量小 alltoall」虽 kernel-sum 更高
（16000 kernels/8.7s vs bulk 3600/4.1s，逐头拆分 700 次 FA + 2850 次 HcclLaunch），但**全部
异步流水、与计算重叠** → 墙钟反而更低。判据：**墙钟/clean-window 为准，kernel-sum 不能直接比**
（跨流重叠会多计数）。

**compile × head-parallel 不兼容**（实测）：逐头 python 循环触发 Dynamo `recompile_limit`
（日志 `[5/8]`）→ 部分帧静默回退 eager；组合不要用于生产，head-parallel 建议 eager 形态。

**comm-stream masking（950PR + H3 bulk 形态）实测否决（2026-09）**：用真实 H3 shapes 的
4 卡微基准测「a2a 与 block 计算重叠」（异步 all_to_all_single + 独立流/事件姿势）：
单 block a2a 仅 7.9ms vs block 计算 25.0ms；重叠只隐藏 3.3ms/block（hidden_ratio 0.418 < 0.5
判据），折算每步理论最多 ~4.1% 且未计事件/流开销 → **低于阈值，不做**。机理：逐层数据依赖
下可重叠的独立计算少 + a2a 占比小（5s 步 comm 17% 且 compile 出图已把 a2a 留 eager 拿过 -35%）。
**换场景才重估**：15s/长序列 comm 占 kernel 57.6%，序列更长时 masking/切分收益随占比放大。
（910B dummy-run 曾报 masking -95% comm——那是 funcol 层 monkey-patch + 不同序列结构，勿迁移。）

## 3. HCCL 带宽验证

**首选官方 hccl_test**（CANN toolkit `tools/hccl_test/`，mpirun 驱动）：

```bash
mpirun --allow-run-as-root -n 4 ./bin/alltoall_test -b 1M -e 512M -f 2 -p 4   # 同 all_reduce/all_gather
```

已知坑：宿主直跑可能报 `HcclGetRootInfo failed / invalid data`（工具与 CANN/驱动组合不兼容）——
此时用**等价工具**：容器内 torchrun + torch_npu 微基准（与推理同 HCCL 栈）。

**等价工具（torchrun + torch_npu）关键姿势**：

- ⚠️ **必须 `torch_npu.npu.set_device(local_rank)` 后再 `init_process_group(backend="hccl")`**；
  缺 set_device 时 HCCL init 报「端口 already been bound」（各 rank 绑同端口），不是端口泄漏。
- 同一进程组内可依次测 all_to_all_single / all_reduce / all_gather_into_tensor，多尺寸（1MB→64MB，
  factor 2）取中位数；带宽口径注明（alltoall 按 per-rank 收发字节/时间）。
- 结果示例（4 卡跨岛组）：alltoall 64MB ≈ 90GB/s（1MB≈6GB/s 递增）；allreduce 64MB≈41GB/s；
  allgather ≈30GB/s。2 卡直连对比可量化 UB/SYS 差（见 §1）。

**隐含带宽校验（无工具时的替代证据）**：由 kernel profile 的 comm 总耗时与模型 shape 推算
per-call 载荷（seq-parallel a2a ≈ local_tokens × heads × head_dim × 2B），载荷/耗时 ≈ 隐含带宽；
与上述实测量级自洽即可判定「无异常超时/重传」。R18/R19 值：5s USP4 a2a 200-250 calls/步、
518-621ms/步 → 隐含 ~50GB/s（低于纯 alltoall 峰值，差额为 per-call host/notify 开销，正常）。

## 4. 多卡环境诊断与恢复

症状与处理（共享机多租户，先 npu-smi 再动刀）：

| 症状 | 可能根因 | 处理 |
|---|---|---|
| 某卡或整组 ~10× 变慢（~4s→44-60s/步，CPU 100% 而单算子 GEMM 正常） | 运行中 SIGKILL 多卡任务 → NPU/HCCL 驱动侧状态异常（可含端口句柄泄漏） | 停止该组实验；**换卡组验证**（同机其他健康卡组常可恢复）；必要时重启容器/复位 NPU（管理员） |
| npu-smi Health=OK 但整组多卡 run 均匀 ~43-44s/步（无热降频斜率） | SIGKILL 驱动损伤**残留态**：npu-smi 健康列看不出（2026-09 实测受损组跨天不自动恢复） | 组可用性必须用**真实多卡 run 验证并核对每步时长**（日志 per-step cost 行）：4 步 smoke「跑通出视频」仍可能每步 43s（时长异常=未通过）；要求驱动级复位后再用 |
| 任意新 HCCL init 报「端口 already been bound」（ss 看不到监听） | NPU 网卡 listen socket 泄漏（强杀/abort 累积）或**未 set_device**（见 §3） | 先查 set_device 姿势；再换全新端口段（如 30000+）；仍失败 → 驱动复位 |
| `hcclCommInitRootInfoConfig error` | ranktable/端口/设备未选 | 查 `/usr/local/Ascend/driver/topo/{chip}/*.json` 是否齐全（容器内常缺失→docker cp/挂载）；查端口段与 set_device |
| 卡 Health Warning/Alarm | 历史告警（可能残留） | 避免使用；同窗口同卡组纪律优先 |

工具：`npu-smi info -t topo/-m/-t temp`、`ss -lunp`、`dmesg | grep -i ascend`。
日志纪律：`session_work` 含凭据勿提交；报告不回显口令。

## 5. 复现与对比纪律

- 固定 rank0 口径（多卡各 rank 值不同）；per-step 用 p50 / 或 **clean-window（steps 2-14）avg**
  （机箱热时 30 步 run 常在 ~14-17 步后热降频 4→7-8s/步，与代码无关——降频口径见
  framework-feature-enablement `lightx2v-mindiesd-case.md` §8）。
- 同窗口同卡组、多跑取中位；单次结果不迁移。
- 并行形态切换先 4 步 smoke + CANN profile 核验 a2a 形态未退化（`hcom_alltoall` 等分而非
  `hcom_alltoallv` 变长），再 30 步墙钟。

## 6. 运行时通信分布采集（shim 法，无仓库改动；H3 USP2 实测案例 2026-09-09）

- **做法**：包装 `torch.distributed` 6 个 collective（all_reduce/all_gather/all_to_all_single/
  all_to_all/broadcast/reduce_scatter），逐 op 记 stage/op/shape/dtype/字节/world；阶段锚 = 各阶段
  模型入口 forward（DiT/VAE 的 `forward`/`decode_latent`；⚠️ text encoder 走 **`encode_ids`** 非 forward）；
  钩子时机 = meta-finder 拦 **`torch` 根导入**（torch.distributed 在 torch 包内被提前加载，独立拦截不稳），
  加载后强制 import 再 wrap；锚在首个 collective 懒安装；**同 prompt 输出被缓存 → 测 encoder 通信须每
  请求不同 prompt**。
- **字节口径**：tensor_bytes=op 本 rank 载荷；moved~ 估算（a2a/broadcast≈×(w-1)/w、all_reduce≈×2(w-1)/w、
  all_gather≈×(w-1)），非 HCCL 硬件计数；rank 对称则整簇移动≈2×per-rank（a2a 双方各半已计入）。
- **实测参考（H3 T2VA USP2，12 步裁剪）**：DiT Ulysses a2a ≈182 次/步、~28.4 GB/步（载荷占比 99.6%，
  通信主体）；text encoder TP all_reduce ~101 次/encode、~47MB；VAE video all_gather 14 次/请求、
  ~1.2GB/请求；audio VAE 无并行（无 DistributedVaeMixin）→ 0 通信。方法与明细见
  framework-feature-enablement `cache-dit-minimax-h3-case.md` §8。
