# Ascend 多卡通信：拓扑 / 并行选型 / 带宽验证 / 环境诊断

> 来源：MiniMax-H3 × LightX2V 实测（Ascend 950PR ×8 单机，2026-09）。结论在
> 「同拓扑、同卡组、同窗口」内成立；换机型/互联需按 §3 方法重验。
> 隐私说明：本文件不含主机 IP / 口令 / 用户名；路径用占位符（`{model_weight_dir}`、`{container}` 等）。

## 目录

1. [拓扑读取与「UB 岛 / SYS 跨岛」判定](#1-拓扑读取与判据)
2. [并行选型依赖拓扑（bulk vs head-parallel 翻转）](#2-并行选型依赖拓扑)
3. [HCCL 带宽验证（等价工具姿势与坑）](#3-hccl-带宽验证)
4. [多卡环境诊断与恢复（端口泄漏 / 卡组健康）](#4-多卡环境诊断与恢复)
5. [复现与对比纪律](#5-复现与对比纪律)
6. [运行时通信分布采集（shim 法，无仓库改动）](#6-运行时通信分布采集shim-法无仓库改动h3-usp2-实测案例-2026-09-09)
7. [通信占比口径：按并行策略与特性叠加重估](#7-通信占比口径按并行策略与特性叠加重估方法增量-2026-09-12)
8. [维护与更新](#8-维护与更新)

## 1. 拓扑读取与判据

```bash
npu-smi info -t topo        # 全卡互连矩阵：UB / SYS / PXB / TBNA
npu-smi info -m             # Slot/Chip 映射（同 Slot 相邻卡倾向同岛）
```

- **UB** = HCCS 高速互连（同岛）；**SYS** = 跨 PCIe/NUMA（异岛）。异岛对带宽明显更低
  （实测 2 卡 alltoall 64MB：SYS 对显著低于 UB 对，约低三成；绝对带宽读数见归档
  `{run_results_dir}/archive/`）。
- 单机常见形态：上半岛 {0..3}、下半岛 {4..7} 各自全 UB，跨岛为 SYS。
- **判据**：4 卡 a2a 实验优先选**单 UB 岛**（如 {4,5,6,7} 或 {0,1,2,3}）；若某卡 Health
  Warning/Alarm 或驱动受损，只能用跨岛组（如 {0,1,5,6}）时，预期通信带宽次优——对比必须
  同组，勿跨组比绝对值。

## 2. 并行选型依赖拓扑

**不要无条件复用历史选型结论**；同一种并行形态在不同互联拓扑下排名可能反转（实测）：

| 拓扑 | 实测结论 | 证据 |
|---|---|---|
| 单 UB 岛（4 卡全 UB） | USP4 **bulk** 最优：comm busy 占比最小（一成量级/block）、同步事件最少（数千/步）；head-parallel 单头拆分使 a2a 次数与同步事件各增一个数量级 → 回退 | CANN profile + 事件统计 |
| SYS 跨岛组（两对 UB + 跨岛链路） | **head-parallel 更优**：eager 同窗口 2×2 复现，rank0 clean-window 明显更快（本组合观测，读数见归档 `{run_results_dir}/archive/`） | 墙钟 2×2 + 配对 CANN profile |

**机制（配对 profile 解释反转）**：跨岛组里 bulk 的「单次大 alltoall」在 SYS 链路上串行暴露
（通信事件未被掩盖）；head-parallel 的「大量小 alltoall」虽 kernel-sum 更高
（逐头拆分带来更多 FA 与 HcclLaunch 次数），但**全部
异步流水、与计算重叠** → 墙钟反而更低。判据：**墙钟/clean-window 为准，kernel-sum 不能直接比**
（跨流重叠会多计数）。

**compile × head-parallel 不兼容**（实测）：逐头 python 循环触发 Dynamo `recompile_limit`
（日志 `[5/8]`）→ 部分帧静默回退 eager；组合不要用于生产，head-parallel 建议 eager 形态。

**comm-stream masking（950PR + H3 bulk 形态）实测否决（2026-09）**：用真实 H3 shapes 的
4 卡微基准测「a2a 与 block 计算重叠」（异步 all_to_all_single + 独立流/事件姿势）：
单 block a2a 的耗时明显小于 block 计算耗时，重叠只隐藏其中一部分（**hidden_ratio 明显低于 0.5
判据**），折算每步理论收益上限也只有个位数百分比且未计事件/流开销 → **低于阈值，不做**。机理：逐层数据依赖
下可重叠的独立计算少 + a2a 占比小（5s 档步内 comm 占比仅一至两成，且 compile 出图已把 a2a 留 eager 拿过通信红利）。
**换场景才重估**：长序列档 comm 占 kernel 总耗时**过半**，序列更长时 masking/切分收益随占比放大。
（910B dummy-run 曾报 masking 大幅削 comm——那是 funcol 层 monkey-patch + 不同序列结构，勿迁移。）

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
- 结果示例（4 卡跨岛组）：alltoall 64MB 为几十 GB/s 量级（1MB 时只有个位数 GB/s，随尺寸递增）；
  allreduce 同尺寸约为 alltoall 的一半，allgather 更低。绝对带宽读数见归档
  `{run_results_dir}/archive/`；2 卡直连对比可量化 UB/SYS 差（见 §1）。

**隐含带宽校验（无工具时的替代证据）**：由 kernel profile 的 comm 总耗时与模型 shape 推算
per-call 载荷（seq-parallel a2a ≈ local_tokens × heads × head_dim × 2B），载荷/耗时 ≈ 隐含带宽；
与上述实测量级自洽即可判定「无异常超时/重传」。本链参考值：5s USP4 档 a2a 每步数百次调用，
由此推算的隐含带宽**明显低于纯 alltoall 峰值**（差额为 per-call host/notify 开销，正常）。

## 4. 多卡环境诊断与恢复

症状与处理（共享机多租户，先 npu-smi 再动刀）：

| 症状 | 可能根因 | 处理 |
|---|---|---|
| 某卡或整组 ~10× 变慢（步长抬升一个量级，CPU 100% 而单算子 GEMM 正常） | 运行中 SIGKILL 多卡任务 → NPU/HCCL 驱动侧状态异常（可含端口句柄泄漏） | 停止该组实验；**换卡组验证**（同机其他健康卡组常可恢复）；必要时重启容器/复位 NPU（管理员） |
| npu-smi Health=OK 但整组多卡 run **每步时长均匀偏大**（无热降频斜率） | SIGKILL 驱动损伤**残留态**：npu-smi 健康列看不出（2026-09 实测受损组跨天不自动恢复） | 组可用性必须用**真实多卡 run 验证并核对每步时长**（日志 per-step cost 行）：4 步 smoke「跑通出视频」仍可能每步时长异常（时长异常=未通过）；要求驱动级复位后再用 |
| 任意新 HCCL init 报「端口 already been bound」（ss 看不到监听） | NPU 网卡 listen socket 泄漏（强杀/abort 累积）或**未 set_device**（见 §3） | 先查 set_device 姿势；再换全新端口段（如 30000+）；仍失败 → 驱动复位 |
| `hcclCommInitRootInfoConfig error` | ranktable/端口/设备未选 | 查 `/usr/local/Ascend/driver/topo/{chip}/*.json` 是否齐全（容器内常缺失→docker cp/挂载）；查端口段与 set_device |
| 卡 Health Warning/Alarm | 历史告警（可能残留）；**共享宿主上「多数卡告警」可为常态**（不代表整机不可用） | 用前以 `npu-smi` 实测**实际可用卡组**，并以真实多卡 run 复核；同窗口同卡组纪律优先 |

工具：`npu-smi info -t topo/-m/-t temp`、`ss -lunp`、`dmesg | grep -i ascend`。
日志纪律：框架仓内会话过程文档（非本仓、不入库）含凭据勿提交；报告不回显口令。

## 5. 复现与对比纪律

- 固定 rank0 口径（多卡各 rank 值不同）；per-step 用 p50 / 或 **clean-window（steps 2-14）avg**
  （机箱热时长跑**后段**会热降频、步长抬高，与代码 / **编译档** / 租户无关——**clean-window 与热降频判定口径**
  见 `../../perf-gate/references/measurement-discipline.md` §7，**本节只给读数**；现象来源见 framework-integration
  `lightx2v-enablement.md` §3.6）。
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
- **实测参考（H3 T2VA USP2，12 步裁剪）**：DiT Ulysses a2a ≈182 次/步、每步载荷几十 GB 量级（载荷占比近乎全部，
  通信主体）；text encoder TP all_reduce ~101 次/encode、载荷几十 MB 量级；VAE video all_gather 14 次/请求、
  载荷 GB 量级；audio VAE 无并行（无 DistributedVaeMixin）→ 0 通信。方法与明细见
  framework-integration `cache-dit-enablement.md` §6（采集方法落 `profiling-collect`）。

## 7. 通信占比口径：按并行策略与特性叠加重估（方法增量 2026-09-12）

> 本节为**追加的方法增量**（来源：vLLM-Omni 0.28 × 扩散模型实测的方法化）。下列占比排序与方向都是
> **该框架 × 该模型 × 该规模下的观测**，不是执行序——换框架 / 模型 / 规模按 §2 与 §5 重测；
> 绝对耗时与绝对加速比不入库（归档见会话产物目录 `{run_results_dir}/archive/`）。

- **选型前先核框架能力，再逐一使能比较**：并行候选必须在**框架实际支持的形态范围内**逐个跑通比较，
  不能只按理论通信量选型——同一形态可能被框架限制否决（例：某框架的 Ring 路径不支持 `attn_mask`
  且对应开关无效 ⇒ 序列并行只能选 USP；又例：**纯 TP 形态（无序列并行）在某框架 × 某模型 × 长序列
  负载下整体不可用**——两种编码器 TP 模式各 fail 在不同位置，根因是形态而非超参，见
  `../../framework-integration/references/vllm-omni-enablement.md` §2；本组合观测）。候选矩阵还须
  **覆盖 2 卡形态**：`TP1×USP2`（Ulysses，
  少次大交换）与 `TP2`（每步大量 allreduce，per-call 开销主导）**收益结构不同**，短 compute 任务
  （少步 / 小分辨率）下前者可反超后者；4-rank 形态在短任务下逐层通信开销 >> GEMM 分摊收益 → 回退。
  **判据**：若「同卡数、不同切分」的那一对里有形态被框架否决，则该对在本组合下取不到，须改报
  同卡数可用的另一种切分对（或明确标注该对缺失），不得用跨卡数对比冒充形态对比。
- **掩盖空间上限 = 未重叠通信占比，且该占比随并行策略改变**：同模型同步数下未重叠 comm 占比随并行
  形态变化（本组合观测：`USP2 > TP4 > USP4(int8)`）；框架若无 comm-stream 机制（单步 `Overlapped=0`），
  掩盖收益上限就是该占比。**判据**：先用 `step_trace_time` 拆 compute / comm(未重叠) / free，再决定
  是否投入掩盖实现；compute-bound 规模下先做长序列重测（占比随序列变长放大）再投入。
- **特性叠加改变单步结构 ⇒ step_trace 须按叠加态重采**：量化把 MatMul 换成量化 GEMM（compute 缩短、
  comm 基本不变 ⇒ **comm 占比抬升**）、稀疏把 attention 换成稀疏算子；而**缓存不改变单 forward 的
  kernel 组成**（其作用是步级跳过 forward）⇒ 用 kernel 级证据判缓存无效属口径错误。每次特性叠加 /
  回退后重采一次同口径 step_trace，并据此重估掩盖空间（量化后通信重审的完整动作见
  `model-auto-optimization/references/lossless-methodology-notes.md` §E）。
- **有损档落地后回跑被显存卡住的通信组合**：量化等降显存动作会解锁此前因显存不可行的并行形态
  （BF16 单 rank 全量驻留超单卡容量 → 量化后可 TP1×USP4）；每个量化档落地后按「并行 × 新显存余量」
  补跑候选（至少少量步快测），把新增可行组合纳入矩阵（方法同上述笔记 §B）。
- **跨拓扑与跨窗口的可比性**：跨拓扑输出**非逐字节**（同 seed 也不同）⇒ 无损对比必须在**同一并行
  配置内**做，跨拓扑数值只作参考；不同时间窗的绝对值受热降频 / 同机负载影响 ⇒ frontier 与回退结论
  以**同窗相邻对**为准（与 §5 一致）。
- **占比也会被采集方式扭曲（2026-09-13 增量）**：profiler 单 forward 捕获会**膨胀该次 forward 的
  墙钟**，而 `Communication(Not Overlapped)` 的**秒数**基本不变 ⇒ 直接算「未重叠 ÷ 被捕获步的
  `Stage`」会**偏低**。可比口径 = 未重叠通信**秒数** ÷ **未开 profiler** 的单步耗时；因此捕获要放在
  被排除的 warmup 请求里，计时请求全程不采集。另：读 `step_trace_time.csv` 必须**精确匹配列名**
  （`Communication` 是 `Communication(Not Overlapped)` 的子串，子串匹配会静默取错列）。
- **占比随步数是否漂移**是同步并行选型的判据之一（少步测得 ≠ 全步可得），协议与阈值见
  `few-step-multirank-protocol.md` §6。

## 8. 维护与更新

- **触发条件**：机型 / 互联拓扑或卡数变化（§1 的 `npu-smi info -t topo` UB / SYS 判定与
  「SYS 对约低三成」的读数；§2 表中「单 UB 岛 USP4 bulk 最优」与「SYS 跨岛组 head-parallel
  更优」的翻转关系）；CANN / 驱动版本变化（§3 `hccl_test` 的
  `HcclGetRootInfo failed / invalid data`、§4 的 `already been bound` 与
  `hcclCommInitRootInfoConfig error`、§2 的 compile × head-parallel `recompile_limit`
  （日志 `[5/8]`）都属版本相关缺陷）；序列长度 / 步数档 / 框架变化（§2 的 masking 否决结论、
  §7 的「换框架 / 模型 / 规模按 §2 与 §5 重测」）。
- **复核方法**：§1 判据用同尺寸同口径重测一次 2 卡 UB / SYS 对；§2 的两条选型结论必须
  **同窗口同卡组**重跑（4 步 smoke + CANN profile 核验 a2a 形态未退化 + 30 步墙钟），判据以
  墙钟 / clean-window 为准 ——「kernel-sum 不能直接比」；masking 值不值得做按 §2 的 4 卡微基准
  重算 `hidden_ratio` 是否仍低于 0.5 判据。
- **口径联动**：§5 的 clean-window（steps 2-14）与热降频判定、§7 的 profiler 捕获膨胀与
  「`Communication` 是 `Communication(Not Overlapped)` 子串」的精确匹配要求，真源在
  `../../perf-gate/references/measurement-discipline.md` 与 `few-step-multirank-protocol.md` §6；
  那两处口径改了，本节读数与判据须回来同步。
