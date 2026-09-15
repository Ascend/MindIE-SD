# 并行采集的 profile 目录隔离

> **加载时机**：同一 host 上**并行**跑多个 run（典型：一模型一 NPU 卡、多配置交错采集）时，
> 确定每个 run 的 profiler 输出目录归属。单 run 串行采集不存在本文件的问题。

## 1. 问题与后果

采集入口的 profiler 输出目录是**相对 CWD 的固定名字**（本仓默认 `profile_l1`，
见 `scripts/collect_profile.py` 的 `DEFAULT_PROFILE_DIR`）。同一 host 上并行启动多个 worker 时，
若都落到该默认目录：

- 多个 run 的 `ASCEND_PROFILER_OUTPUT/` **互相覆盖** → `kernel_details.csv` 缺失 / 截断 / 混档；
- 现象是「某个 run 的 csv 读不到」或「另一个 run 被拖慢」，且**事后无法从共享目录判断归属**；
- 回传打包时会把多 run 数据打进同一个 tar，污染 profiling-analyze 的输入契约。

## 2. 约定（强制）

1. **一 run 一目录**：按 `模型 × 配置` 命名，例如
   `{work}/profiles/{model}_{eager|compile}`；目录名必须能从名字反推 run 身份。
2. **采集入口透传目录**：入口只负责把目录名透传到远端脚本（本仓入口已支持 `--profile-dir`），
   **不在采集入口里替被测脚本命名**。
3. **被测脚本自持默认值**：脚本侧以「`--profile-dir` 参数 > 环境变量 > 默认目录」的优先级取值
   （载体自己的参数名/变量名由载体登记，采集侧不另立一套）。
4. **回传按 run 聚合**：采集完按模型/配置分别回传或聚合，**不要事后从共享默认目录里猜归属**。
5. **纯 wall 计时无此冲突**：未开 profiler 的 wall-only run 不写该目录，只有 kernel 级报表
   （需要 `kernel_details.csv`）才要求隔离。

## 3. 与其它采集纪律的关系

| 纪律 | 维度 | 说明 |
|---|---|---|
| 本文件 | **目录** | 一 run 一输出目录，防覆盖 |
| 只 rank0 采集 | **进程** | 防多卡重复输出（见 SKILL「三方框架入口」） |
| warmup 在 profiler 外 | **时间窗** | 防 JIT / 首次分配污染（见 SKILL「Warmup 配置」） |

三者正交，**都要满足**；缺任一都会让 kernel 级读数不可复核。

## 4. 采集后自查

- 每个 tar 解包后确认 `ASCEND_PROFILER_OUTPUT/` 三件套（`kernel_details.csv` +
  `trace_view.json` + `step_trace_time.csv`）齐全；
- CSV 的算子计数 / 步数与**该 run 的配置**对得上（对不上 = 串档）；
- 隔离目录名与实际 run 身份一致（写进产物归档文件名，供 profiling-analyze 引用）。

## 维护与更新

当采集入口的目录参数语义变化、或新增支持并行采集的载体时，更新本文件与
`scripts/collect_profile.py` 的目录透传说明。
