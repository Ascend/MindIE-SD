# 故障排查决策树（安装 / 部署域）

> **本文件只收「安装就绪」域的问题**——把部署环境装起来（mindiesd 编译安装、三方框架全栈、
> 容器与依赖）与把权重备齐。**止于安装完成**：`import mindiesd` 成功、版本配套就位、权重就位。
> 运行期问题（服务启动 / 特性使能 / 选档 / 精度判定 / 多卡归因 / 传输域）一律**收敛为指针**，
> 不在本文件展开——先按下表定位归属，再去对应技能。

## 归属速查（越界问题都从这里出去）

| 症状族 | 归属 |
|---|---|
| 服务启动 / 运行入口 / 请求级 task 档位 / 档位报错怎么回退 | `framework-integration/references/run-entry-and-request-tiers.md` |
| 特性使能不生效 / 计数契约 / 三层证据 / 回退判据 | `framework-integration/SKILL.md` §1 |
| 显存不足选哪一档降 / 使能档 crash 退到哪一档 | `dit-perf-opt/references/resource-fallback-tiers.md` |
| 多卡运行期劣化（SIGKILL 残留态 / 端口 bind / 拓扑选卡） | `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §1–§5 |
| 热降频与 clean-window **取数口径** | `perf-gate/references/measurement-discipline.md` §7（读数仍在 `dit-parallel-opt` 上表文件） |
| 自研算子运行期不可见（`inferShape does not exist`）/ golden 校验 | `operator-dev/references/custom-op-runtime-deploy-verify.md` |
| 输出异常与精度判定（NaN / 黑图 / 花屏 / eager vs compiled） | `accuracy-gate`（`../../accuracy-gate/references/silent-failure-localization.md`、`../../accuracy-gate/references/equivalence-criteria.md`） |
| SSH / 认证 / CRLF / 传输域 / 本地开发机 schannel | `remote-access/SKILL.md`「故障排查」表 + `remote-access/references/transport-troubleshooting.md` |
| 选空闲卡 / 卡占用与进程归属 | `remote-access/SKILL.md`「空闲卡选择」 |
| 测量口径 / 报数与入库 | `perf-gate` |

## A. 部署失败

```text
部署失败
├─ SSH 认证失败 → 不属本技能：remote-access「故障排查」表（核对 IP/用户名/密码与网络连通性）
├─ 编译错误
│   ├─ CANN 环境未 source → source /usr/local/Ascend/ascend-toolkit/set_env.sh
│   ├─ build_tik_ops.sh 失败 → 注释掉 build_ops.sh 中的 source build_tik_ops.sh 行
│   └─ 缺少编译依赖 → pip install build wheel cmake
└─ 文件缺失（首次部署）
    ├─ model/__init__.py 缺失 → 全量传输该目录
    └─ 新增 .py 文件未识别 → 重新执行 pip install -e .
```

## B. 资源不足（安装 / 部署期）

```text
资源不足
├─ CPU OOM（构造阶段）→ 不是"降档"问题：改进构造方式（如 meta→to_empty）
└─ NPU OOM（推理阶段）→ 不属本技能：降显存档位与回退顺序见
    dit-perf-opt `../../dit-perf-opt/references/resource-fallback-tiers.md` §1；
    卡占用 / 选空闲卡见 remote-access `../../remote-access/SKILL.md`「空闲卡选择」
    排查步骤: npu-smi info -t memory -i {device_id} 看显存；确认是否有其他进程占用
```

## C. 输出异常（NaN / 黑图 / 花屏）

**不属本技能**（精度与静默错误判定）：判据、等价分层与排障入口见 `accuracy-gate`
（`../../accuracy-gate/references/silent-failure-localization.md`；"等价替换有没有把结果改坏"见
`../../accuracy-gate/references/equivalence-criteria.md`）。
其中**加载期**症状若源于传输：tokenizer / 权重加载异常常由**二进制文件被 CRLF 转换损坏**引起
→ `remote-access`（`../../remote-access/SKILL.md`「故障排查」表 +
`../../remote-access/references/transport-troubleshooting.md`）。

## D. 安装 / 部署期 NPU 与进程异常

```text
NPU 异常（安装验证期）
├─ 进程卡住无响应 / 卡处于 ERROR → npu-smi info -l 看卡状态；复位与卡组可用性判定见
│   dit-parallel-opt `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §4
│   （Health=OK ≠ 组可用）
├─ 多卡互联 / 拓扑相关 → dit-parallel-opt
│   `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §1
├─ core dump / 进程退出
│   ├─ 算子 ACL error → 查询 CANN 错误码文档
│   ├─ triton 版本错误 → pip uninstall triton -y && pip install triton-ascend
│   └─ 特定算子 crash → **回退到 eager 路径**属选档 + 回退姿势：
│       dit-perf-opt `../../dit-perf-opt/references/resource-fallback-tiers.md` §2（选哪一档）+
│       framework-integration `../../framework-integration/references/run-entry-and-request-tiers.md`
│       §3（怎么退）；自研算子运行期不可见另见 operator-dev
│       `../../operator-dev/references/custom-op-runtime-deploy-verify.md`
└─ 排查步骤
    1. 查看容器日志: docker logs {container}
    2. 查看 CANN 日志: /var/log/npu/slog/
    3. 用单卡最小配置复现（最小分辨率 / 1 步）
    4. 若稳定复现 → 记录并转对应技能（上述归属速查表）
```

## E. 本地开发机（Windows）

**已收敛为指针**（SSH / 传输域属 `remote-access`）：

- git / curl HTTPS 握手失败（`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`）
  → `remote-access/references/transport-troubleshooting.md` §1；
- `deploy_to_remote.py` 的**路径语义**与**排除列表**（`refs/` 等大目录会全量传输）单点在本技能
  `SKILL.md`「部署脚本」节；远端布局与本仓约定不一致时不要直接跑脚本，改精准 SFTP 同步；
- 凭据与日志纪律 → `remote-access/SKILL.md`（环境变量 `MINDIE_SSH_PASSWORD` / SSH key；不回显密码）。

## 工具速查

| 工具 | 用途 |
|------|------|
| `npu-smi info -l` | 列出所有 NPU 卡状态 |
| `npu-smi info -t memory -i 0` | 查看卡 0 显存 |
| `docker logs {container}` | 查看容器运行日志 |
| `python -c "import torch_npu; print(torch_npu.__version__)"` | 确认 TorchNPU 版本 |

### F. 多卡运行期劣化（**正文已收敛为指针**，不属本技能）

- 症状与恢复（整组数量级变慢 = SIGKILL 驱动损伤残留态、端口 `already been bound`、
  `hcclCommInitRootInfoConfig error` 的 ranktable 检查、拓扑选卡）→
  `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §1–§5；
- **clean-window / 热降频的测量口径**（怎么取数、怎么判热降频）→
  `../../perf-gate/references/measurement-discipline.md` §7（口径单点），具体读数仍归上述 dit-parallel-opt 文件；
- 本技能只保留**安装侧**一条：容器**必须挂载** HCCL ranktable `/usr/local/Ascend/driver/topo`
  （见 §A 与 `vllm-omni-build.md` Step 2.1）。并列/掩盖/选型归 `dit-parallel-opt`。

### G. 自研 CANN 算子部署顺序与 golden 校验（**正文已收敛为指针**，不属本技能）

- **部署顺序**（产物只在 `{repo}/mindiesd/ops/vendors/*`、运行期由 `import mindiesd` 设
  `ASCEND_CUSTOM_OPP_PATH`、**必须先 import mindiesd 再初始化 NPU / 建张量**）→
  `../../framework-integration/SKILL.md` §1.5 + `../../framework-integration/references/cache-dit-enablement.md` §2.2；
- **部署校验与 golden 通过判据**（可见性 → 走的是哪一个 → 数值）→
  `../../operator-dev/references/custom-op-runtime-deploy-verify.md`；
- gloo 偶发 `ss1.ss_family == ss2.ss_family (10 vs 2)`（IPv4/IPv6 混用）→ 多卡启动陷阱，
  见 `../../dit-parallel-opt/references/ascend-parallel-traps.md`（重试；持续出现则强制 IPv4）。

### H. 三方框架容器：装库 / 权重缺口（自 framework-integration 迁入）

> 归属判据：**"库没装 / 权重不全 / 装错树"属环境安装**，不是特性使能问题——修复后回到
> `framework-integration` 继续使能与验证。

```text
vllm serve --omni 启动即失败
├─ ImportError: libxcb.so.1: cannot open shared object file
│   └─ opencv-python 依赖 X11 库（容器缺系统库，非 Python 依赖问题）→
│      dnf install -y libxcb xcb-util* libX11 libXext mesa-libGL ...
├─ ValueError: ... weights were not initialized from checkpoint
│   └─ 权重分片缺失：对照 {model_dir}/*.safetensors.index.json 的 weight_map 逐分片核对，
│      缺失分片从 hf-mirror（https://hf-mirror.com/{org}/{model}/resolve/main/...）补下载；
│      下载前先确认远端是否已存在该分片（避免重复下载）
├─ pip 报 `Invalid wheel filename (wrong number of parts)`
│   └─ 第三方 wheel 被**重命名**（如 `torch.whl` / `torch_npu.whl`）后安装：pip 要求文件名符合
│      `{name}-{version}-{build}-{py}-{abi}-{platform}.whl` → **保留原始文件名**重新安装
│      （形如 `<name>-<version>-<build>-<py>-<abi>-<platform>.whl`；**版本号按实际下载的包**，不要为下载方便改名）
└─ 部分机型代际上设置 MINDIE_SD_FA_TYPE 导致算子路由异常
    └─ 该变量在这些代际不适用 → 删除（**用 `npu-smi info -l` 确认目标代际后复核**，勿与其他代际场景写法混用）

同机多棵 editable mindiesd 树：按「能力实体」选树，不按目录名 / 新旧选树
└─ 先列出目标特性依赖的能力并逐项验证：
   ├─ 稀疏类查算子签名是否含该框架需要的参数（如 rf_v2 需要 video_spans，缺则直接不兼容）
   └─ 融合类查插件 md5 / import 是否成功（如 mm_swiglu_mxquant）
   特性跑不通先核对"这棵树有没有这个能力"，再谈接线与算子
```

> 装库 / 权重缺口修好后，**服务启动、特性使能与验证**回到
> `framework-integration`（`../../framework-integration/references/run-entry-and-request-tiers.md`
> 与 `../../framework-integration/SKILL.md` §1）。

## 失效信号与复核

- **§A / §D 的绕行条目（注释掉 `build_ops.sh` 里的 `source build_tik_ops.sh` 行、`triton` 换 `triton-ascend`、补装 `build wheel cmake`、`pip install -e .` 认领新增 `.py`）都是某一版工具链下的补丁**：换 CANN / torch_npu 后在干净容器里按原样重跑一次编译安装，某条报错不再出现即该条已失效，删除本项而不是继续照抄绕行。
- **§H 的四类缺口分别绑基础镜像、权重目录与设备代际**：`ImportError: libxcb.so.1` 用 `dnf list installed libxcb` 复核；`weights were not initialized from checkpoint` 按本文件既有方法对照 `{model_dir}/*.safetensors.index.json` 的 `weight_map` 重数一遍分片；`MINDIE_SD_FA_TYPE` 那条要用 `npu-smi info -l` 确认目标代际后才成立；`Invalid wheel filename` 那条看 pip 当前版本是否已接受被重命名的 wheel。
- **§F / §G 的运行期条目（ranktable 挂载 `/usr/local/Ascend/driver/topo`、必须先 import mindiesd 再初始化 NPU、gloo `ss1.ss_family == ss2.ss_family (10 vs 2)`）属容器挂载与网络栈状态**：重跑一次多卡最小启动即可判定——`hcclCommInitRootInfoConfig error` 复现即 ranktable 仍缺；gloo 报错不再出现即该条转为历史记录（IPv4/IPv6 已统一）。
- **§E 的 `deploy_to_remote.py` 路径语义与排除列表是脚本实现的事实，不是稳定结论**：以 `SKILL.md`「部署脚本」节的当前实现重新核对一次（读脚本排除规则 + 比对该次同步实际传过去的文件集合），脚本一改本条的绕行描述即过期。

## 维护与更新

当部署 / 编译 / 安装故障模式变化或新增高危问题时，按 dev-workflow 的复盘流程更新本文件；
**运行期**问题的知识更新到上表对应技能，本文件只维护归属指针与安装域条目。
