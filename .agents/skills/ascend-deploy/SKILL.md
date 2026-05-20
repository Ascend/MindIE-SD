---
name: ascend-deploy
description: MindIE-SD 代码部署与编译安装。已在昇腾设备上直接编译安装；本地开发机通过 SSH 推送到远端昇腾设备，
             支持 Docker 容器内源码编译。当用户需要部署安装 MindIE-SD、更新远端容器内代码、
             或管理多人共享 NPU 环境时使用此 skill。
             即使用户只提到"把代码推到服务器"而未说 SSH 或昇腾，只要上下文涉及部署都应触发。
             由 dev-workflow 的部署阶段触发。
---

# 部署 MindIE-SD

## 路径判断

```text
你当前在哪里？
├─ 已在昇腾设备上 → 跳到 §2 编译安装
└─ 本地开发机（需推送到远端） → §1 远端前置 → §2 编译安装
```

## §1 远端前置（仅本地→远端场景）

### Step 1: 收集参数 + 传输

部署前必须向用户确认以下信息（无默认值）：

| 参数 | 说明 | 示例 |
|---|---|---|
| **远端 IP** | 昇腾服务器地址 | `<远端IP>` |
| **用户名** | SSH 登录用户 | `<用户名>` |
| **密码** | SSH 登录密码 | — |
| **远端工作目录** | 代码存放路径 | `<远端工作目录>` |
| **Docker 镜像** | 已配置好 CANN+PyTorch 的镜像 | `<docker镜像:tag>` |
| **容器名** | 远端容器名称 | `<容器名>` |
| **容器状态** | 已运行 / 需新建 | — |
| **路径映射** | 容器内外路径是否一致 | 是 / 否 |

### 用户确认（阻断点）

部署前向用户展示以下信息并确认：

> **远端**: {IP} / {容器名} / {工作目录}
> **镜像**: {镜像名}（从 GitCode 或官方文档验证最新版本）
> **NPU 可用卡数**: {空闲数} / {总数}（通过 `npu-smi info -l` 确认）
> **是否继续？** [Y/N]

禁止猜测未经验证的参数（镜像版本、容器配置等），所有参数必须由用户确认后方可执行。

### 连接复用原则

⚠️ **所有远端操作（传输、编译、验证、npu-smi）必须复用同一个 SSH 连接。**
严禁每个步骤独立 `ssh.connect()`——远端 SSH 有 `MaxStartups` 限制，短时间多次连接会触发拒绝访问。

- 单个 `paramiko.SSHClient` 对象从传输到验证全程复用，传输完成后不立即 `ssh.close()`
- `sftp.stat` 逐个文件比对是主要瓶颈：先 `ls -l` 拉取远端文件清单，本地 diff 后仅传输变更文件
- `docker exec` 命令用 `;` 串联多个操作，减少 bash login shell 初始化次数

### Step 2: CRLF 换行符处理

Windows 本地源码中的 CRLF 换行符会在远端 Linux 容器中导致 shell 脚本编译失败。

**核心原则：不在本地修改源码文件的换行符。** 直接修改本地文件会产生大量无意义的 git diff。

换行符转换由 `deploy_to_remote.py` 在传输阶段自动处理，仅对 `.sh`/`.py` 文件做 CRLF→LF，转换仅作用于远端临时工作副本。

执行传输 + 编译：

```powershell
python deploy_to_remote.py
```

脚本自动完成：收集本地文件 → CRLF 转换 → 增量传输 → 容器内编译安装。

### 容器模式选择

根据部署目标选择容器创建模式：

| 模式 | 参数 | 适用场景 |
|------|------|---------|
| **特权模式** | `--privileged` | 开发调试，最大权限 |
| **基础模式** | 仅 `--device /dev/davinci*` | 纯推理，最小权限 |
| **全量模式** | 基础 + profiling/logging 挂载 | 性能调试 |

全量模式附加挂载：

```text
-v /var/log/npu/profiling/:/var/log/npu/profiling/
-v /var/log/npu/slog/:/var/log/npu/slog/
```

设备映射参数：

```text
--device /dev/davinci_manager --device /dev/devmm_svm
--device /dev/hisi_hdc
--device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3
```

> 完整容器创建指南见 ascend-docker skill（Ascend agent-skills）。

## §2 编译安装（共用）

以下步骤本地和远端通用。

### Step 3: build_tik_ops.sh 规避

`build_tik_ops.sh` 在部分环境会失败（参考 [issue#64](https://gitcode.com/Ascend/MindIE-SD/issues/64)），部署前注释掉：

在 `build/build_ops.sh` 中将：

```bash
source ${current_script_dir}/build_tik_ops.sh
```

改为：

```bash
# source ${current_script_dir}/build_tik_ops.sh
```

### Step 4: 编译 + 安装

**远端（Docker 容器内）**：

```bash
docker exec <容器名> bash -lc '
source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
cd <工作目录>/MindIE-SD &&
pip install build wheel -q &&
python setup.py build_py &&
pip install -e . &&
echo DEPLOY_SUCCESS
'
```

**本地（已在昇腾设备上）**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <项目根目录>
pip install build wheel -q
python setup.py build_py
pip install -e .
```

### Step 5: 验证

确认编译安装成功：

```bash
# 远端
docker exec <容器名> bash -lc 'python3 -c "import mindiesd; print(mindiesd.__version__)"'

# 本地
python -c "import mindiesd; print(mindiesd.__version__)"
```

### Step 6: 兼容性前置检查（§B 远端路径）

在远端容器内验证环境版本兼容性：

```bash
# PyTorch + torch_npu 版本匹配
docker exec <容器名> bash -lc 'python -c "
import torch, torch_npu
print(f\"PyTorch={torch.__version__}, torch_npu={torch_npu.__version__}\")
assert torch_npu.__version__ >= \"2.6\", \"torch_npu too old\"
"'

# CANN 环境
docker exec <容器名> bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    cat /usr/local/Ascend/ascend-toolkit/version.cfg 2>/dev/null | head -3'
```

本地路径直接用 `python -c` + `source set_env.sh` 验证。

| 检查项 | 最低要求 | 不满足时 |
|--------|---------|---------|
| PyTorch | >= 2.6 | 升级 PyTorch 版本 |
| torch_npu | >= 2.6，与 PyTorch 主版本匹配 | 升级 torch_npu |
| CANN | >= 8.0.0，含 bisheng 编译器 | 升级 CANN SDK |
| Python | >= 3.10 | 升级 Python |
| cmake / build / wheel | 可用 | `pip install cmake build wheel` |

> 版本不匹配时中止，参考故障排查表或编译依赖表。编译依赖见下方「编译依赖」章节。

### Step 7: NPU 检查 / OOM 处理

运行前确认 NPU 卡状态：

```bash
# 远端
docker exec <容器名> bash -lc 'npu-smi info -l'

# 本地
npu-smi info -l
```

### NPU 状态检查

| 检查项 | 命令 | 目的 |
|--------|------|------|
| 平台识别 | `npu-smi info -t product` + `dmidecode` | 区分 A2(910B) vs A3 |
| NPU 健康 | `npu-smi info -t health` | 运行前预检 |
| 显存使用 | `npu-smi info -t memory -i <id>` | 选择空闲卡 |
| 进程占用 | `npu-smi info proc` | 检查设备是否被占用 |
| AI Core 利用率 | `npu-smi info -t usages` | 运行时监控 |
| 温度/功耗 | `npu-smi info -t temp / power` | 运行稳定性检查 |

完整命令参考见 npu-smi skill（Ascend agent-skills）。

### OOM 处理

| 方案 | 说明 | 效果 |
|---|---|---|
| `--mode cpu_offload` | `pipe.enable_sequential_cpu_offload()` | 峰值 ~19GB（Wan2.2） |
| `torchrun --nproc_per_node=N` | 多卡环境验证 hccl | 每卡独立加载 |
| OOM 优雅退出 | try/except + 明确提示 | 告知预期结果 |

> 需要 profiling 数据时使用 profiling-collection skill。
> 部署完成后，使用 model-verification §B 验证已部署模型的推理正确性。

## 编译原理

`python3 setup.py build_py` 执行以下步骤：

1. 通过 `build/build_ops.sh` 编译 AscendC 自定义算子（laser_attention, la_preprocess 等）
2. 通过 `build/build_plugin.sh` 用 cmake 编译 C++ PyTorch 插件，生成 `.so` 文件
3. 将编译产物拷贝到 `mindiesd/plugin/` 和 `mindiesd/ops/`

`pip install -e .` 以可编辑模式安装，使代码修改即时生效。

> `pip install -e .` 仅在以下目录有新增或变更时才需要重新执行：
>
> - `mindiesd/` — Python 包源码索引需刷新
> - `csrc/` — C++ 源码需重新编译为 `.so`
> - `build/` — 编译脚本变更
>
> 若变更仅涉及 `examples/`、`tests/`、`docs/` 等非包目录，可跳过此步骤，
> 直接使用远端已有的安装版本。

## 编译依赖

远端容器内必须满足的最低编译条件：

| 依赖 | 要求 |
|---|---|
| CANN | >= 8.0.0，含 bisheng 编译器 |
| Python | >= 3.10 |
| PyTorch | 2.6 / 2.7 / 2.8 / 2.9 |
| torch_npu | 与 PyTorch 版本匹配 |
| 环境变量 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 编译工具 | cmake, build, wheel (`pip install build wheel`) |

## 启动日志排查

部署或推理时遇到第三方库的警告、错误、预期外的打印输出，使用以下方法定位源头：

### 追踪 import 链

当某个第三方库在导入时打印预期外的信息：

```python
import sys
_real_import = __builtins__.__import__
def _tracing_import(name, *args, **kwargs):
    if '目标模块' in name:
        import traceback
        print(f'=== 导入 {name} ===')
        traceback.print_stack()
    return _real_import(name, *args, **kwargs)
__builtins__.__import__ = _tracing_import
```

### 追踪 logger warning

当怀疑某个 logger 输出了预期外的 warning：

```python
import logging, traceback
class TraceHandler(logging.Handler):
    def emit(self, record):
        if '搜索关键词' in record.getMessage():
            traceback.print_stack()
h = TraceHandler()
h.setLevel(logging.WARNING)
logging.getLogger('目标logger名').addHandler(h)
```

### 查找 C 级打印

当打印来自 C/C++ 库（非 Python 可控）：

```bash
strings /path/to/library.so | grep "搜索关键词"
```

## 故障排查

高危问题速查（完整决策树见 references/troubleshooting-tree.md）：

| 症状 | 原因 | 解决 |
|---|---|---|
| 编译失败 | CANN 环境未 source | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `build_ops.sh` exit code 101 | build_tik_ops.sh 失败 | 执行 §2 Step 3 注释掉该行 |
| `import triton` 成功但 `0 active drivers` | 安装了标准 triton（非 Ascend 版本） | `pip uninstall triton -y && pip install triton-ascend && pip install pybind11` |
| SSH 连接过多导致拒绝访问 | 远端 `MaxStartups` 限制 | 遵循 §1 连接复用原则 |
| `ModuleNotFoundError: mindiesd` | `pip install -e .` 未重新索引 | 重新执行 `pip install -e .` |

> 其他问题（SSH 认证、文件缺失、docker exec 引号转义、CRLF、环境依赖等）见 references/troubleshooting-tree.md。

## 参考

- MindIE-SD Docker 镜像: `docker/Dockerfile_910b_aarch64.ubuntu`
- 安装文档: `docs/zh/installation.md`
- 编译指南: `docs/zh/developer_guide/build_guide.md`
- 已知问题: <https://gitcode.com/Ascend/MindIE-SD/issues/64>

## Reference Files

- 🔍 `references/troubleshooting-tree.md` — 加载时机: 部署或编译遇到异常，需系统排查定位根因时

## Bundled Scripts

- `scripts/deploy_to_remote.py` — 增量传输 + 编译部署主脚本（**仅 §1 远端路径需要**）
- `scripts/pick_free_device.py` — 自动检测 HBM 占用最低的空闲 NPU 卡（本地/远端通用）

> Profiling 采集脚本 (`deploy_and_profile.py`) 已拆分并迁移至 profiling-collection skill（见 `collect_profile.py`）。

## 维护与更新

当远端昇腾环境变化（torch/torch_npu 版本升级、CANN SDK 更新）、
容器配置调整、npu-smi 命令行为变化或发现新的部署问题时，
按 dev-workflow 的复盘流程更新本 skill。
