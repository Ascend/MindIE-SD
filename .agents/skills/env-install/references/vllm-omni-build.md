# vLLM-Omni 全栈源码构建（Ascend 950PR / 950DT，x86_64）

> **目录** · [版本配套矩阵](#版本配套矩阵) · [Step 2.1: 容器环境预检](#step-21-容器环境预检) · [Step 2.2: 安装 torch + torch_npu](#step-22-安装-torch--torch_npu) ·
> [Step 2.3: 构建 vllm 0.26.0](#step-23-构建-vllm-0260) · [Step 2.4: 构建 vllm-ascend（releases/v0.26.0rc）](#step-24-构建-vllm-ascendreleasesv0260rc) · [Step 2.5: 构建 vllm-omni（main）](#step-25-构建-vllm-omnimain) ·
> [Step 2.6: 构建 mindiesd（全栈环境内）](#step-26-构建-mindiesd全栈环境内) · [维护与更新](#维护与更新)
>
> **维护说明**：本文件内容来自 env-install SKILL「三方框架全栈安装（vLLM-Omni 950PR 源码构建）」
> 一节拆分下沉（progressive disclosure：SKILL.md 正文保持 <500 行，只保留精简指针）。版本配套矩阵、
> Step 2.1–2.6 分步命令与内联已知坑 / ⚠️ 警告以本文件为准；更新 vLLM-Omni 构建路径、版本矩阵
> 或发现新的构建已知坑时改本文件即可。

## 版本配套矩阵

当目标是在远端容器内用 **vLLM-Omni 托管扩散模型**（Qwen-Image-2512 / Wan2.2 / MiniMax-H3 等），
需要安装完整栈：`torch + torch_npu + vllm + vllm-ascend + vllm-omni + mindiesd`。
官方预构建镜像（`quay.io/ascend/vllm-omni:*`）仅覆盖 Atlas A2/A3（aarch64）；
⚠️ **Ascend 950PR / 950DT（x86_64）必须从源码构建**。

以 950PR + vllm 0.26.0 为例（2026-08 实测可用）：

| 组件 | 版本 | 获取方式 |
|---|---|---|
| CANN | 9.1.0 | 容器镜像自带（`cann:9.1.0-950-*`） |
| torch | **2.11.0+cpu**（由 vllm 决定） | `pip install torch==2.11.0+cpu -i https://download.pytorch.org/whl/cpu` |
| torch_npu | **2.11.0**（与 torch 配套） | gitcode `Ascend/pytorch` release：`v26.1.0-pytorch2.11.0` 下的 `torch_npu-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl` |
| vllm | **0.26.0** | 源码构建（`VLLM_TARGET_DEVICE=empty`） |
| vllm-ascend | **releases/v0.26.0rc 分支** | 源码 `pip install -e . --no-deps --no-build-isolation` |
| vllm-omni | **main 分支**（配套 vllm 0.26） | 源码 `VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation` |
| mindiesd | dev 分支 | 源码 `python setup.py build_py && pip install -e .` |

> **版本推导顺序**：vllm 0.26.0 的 `pyproject.toml` 锁定 `torch == 2.11.0` →
> 从 [gitcode Ascend/pytorch releases](https://gitcode.com/Ascend/pytorch/releases) 选择
> `v26.1.0-pytorch2.11.0` 下载配套 torch_npu（版本 tag 含 `pytorch2.11.0` 字样）。
> ⚠️ 不要用 vllm-ascend / vllm-omni 的 `requirements.txt` 里硬编码的 torch 版本
> （旧 pin，会降级 torch）。

## Step 2.1: 容器环境预检

```bash
docker exec {容器} bash -lc 'uname -a'                    # 确认 x86_64
docker exec {容器} bash -lc 'npu-smi info -l | head -20'  # 确认 Ascend950PR 及卡数
docker exec {容器} bash -lc 'python --version'            # 确认 3.12（cp312 wheel）
```

⚠️ **必须检查容器是否挂载 HCCL ranktable 目录**：`/usr/local/Ascend/driver/topo/`
（含 `950/atlas_350_*.json`）缺失时，vllm 多卡启动会报 `hcclCommInitRootInfoConfig error code is 4`。
容器创建时需挂载或 `docker cp /usr/local/Ascend/driver/topo {容器}:/usr/local/Ascend/driver/topo`。

## Step 2.2: 安装 torch + torch_npu

```bash
# 容器内（cp312 与 Python 3.12 匹配）
pip install torch==2.11.0+cpu -i https://download.pytorch.org/whl/cpu
# 从 gitcode 下载 torch_npu wheel 后安装
pip install torch_npu-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"
```

## Step 2.3: 构建 vllm 0.26.0

```bash
cd /home/{user}/code/vllm-0.26.0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ⚠️ 先升级 setuptools（容器默认 67.4 太旧，pyproject 解析报 license 错误）
pip install "setuptools>=77.0.3,<81.0.0" setuptools-scm setuptools-rust wheel ninja regex

# 跳过 rust：vllm 0.26 的 rust 前端是 optional（VLLM_REQUIRE_RUST_FRONTEND 默认不强制）
export VLLM_TARGET_DEVICE=empty
export VLLM_REQUIRE_RUST_FRONTEND=0
export VLLM_USE_PRECOMPILED=0
pip install -e '.[audio]' --no-build-isolation \
    -i https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Rust 可完全跳过（2026-08 第二套环境实测）**：`--no-build-isolation` 下 rust 前端
> optional=True 会降级为 warning，无需安装 cargo/rustc。
> 若必须走默认隔离构建（不带 `--no-build-isolation`），才需要 rust 工具链；
> 此时官方源（static.rust-lang.org / rsproxy.cn）可能卡死。
>
> **numpy 下载断线处理**：从 huaweicloud 拉 numpy 2.3.5（16.6MB）反复断线时，
> 先用阿里云预下载再本地安装：
>
> ```bash
> pip download numpy==2.3.5 --no-deps -i https://mirrors.aliyun.com/pypi/simple -d /tmp/np_dl
> pip install /tmp/np_dl/numpy-2.3.5-*.whl
> ```

## Step 2.4: 构建 vllm-ascend（releases/v0.26.0rc）

```bash
cd /home/{user}/code/vllm-ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
SOC_VERSION=$(npu-smi info -l 2>/dev/null | grep -i "Chip Name" | head -1 | awk '{print $NF}')
pip install -e . --no-deps --no-build-isolation
python -c "import vllm_ascend"   # 期望打印 "Platform plugin ascend is activated"
```

已知坑（与 framework-feature-enablement 的 `references/troubleshooting-vllm-omni.md` §E1 同源）：

- **CRLF**：Windows 打包上传的 `.sh` 报 `$'\r': command not found` →
  先 `find . -name '*.sh' -exec sed -i 's/\r$//' {} +`
- **catlass 子模块**：`csrc/third_party/catlass` 缺失时 build_aclnn 失败 →
  从其他用户已 clone 的 vllm-ascend 复制
- **patch 文件缺失**：`csrc/cmake/third_party/build/modules/patch/` 被上传排除（路径含 `build`）→
  需补传
- **torch 版本检查**：CMakeLists.txt 硬编码 `VERSION_EQUAL "2.10.0"`，与 torch 2.11.0 冲突 →
  sed 放宽为同时接受 `2.10.0` / `2.11.0`

## Step 2.5: 构建 vllm-omni（main）

```bash
cd /home/{user}/code/vllm-omni
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_OMNI_TARGET_DEVICE=npu
export VLLM_OMNI_VERSION_OVERRIDE=0.26.0   # 源码包无 .git 时版本检测返回 dev，导致 InvalidVersion
pip install -e . --no-build-isolation --no-cache-dir
```

> - 无 `.git` 目录时 `get_version()` 返回 `dev`，拼接 `+npu` 产生非法版本 →
>   必须设 `VLLM_OMNI_VERSION_OVERRIDE`
> - `/` 分区不足时 pip 构建缓存写满 → `pip cache purge` +
>   `export PIP_CACHE_DIR=/home/{user}/.cache/pip`
> - 安装完成后**重新装回 torch 2.11.0**：vllm-omni 依赖解析可能把 torch 降到 2.10.0
>   （vllm-ascend requirements 旧 pin），需
>   `pip install torch==2.11.0+cpu torchaudio==2.11.0 torchvision==0.26.0 -i https://download.pytorch.org/whl/cpu`

## Step 2.6: 构建 mindiesd（全栈环境内）

```bash
cd /home/{user}/code/MindIE-SD
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
sed -i 's|^source ${current_script_dir}/build_tik_ops.sh|# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh
python setup.py build_py
pip install -e . --no-deps
python -c "import mindiesd; print(mindiesd.attention_forward, mindiesd.fast_layernorm)"
```

> ⚠️ 上传打包时**不要排除源码树中的 `build/` 目录**（含 build_ops.sh/build_plugin.sh 等脚本），
> 否则 `python setup.py build_py` 报 `No such file or directory: .../MindIE-SD/build`。

安装完成的下一步：真实权重确认/下载（见 SKILL.md「权重确认与下载」与 references/weights-prep.md）。服务启动（`vllm serve`、curl 验证、特性叠加）
属 framework-feature-enablement，本技能止于安装。

## 维护与更新

当 vllm / vllm-ascend / vllm-omni 版本矩阵变化、950PR 构建路径调整或发现新的构建已知坑时，
按 dev-workflow 的复盘流程更新本文件（SKILL.md 正文只保留指针，细节以本文件为准）。
