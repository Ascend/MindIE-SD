---
name: framework-integration
compatibility: vLLM + vLLM-Ascend + vLLM-Omni（950PR 源码构建路径）或 diffusers/transformers 或 DiffSynth-Engine（diffsynth_engine）；远端容器 + CANN；mindiesd 已部署
description: 三方推理框架对接与验证：vLLM-Omni 托管扩散模型（Ascend 950PR 源码构建全栈）、
             Cache DiT + diffusers、魔乐社区（modelscope），以及外部独立推理框架
             （DiffSynth-Engine 等）的 MindIE-SD compile 接入。覆盖已部署模型的框架侧验证
             （curl/1 步推理/特性叠加开关）、vLLM-Omni 全栈部署排障，
             与外部框架的部署/compile 适配/融合算子使能判断。
             当用户需要部署或验证 vLLM-Omni 托管的模型、确认量化/稀疏/Cache 开关生效、
             排查框架侧启动问题、或把 MindIE-SD 编译接入外部推理框架时使用此 skill。
             即使用户只提到"起 vllm 服务"、"模型在 vllm 里跑不通"、把 mindiesd 编译
             接到 diffsynth_engine / 其他推理框架、或问某个融合算子在外框架里是否生效
             而未明确说框架名，也应触发。
             由 dev-workflow 的部署验证阶段触发。
---

# 三方框架对接与验证

## 覆盖范围

| 框架 | 场景 | 入口 |
|---|---|---|
| vLLM-Omni | 扩散模型托管服务（Qwen-Image / Wan2.2 / MiniMax-H3），950PR 源码构建全栈 | §2 + `references/troubleshooting-vllm-omni.md` |
| Cache DiT + diffusers | from_pretrained 加载已部署模型，1 步推理验证 | §1 |
| 魔乐社区（modelscope） | 社区入口，量化/稀疏/Cache 特性叠加验证 | §1 |
| DiffSynth-Engine（外部框架） | 独立推理框架接入 MindIE-SD compile：部署 / compile 适配 / 融合算子使能判断 | §3 + `references/diffsynth-engine-notes.md` |

前置：模型已通过 ascend-deploy 部署到 NPU 设备，`import mindiesd` 成功。

---

## §1 部署验证（已部署模型）

验证已部署模型在真实权重下的推理正确性。按框架选择验证方法：

### 1.1 vLLM Omni 部署

```bash
# 1. 检查服务状态
curl http://localhost:8000/health

# 2. 发送 1 次推理请求
curl http://localhost:8000/generate -H "Content-Type: application/json" \
    -d '{"prompt": "test", "max_tokens": 1}'

# 验证: HTTP 200 + 输出非空
```

### 1.2 Cache DiT + diffusers 部署

```python
# from_pretrained 加载已部署模型
pipe = FluxPipeline.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
).to("npu")

# 跑 1 步推理
output = pipe("test prompt", num_inference_steps=1)

# 验证: 无异常、无 OOM、output.images[0] shape 合法
print(f"Output shape: {output.images[0].size}")
```

### 1.3 魔乐社区部署

按社区指定入口执行，重点检查特性叠加是否生效：

- 量化开关 → 权重精度是否符合预期
- 稀疏开关 → sparsity 参数是否生效
- Cache 开关 → 缓存命中日志有无

### 1.4 验证通过标准

| 检查项 | 标准 |
|--------|------|
| 推理无异常 | 无 `RuntimeError` / `OOM` / `CUDA error` |
| 输出合法 | shape > 0，非全零输出 |
| 显存正常 | 峰值 < 物理显存 90% |
| 特性叠加 | 量化/稀疏/Cache 开关生效 |

---

## §2 vLLM-Omni + MindIE-SD 全栈部署（Ascend 950PR 源码构建路径）

当目标是在远端容器内运行 **vLLM-Omni 托管的扩散模型**（Qwen-Image-2512 / Wan2.2 / MiniMax-H3 等），
需要安装完整栈：`torch + torch_npu + vllm + vllm-ascend + vllm-omni + mindiesd`。
官方预构建镜像（`quay.io/ascend/vllm-omni:*`）仅覆盖 Atlas A2/A3（aarch64）；
**Ascend 950PR / 950DT（x86_64）必须从源码构建**。

### 版本配套矩阵（以 950PR + vllm 0.26.0 为例，2026-08 实测可用）

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
> 从 [gitcode Ascend/pytorch releases](https://gitcode.com/Ascend/pytorch/releases) 选择 `v26.1.0-pytorch2.11.0`
> 下载配套 torch_npu（版本 tag 含 `pytorch2.11.0` 字样）。
> 不要用 vllm-ascend / vllm-omni 的 `requirements.txt` 里硬编码的 torch 版本（旧 pin，会降级 torch）。

### Step 2.1: 容器环境预检

```bash
docker exec <容器> bash -lc 'uname -a'                    # 确认 x86_64
docker exec <容器> bash -lc 'npu-smi info -l | head -20'  # 确认 Ascend950PR 及卡数
docker exec <容器> bash -lc 'python --version'            # 确认 3.12（cp312 wheel）
```

⚠️ **必须检查容器是否挂载 HCCL ranktable 目录**：
`/usr/local/Ascend/driver/topo/`（含 `950/atlas_350_*.json`）缺失时，
vllm 多卡启动会报 `hcclCommInitRootInfoConfig error code is 4`。
容器创建时需挂载或 `docker cp /usr/local/Ascend/driver/topo <容器>:/usr/local/Ascend/driver/topo`。

### Step 2.2: 安装 torch + torch_npu

```bash
# 容器内（cp312 与 Python 3.12 匹配）
pip install torch==2.11.0+cpu -i https://download.pytorch.org/whl/cpu
# 从 gitcode 下载 torch_npu wheel 后安装
pip install torch_npu-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"
```

### Step 2.3: 构建 vllm 0.26.0

```bash
cd /home/<user>/code/vllm-0.26.0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=$HOME/.cargo/bin:$PATH      # 需要 rust 工具链（setuptools-rust）
VLLM_TARGET_DEVICE=empty pip install -e '.[audio]' --extra-index-url https://download.pytorch.org/whl/cpu
```

> Rust 前端（`vllm/_rust_*.so`）默认 optional；但**首次构建卡在 `rustc -V` 通常是
> cargo 拉取 crates.io index 超时**。配置国内镜像：
>
> ```bash
> mkdir -p /root/.cargo
> cat > /root/.cargo/config.toml <<'EOF'
> [source.crates-io]
> replace-with = "rsproxy-sparse"
> [source.rsproxy-sparse]
> registry = "sparse+https://rsproxy.cn/index/"
> [net]
> git-fetch-with-cli = true
> EOF
> ```

### Step 2.4: 构建 vllm-ascend（releases/v0.26.0rc）

```bash
cd /home/<user>/code/vllm-ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
SOC_VERSION=$(npu-smi info -l 2>/dev/null | grep -i "Chip Name" | head -1 | awk '{print $NF}')
pip install -e . --no-deps --no-build-isolation
python -c "import vllm_ascend"   # 期望打印 "Platform plugin ascend is activated"
```

已知坑（详见 `references/troubleshooting-vllm-omni.md` §E1）：

- **CRLF**：Windows 打包上传的 `.sh` 报 `$'\r': command not found` → 先 `find . -name '*.sh' -exec sed -i 's/\r$//' {} +`
- **catlass 子模块**：`csrc/third_party/catlass` 缺失时 build_aclnn 失败 → 从其他用户已 clone 的 vllm-ascend 复制
- **patch 文件缺失**：`csrc/cmake/third_party/build/modules/patch/` 被上传排除（路径含 `build`）→ 需补传
- **torch 版本检查**：CMakeLists.txt 硬编码 `VERSION_EQUAL "2.10.0"`，与 torch 2.11.0 冲突 →
  sed 放宽为同时接受 `2.10.0` / `2.11.0`

### Step 2.5: 构建 vllm-omni（main）

```bash
cd /home/<user>/code/vllm-omni
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_OMNI_TARGET_DEVICE=npu
export VLLM_OMNI_VERSION_OVERRIDE=0.26.0   # 源码包无 .git 时版本检测返回 dev，导致 InvalidVersion
pip install -e . --no-build-isolation --no-cache-dir
```

> - 无 `.git` 目录时 `get_version()` 返回 `dev`，拼接 `+npu` 产生非法版本 → 必须设 `VLLM_OMNI_VERSION_OVERRIDE`
> - `/` 分区不足时 pip 构建缓存写满 → `pip cache purge` + `export PIP_CACHE_DIR=/home/<user>/.cache/pip`
> - 安装完成后**重新装回 torch 2.11.0**：vllm-omni 依赖解析可能把 torch 降到 2.10.0
>   （vllm-ascend requirements 旧 pin），需 `pip install torch==2.11.0+cpu torchaudio==2.11.0 torchvision==0.26.0 -i https://download.pytorch.org/whl/cpu`

### Step 2.6: 构建 mindiesd

```bash
cd /home/<user>/code/MindIE-SD
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
sed -i 's|^source ${current_script_dir}/build_tik_ops.sh|# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh
python setup.py build_py
pip install -e . --no-deps
python -c "import mindiesd; print(mindiesd.attention_forward, mindiesd.fast_layernorm)"
```

> 上传打包时**不要排除源码树中的 `build/` 目录**（含 build_ops.sh/build_plugin.sh 等脚本），
> 否则 `python setup.py build_py` 报 `No such file or directory: .../MindIE-SD/build`。

### Step 2.7: 启动 vLLM-Omni 服务

以 Qwen-Image-2512（950PR 8 卡）为例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve /data/models/Qwen-Image-2512 \
  --omni --host 0.0.0.0 --port 8091 --trust-remote-code \
  --num-gpus 8 --tensor-parallel-size 8 \
  --vae-use-tiling --vae-patch-parallel-size 8
```

验证：

```bash
curl -s http://127.0.0.1:8091/health                    # 期望 200 OK
curl -s http://127.0.0.1:8091/v1/models                  # 期望列出模型
curl -s http://127.0.0.1:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/models/Qwen-Image-2512","prompt":"a red teapot","size":"1024x1024","num_inference_steps":20,"seed":42}'
# 响应含 b64_json（PNG），解码后 file 校验应为 "PNG image data, 1024 x 1024"
```

其他已知启动坑（`references/troubleshooting-vllm-omni.md` §E2）：

- `ImportError: libxcb.so.1` → opencv-python 依赖 X11 库 → `dnf install -y libxcb xcb-util* libX11 ... mesa-libGL`
- 权重分片缺失 → `ValueError: ... weights were not initialized from checkpoint` →
  对照 `*.safetensors.index.json` 逐个核对分片，缺失的从 hf-mirror 补下载
- 950PR 上**不要设置** `MINDIE_SD_FA_TYPE`（该变量不适用于 950PR/950DT）

### 可选优化（950PR 推荐）

默认启用 `--quantization mxfp8`，无需逐层卸载；多卡推理用
`--usp 8 --ring 1 --text-encoder-tp-size 8 --vae-parallel-mode tile` 等。

---

## 故障排查

- `references/troubleshooting-vllm-omni.md` — vLLM-Omni 全栈构建/启动/运行期问题（§E1/E2/E3）
- 通用部署问题（SSH、docker exec、CRLF、环境依赖）→ `ascend-deploy/references/troubleshooting-tree.md`

---

## §3 DiffSynth-Engine（外部框架）接入 MindIE-SD compile

> 详细步骤见 `references/diffsynth-engine-notes.md`。本节为速览。

**适用场景**：把 `diffsynth_engine`（Qwen-Image 等扩散模型的独立推理框架）接入
MindIE-SD 编译，使 pattern matcher 的融合算子在其推理图上生效。

### 3.1 部署要点

- 独立 pip 包：增量传输到远端容器 + `pip install -e . --no-deps`
  （pyproject 锁旧版 transformers/diffusers，带依赖会破坏容器环境）
- 无 `.git` 时 setuptools-scm 报错 → `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DIFFSYNTH_ENGINE=1.0.0`
- mindiesd 用 compile 工作区：脚本内 `sys.path.insert(0, "<工作区>/mindie-sd-compile")`

### 3.2 compile 适配（3 处）

| 位置 | 改动 |
|---|---|
| `configs/base.py` + `args.py` | `PipelineConfig.compile_backend="mindie"` + `--compile-backend` CLI |
| `pipelines/base.py` | `compile_transformer_blocks` 对 `_repeated_blocks` 逐个 `submodule._compiled_call_impl = torch.compile(submodule._call_impl, backend=MindieSDBackend(), fullgraph=False)` |
| 模型层 | Qwen-Image RoPE 改实数域等价（命中 `qwen_rope_pattern`）；text encoder key 归一化（transformers>=5.x 布局） |

> ⚠️ `torch.compile(submodule, backend=...)` **不赋值不生效**，必须写入
> `_compiled_call_impl`（等价 `nn.Module.compile()`）。未赋值时 warmup 与 eager 一致、pattern 0 命中。

### 3.3 融合算子使能判断（与 dummy run 一致）

外部框架（真实权重 60 层）与 dummy run（随机权重 2 层）的**融合 kernel 使能集合一致**
（qwen_rope → `RotaryPositionEmbeddingV2` / `npu_rotary_mul`，qk_norm → `npu_rms_norm`，
残差 gate → `residual_gate_add_kernel`，调制 → `AdaLayerNormV2`，GELU → `FastGelu`）。

- **使能判断用 dummy run 即可**（pattern 匹配基于图结构，不依赖层数/权重）
- 三层证据（可靠性递增）：`MINDIE_LOG_LEVEL=DEBUG` 日志 → `graph_log_url` DOT 图 → `kernel_details.csv`
- ⚠️ 日志 2048 字符截断会误判 0 命中 → 以 DOT 图 / kernel_details.csv 为准
- ⚠️ 图命中 ≠ 运行期全部生效：`residual_gate_add` 对 4D attention 张量运行期 fallback，
  需看融合 kernel 实际执行次数
- **不做耗时比较**：dummy 与真实权重的耗时差异由层数主导，耗时评估必须
  真实权重 + kernel diff（见 compilation-dev Phase 6）

## Reference Files

- 🔍 `references/troubleshooting-vllm-omni.md` — 加载时机: vLLM-Omni 构建或启动遇到异常时
- 🔍 `references/diffsynth-engine-notes.md` — 加载时机: 把 MindIE-SD compile 接入
  DiffSynth-Engine 或类似外部推理框架（部署 / compile 适配 / 融合算子使能判断）时

## 维护与更新

当 vllm / vllm-ascend / vllm-omni 版本矩阵变化、950PR 构建路径调整、
或新增框架对接经验时，按 dev-workflow 的复盘流程更新本 skill。
