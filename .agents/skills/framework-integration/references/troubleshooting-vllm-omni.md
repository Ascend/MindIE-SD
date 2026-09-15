# vLLM-Omni 全栈部署问题排查（950PR 源码构建）

## P0. 运行最佳实践（前置预防）→ 编排层口径已单点化

本节原载的**编排级机制**（基线冻结与"配置+日志+输出"三方一致、多次跑选树、参数语义先核、
staying-dense / no-op 检查链、并行形态可行性冒烟、同窗相邻对 + 反转 AB、产物与运行留痕同目录）
属编排契约与测量口径，**单点化到编排层**，本文件不复述：

- 产物目录 / 运行态 / 报表契约 → `model-auto-optimization`（`references/{artifact-layout,run-state}.md`、
  `workflows/optimization-flow.md`）
- 同窗 A/B、噪声阈值、探索态 vs 验收态、no-op / md5 判据 → `perf-gate`
- 本框架侧的**基线注意力后端显式指定**（`--diffusion-attention-config` 冻结 SDPA 对照）写在
  `vllm-omni-enablement.md` §3.1（属该框架的开关写法）

共同要点（一句话）：**任何"预期会改变执行序列"的配置，落定后必须用输出对比 / 计数差异确认真实
参与——配置"成功"不等于特性"生效"。**

## E1. 构建/安装阶段

```text
编译错误
├─ .sh 脚本报 $'\r': command not found
│   └─ Windows 打包上传的 CRLF 换行 → find . -name '*.sh' -exec sed -i 's/\r$//' {} +
├─ pip: "configuration error: `project.license` must be valid exactly by one definition"
│   └─ 容器 setuptools 太旧（默认 67.4，vllm 要求 >=77.0.3,<81.0.0）
│       → pip install "setuptools>=77.0.3,<81.0.0"（仅 --no-build-isolation 路径需要）
├─ rustc/cargo 缺失或 rustup 安装卡死（static.rust-lang.org / rsproxy.cn 均不通）
│   ├─ 首选：跳过 rust —— export VLLM_REQUIRE_RUST_FRONTEND=0 + --no-build-isolation
│   │   （vllm 0.26 rust 前端 optional=True，降级为 warning）
│   └─ 备选：dnf install -y cargo（openEuler 源有 cargo 1.90，但 rust-std 下载也可能超时）
├─ numpy 下载反复断线（huaweicloud Connection interrupted）
│   └─ pip download numpy==2.3.5 --no-deps -i https://mirrors.aliyun.com/pypi/simple -d /tmp/np_dl
│       → pip install /tmp/np_dl/numpy-2.3.5-*.whl（阿里云稳定）
├─ build_aclnn.sh: "dependency catlass is missing, try to fetch it..." 然后 fetch failed
│   ├─ catlass 是 git submodule（.gitmodules 指定 gitcode.com/cann/catlass.git）
│   └─ 无 .git / 网络不通时 → 从其他用户已 clone 的 vllm-ascend 复制 csrc/third_party/catlass
├─ "protobuf_25.1_change_version.patch: No such file or directory"
│   └─ patch 目录路径含 build（csrc/cmake/third_party/build/modules/patch/），
│       打包脚本误排除了整个 build 目录 → 补传 patch 文件
├─ "Expected PyTorch version 2.10.0, but found 2.11.0"
│   └─ vllm-ascend CMakeLists.txt 硬编码 VERSION_EQUAL "2.10.0"
│       → sed 放宽为同时接受 2.10.0/2.11.0
├─ pip: "Invalid version: 'dev+npu'"
│   └─ 源码包无 .git，setuptools_scm 返回 dev → 设 VLLM_OMNI_VERSION_OVERRIDE=0.26.0
├─ pip: "No space left on device"（构建 wheel 缓存写满 / 分区）
│   └─ pip cache purge + export PIP_CACHE_DIR=/home/{user}/.cache/pip（挂到大分区）
├─ pip 依赖解析把 torch 降级到 2.10.0
│   └─ vllm-ascend/vllm-omni 的 requirements.txt 是旧 pin
│       → 装完后重新 pip install torch==2.11.0+cpu torchaudio==2.11.0 torchvision==0.26.0
└─ mindiesd: "No such file or directory: .../MindIE-SD/build"
    └─ 源码树 build/ 目录含构建脚本，打包时被误排除 → 补传 build/*.sh
```

## E2. 启动阶段（vllm serve --omni）

```text
启动失败
├─ "Orchestrator initialization failed" + "hcclCommInitRootInfoConfig error code is 4"
│   └─ 容器缺 HCCL ranktable：/usr/local/Ascend/driver/topo/950/atlas_350_*.json
│       → docker cp /usr/local/Ascend/driver/topo {容器}:/usr/local/Ascend/driver/topo
│       （容器重启后丢失，需重建或持久化）
├─ ImportError: libxcb.so.1: cannot open shared object file
│   └─ opencv-python 依赖 X11 → dnf install -y libxcb xcb-util* libX11 libXext mesa-libGL ...
├─ ValueError: "The quantization config is None, and the following weights were not initialized"
│   └─ 权重分片缺失（对照 *.safetensors.index.json 的 weight_map 逐分片核对）
│       → 缺失分片从 hf-mirror（https://hf-mirror.com/{org}/{model}/resolve/main/...）补下载
├─ 版本检测：get_device_type 报 soc_version 不支持
│   └─ 950PR 需 SOC_VERSION=ascend950pr_9579（setup.py 自动从 npu-smi 识别，无需手设）
└─ 950PR 上设置 MINDIE_SD_FA_TYPE 导致算子路由异常
    └─ 950PR/950DT 不适用该变量，删除即可
```

## E3. 运行期

```text
生成失败 / 输出异常
├─ 请求超时（首图慢） → 950PR 首图含编译 warmup，Qwen-Image-2512 1024x1024 20 步首图约秒级（明显慢于后续步），后续更快
├─ 显存不足 → 950PR 每卡 128GB；多卡用 --tensor-parallel-size 8 / --usp 8 分摊
├─ attention backend 未生效 → 确认日志 "Resolved diffusion attention backend 'FLASH_ATTN'"
│   （mindiesd 已安装时平台默认 FLASH_ATTN；缺失则回退 SDPA，检查 mindiesd 是否 import 成功）
├─ POST /v1/images/generations → 500 "Missing preprocess images that should have been
│   created by the preprocess function"
│   └─ 模型是 **Edit/I2I 类**（Qwen-Image-Edit-2511 等 QwenImageEditPlusPipeline），
│       必须用 /v1/images/edits（multipart，带 image=@文件 + prompt），
│       不能用 /v1/images/generations（那是纯 T2I 端点）
└─ 端点选型速查：
    ├─ QwenImagePipeline（Qwen-Image / Qwen-Image-2512）→ /v1/images/generations（仅 prompt）
    └─ QwenImageEditPlusPipeline（Qwen-Image-Edit-2511）→ /v1/images/edits（multipart）
```

## 失效信号与复核

- **E1 的条目都是「钉版修补」，随 vllm-omni / vllm-ascend 提交与基础镜像过期**：逐条按原命令重跑一次源码构建——`Expected PyTorch version 2.10.0, but found 2.11.0`（sed 放宽）、`Invalid version: 'dev+npu'`（`VLLM_OMNI_VERSION_OVERRIDE`）、`project.license must be valid exactly by one definition`（升级 setuptools）、`dependency catlass is missing`、`protobuf_25.1_change_version.patch: No such file or directory`、`No such file or directory: .../MindIE-SD/build`，哪条不再出现即已修，删掉该绕行。
- **E1 的 rust / numpy / 磁盘三条是「可选依赖与网络」绕行**（`VLLM_REQUIRE_RUST_FRONTEND=0`、`pip download numpy==2.3.5` 走镜像、`pip cache purge` + 换 `PIP_CACHE_DIR`）：换源或换分区后重跑构建，若直接安装即成功，这三条降级为历史记录。
- **E2 的 `hcclCommInitRootInfoConfig error code is 4` 绑容器挂载与单板**：重建容器后重跑 `vllm serve --omni` 看该错误码是否复现（复现 ⇒ ranktable 仍缺，按原命令 `docker cp` 补齐）；`get_device_type` 报 soc_version 不支持时用 `npu-smi info -l` 核对型号，型号一变则 `SOC_VERSION=ascend950pr_9579` 与 `MINDIE_SD_FA_TYPE` 两条的适用性都要重新核对。
- **E3 的两条结论绑版本默认值与模型类**：起服务后 grep 日志是否仍打印 `Resolved diffusion attention backend 'FLASH_ATTN'`（若回落 SDPA，先复核 mindiesd 是否 import 成功）；端点选型按服务实际加载的 pipeline 类重判——若 `QwenImageEditPlusPipeline` 不再报 `Missing preprocess images that should have been created by the preprocess function`，说明选型表已过期。

## 维护与更新

当 vllm-omni 版本矩阵或 950PR 构建/启动行为变化时，按 dev-workflow 的复盘流程更新本文件。
