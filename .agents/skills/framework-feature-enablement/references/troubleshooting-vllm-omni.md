# vLLM-Omni 全栈部署问题排查（950PR 源码构建）

## P0. 通用运行最佳实践（前置预防，先于排障）

> 本文件 E1-E3 是"出了问题怎么定位"；本节是"怎么做从一开始就不出这些问题的可复用规则"。
> 适用于 vLLM-Omni × MindIE-SD 优化闭环的基线冻结与各特性使能。共同要点：**任何"预期会改变
> 执行序列"的配置，落定后必须用输出对比/计数差异确认真实参与，配置"成功"不等于特性"生效"**。

- **基线注意力后端显式指定，禁止依赖默认路由**。安装了 editable mindiesd 的宿主，平台默认会把
  diffusion attention 路由到 FLASH_ATTN；若想冻结"未加速对照"（如 SDPA），必须显式传
  `--diffusion-attention-config '{"default":{"backend":"TORCH_SDPA"}}'`，并在服务日志确认 resolve 行，
  同时记录输出 md5/off-identity——口径冻结以"显式配置 + 日志 + 输出"三方一致为准，不靠"以为没装
  mindiesd 就默认 SDPA"。
- **多 mindiesd 树并存时按"能力实体"选树，不按目录名/新旧选树**。同机多棵 editable 树时，先列出
  目标特性依赖的能力并逐项验证：稀疏类查算子签名是否含该框架需要的参数（如 rf_v2 需要
  `video_spans`，缺则直接不兼容）；融合类查插件 md5 / import 是否成功（如 `mm_swiglu_mxquant`）。
  特性跑不通先核对"这棵树有没有这个能力"，再谈接线与算子。
- **配置参数语义先核源码/文档再用，不靠直觉取名**。例：稀疏 `end_step` 语义是"末 N 步保留
  dense"，不是"从第 N 步起稀疏"——误设成全程步数会让稀疏全程 staying-dense。改参数前先确认语义，
  改完后用输出对比验证确实参与（见下条）。
- **staying-dense / no-op 检查链（fail-closed）**：凡声明了稀疏/量化/缓存档但输出与上档
  逐字节相同（md5 一致），一律按"未生效"处理并排查，不得当作"零收益结论"入账；正确核验 =
  (1) kernel/计数差异（如 sparse 档 fused kernel 数、Qmm 数变化）+ (2) 输出 off-identity 两路并行。
- **并行形态先做 10 步可行性冒烟，再投全预算**：新并行组合（尤其 2 卡/小 rank 组）先 10 步快测；
  通信 init 报错（如 hcclCommInitRootInfoConfig error 15）即回退到已验证形态，不硬解。拓扑先查
  npu-smi 同岛选卡（跨岛 SYS 慢）。
- **同窗相邻对 + 反转 AB，热窗剔除留痕**：共享宿主多租户热节流可使同档 e2e 高 20-100%；对比结论
  取同一时间窗的相邻配置对（r1/r3 稳定对），反转顺序抵消漂移，异常窗（同档 s 级异常抬升）剔除并
  在 evidence 注明，不静默丢弃。
- **产物与运行留痕同目录**：serve 日志/输出媒体/证据按 runs/{date}_{model}_optimization/ 归档，
  双报表与 evidence.json 同目录，保证口径可回查（见 run-state.md 规范）。

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
├─ 请求超时（首图慢） → 950PR 首图含编译 warmup，Qwen-Image-2512 1024x1024 20 步约 5s，后续更快
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

## 维护与更新

当 vllm-omni 版本矩阵或 950PR 构建/启动行为变化时，按 dev-workflow 的复盘流程更新本文件。
