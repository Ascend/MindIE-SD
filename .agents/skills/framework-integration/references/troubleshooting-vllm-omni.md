# vLLM-Omni 全栈部署问题排查（950PR 源码构建）

## E1. 构建/安装阶段

```text
编译错误
├─ .sh 脚本报 $'\r': command not found
│   └─ Windows 打包上传的 CRLF 换行 → find . -name '*.sh' -exec sed -i 's/\r$//' {} +
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
│   └─ pip cache purge + export PIP_CACHE_DIR=/home/<user>/.cache/pip（挂到大分区）
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
│       → docker cp /usr/local/Ascend/driver/topo <容器>:/usr/local/Ascend/driver/topo
│       （容器重启后丢失，需重建或持久化）
├─ ImportError: libxcb.so.1: cannot open shared object file
│   └─ opencv-python 依赖 X11 → dnf install -y libxcb xcb-util* libX11 libXext mesa-libGL ...
├─ ValueError: "The quantization config is None, and the following weights were not initialized"
│   └─ 权重分片缺失（对照 *.safetensors.index.json 的 weight_map 逐分片核对）
│       → 缺失分片从 hf-mirror（https://hf-mirror.com/<org>/<model>/resolve/main/...）补下载
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
└─ attention backend 未生效 → 确认日志 "Resolved diffusion attention backend 'FLASH_ATTN'"
    （mindiesd 已安装时平台默认 FLASH_ATTN；缺失则回退 SDPA，检查 mindiesd 是否 import 成功）
```

## 维护与更新

当 vllm-omni 版本矩阵或 950PR 构建/启动行为变化时，按 dev-workflow 的复盘流程更新本文件。
