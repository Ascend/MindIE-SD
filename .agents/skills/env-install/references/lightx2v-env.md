# LightX2V 调优环境（MindIE-SD 接入实测环境）

> 定位：LightX2V 的部署形态与 vLLM-Omni 不同——**editable 源码 + `PLATFORM` 环境变量驱动**，
> 无独立服务进程。本文记录把该环境从零复现/更新到就绪的姿势（Ascend 950PR 实测）。
> 边界：止于「环境就绪」（import 成功、版本配套、权重就位）；运行/使能姿势见
> `framework-feature-enablement/references/lightx2v-mindiesd-case.md`。

## 1. 环境形态（先理解再动手）

| 项 | 形态 |
|---|---|
| 代码 | 源码目录 + `pip install -e`（editable），远端常非 git 部署 |
| 运行时 | `docker exec` 进容器 → source CANN env → `torchrun` 启动推理 |
| 模型 | 共享模型目录（容器与宿主机挂载一致） |

## 2. 版本配套（950PR 实测）

| 组件 | 版本 | 说明 |
|---|---|---|
| python | 3.12.x | 容器内置 |
| torch / torch_npu | 2.11.0 / 2.11.0（主版本匹配） | 使用前 source CANN env |
| CANN toolkit | 9.1.0（驱动 25.7.rc1 系） | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| lightx2v | 0.1.0（editable） | 应含上游 #1471（compile backend + hccl_eager a2a） |
| mindiesd | 3.1.0（editable） | 与 lightx2v 同容器 |

## 3. 安装与就绪验证

```bash
# 1) 准备代码（远端非 git 部署：直接 rsync/scp 源码树到共享目录，容器内即可见）
# 2) 容器内 editable 安装（首次；之后源码更新无需重装）
docker exec {container} bash -c 'cd {repo}/lightx2v && pip install -e . --no-deps'
docker exec {container} bash -c 'cd {repo}/MindIE-SD && pip install -e . --no-deps'
# 3) 就绪验证
docker exec {container} bash -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  export PLATFORM=ascend_npu; \
  python -c "import lightx2v, mindiesd, torch_npu; print(\"env ok\")"'
```

- ⚠️ **`import lightx2v` 前必须 `export PLATFORM=ascend_npu`**：否则设备初始化按默认平台走，
  报 `ERR99999 UNKNOWN application exception` 类异常（不是代码问题）
- editable 安装后更新源码（`rsync` 新文件）直接生效，**无需重装、无需重编译**；
  mindiesd 同理
- `npu-smi info` 检查卡 Health（避开 Alarm 卡）；性能对比必须固定同卡组

## 4. 运行入口（环境侧）

```bash
docker exec {container} bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export PLATFORM=ascend_npu
  export ASCEND_RT_VISIBLE_DEVICES={cards}
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export HCCL_NPU_SOCKET_PORT_RANGE=20000-21000   # 端口 16666 冲突时
  export PYTHONPATH={repo}/lightx2v:$PYTHONPATH
  cd {repo}/lightx2v
  torchrun --standalone --nproc_per_node={n} -m lightx2v.infer \
    --model_cls minimax_h3 --task t2av \
    --model_path {model_dir}/MiniMax-H3 \
    --config_json {repo}/lightx2v/configs/platforms/ascend_npu/minimax_h3_t2av_sp_compile_5s.json \
    --save_result_path {out}.mp4
'
```

运行/配置细节（compile_backend、`hccl_eager`、kernel diff 方法）→
`framework-feature-enablement/references/lightx2v-mindiesd-case.md`。

## 5. MiniMax-H3 权重（LightX2V t2av 视角）

- 路径 `{model_weight_dir}/MiniMax-H3`（完整 ~269G；下载姿势见「权重确认与下载」）
- **LightX2V t2av 实际需要**：`transformer`（62G）+ `text_encoder`（63G）+ `vae`（9.8G）+
  `audio_vae` / `audio_scheduler` / `scheduler` / `tokenizer` / `processor` / `configuration.json`
- **`FL2VA`（135G）不需要**——它是其他任务组件（vLLM-Omni 某些任务格式才需要）；
  为 LightX2V t2av 准备权重时跳过 FL2VA 可省约一半存储
- 校验：`{root}/transformer/model.safetensors.index.json`、`{root}/text_encoder/...` 存在，
  `find {root} -name '*.incomplete'` 为空

## 6. 共享机注意

调优机常有多租户：其他服务的推理进程可能占用部分卡（vllm / rtp_llm 等）。实验前：

- `npu-smi info` 与 `npu-smi info -t proc-mem` 查看卡占用与进程归属，**不要 kill 他人进程**
- 4 卡（USP4）实验需要 4 张 Health=OK 的空闲卡；凑不齐时先与占用方协调，或改用可用卡组
  并保持对比同卡组（口径一致性优先于卡编号）
