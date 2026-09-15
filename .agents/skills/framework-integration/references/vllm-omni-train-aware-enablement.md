# vLLM-Omni：训练感知优化的开启方式（框架差异记录）

> 定位：本文件只放 **vLLM-Omni 侧特有**的内容 —— 开启方式（命令 / 开关 / 日志契约）、
> 本框架的模型侧契约、并行与负载前提。**通用方法与判定纪律见
> `train-aware-lossy-method.md`**（分类 / 三条前置契约 / 协同定位 / 质量分层 / 归因链）。
> 支持面状态（✅/🟡/❌/❓）见 `framework-support-matrix.md`「训练感知」表。
> **本文件不写实测加速比**（比例随模型与框架能力变化，不可迁移）；产物坐标与实测记录见 §6。

## 1. `少步蒸馏`：框架原生支持（零代码改动）

### 1.1 开启姿势

| 位置 | 内容 |
|---|---|
| 服务侧 | `--task-type fl2va --lora-backend peft --lora-path <蒸馏适配器>` |
| 请求侧 | `num_inference_steps=<少步数>` + `lora={"name": …, "path": …, "scale": 1.0}` |

- **版本边界**：该 LoRA/蒸馏链路**仅 0.28 有**（0.26 无），跨版本前先查支持矩阵。
- 适配器为 native 布局（`rank 64 / alpha 64`），目标模块由适配器自带清单决定。

### 1.2 前置契约落点（本框架的具体形态）

- 模型 `model_index.json` **不得** pin `base_schedule`（与请求侧步数冲突）。
- 适配器 metadata 声明 `base_schedule` 时，`num_inference_steps` 语义 = **denoiser 评估次数**
  （不是 sigma 区间数，详见方法文件 §3.1）。

### 1.3 日志契约（防 no-op，必须逐项核对）

```text
Loaded LoRA model: num_modules=259      # 目标模块计数
Increasing max LoRA rank: 0 -> 64       # rank 生效
<每个 worker> Activating adapter        # 各 worker 均激活
```

## 2. `VAE解码替换`：框架未提供 → fork 探针实现 `[探针]`

- **框架支持面**：`❌ 待开发`（框架无该组件接口）。
- `[探针]` 落点：框架源码树内**新增解码器模块**（与原生 VAE **同接口** + 布局自动判别 +
  **fail-closed**）+ pipeline 构造处按 env 分派；**未设 env = 原生路径不变**。
- **未合入上游 / `.bak` 保留 / 默认关** ⇒ 按本仓「经验 vs 探针」判为探针。
- 布局判别（fail-closed，认不出即抛错而非静默误载）：
  - `taehv-temporal`：官方时序布局，`decoder.` 前缀、128 张量（64 解码 + 64 编码）、F16；
    构造后参数 **9,868,236** 与结构逐位吻合。
  - `tae-2d`：逐帧 2D 布局（裸索引、81 张量、F32）—— 对本模型属**架构级错配**（见方法文件 §3.3）。
- 开关与对拍用 env：

| env | 作用 |
|---|---|
| `OMNI_H3_TAE` | 使能替换（未设 = 原生路径） |
| `OMNI_H3_TAE_DENORMALIZE` | 恢复「替换件自行再做一次 `*std+mean`」的旧行为，**仅用于 A/B 对拍定位错因** |
| `OMNI_H3_TAE_OUTPUT_PERCENTILE` | 输出逐帧百分位拉伸（后处理权重相关，换 checkpoint 必须重验，见方法文件 §5.1） |

## 3. 该模型侧契约（MiniMax-H3 视频 VAE；换模型须重新逆向）

### 3.1 帧数契约（解码器替换的几何前置）

```text
输出帧数 out = 4T − 3·⌈T/5⌉        （T = latent 帧数）
实测命中：T=37 → 124、T=107 → 362（9/9 全中）
配置来源：video_vae/config.json 的 vae_clip_length=17 + vae_token_drop=3
交叉校验（参考实现）：upscale_ratio(a)=(a−2)//5*17+5、downscale_ratio(a)=(a−1)//17*5+2
```

### 3.2 latent 约定（本框架最容易踩的一条）

- 本框架的视频 VAE **包装层自己执行** `latent * std + mean`。
- 参考实现（ComfyUI 生态）的 VAE checkpoint 把 `latents_mean/std` 作为**权重**存着、在 VAE
  **内部**反归一化 ⇒ 两侧消费的都是**已归一化** latent；参考实现的 `process_input = identity`
  意思就是「**原样喂入**」。
- 因此替换件**不得**再自行归一化（多算一次会过驱动头部 `tanh` 类非线性，见方法文件 §5.4）。

### 3.3 参考实现的判定契约（该组件的构造指纹）

末层卷积输出 **12 = 3 × patch²** 通道（配 `pixel_shuffle`）+ 因果记忆块（`MemBlock` 记忆）+
时序上采样 **×4** + 空间 **×16**（3 次 ×2 × patch 2）；`latent_channels=24`、`patch_size=2`、
`n_f=[256,128,64,64]`、`decoder_time_upscale=(False,True,True)`；`scale_factor = 1.0`。
按此逐 index 复刻 → `strict=True` 载入成功、参数计数吻合。

## 4. 并行与负载前提（本框架的约束）

```text
--tensor-parallel-size 1 --usp 4 --ring 1 --enable-distributed-layerwise-offload
--text-encoder-tp-size 4 --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling
```

- `TP1` 使 DiT 权重在每卡**全量复制**（超出单卡容量）⇒ **分布式逐层 offload 强制**；
  日志判据 `Distributed layer-wise offloading enabled on <N> blocks …`。
- `--text-encoder-tp-size` **依赖并行形态**（本组合观测，换并行形态 = 重判）：序列并行（USP）形态须与
  DiT world 对齐（否则子组为 `None` 触发 assert）；**纯 TP 形态（`usp=1`）在本组合下整体不可用，与
  tetp 取值无关**（对齐到 DiT world → 视觉 seam 报 `pixel_values and image_grid_thw must be provided
  together`；改用默认 `1` → 更晚在 `_build_denoise_inputs` 报 `IndexError: tuple index out of range`）。
  详见 `vllm-omni-enablement.md` §2 并行前提。
- 环境隔离：`VLLM_OMNI_DISABLE_VLLM_ASCEND=true`、`VLLM_PLUGINS=""`。
- 负载上限：`duration` ∈ [4, 15] s、`fps` 固定 24（超出即被拒）。

## 5. 与免训练有损档叠加时的框架观察

> 本节是该框架 × 该模型 × 该规模下的**观测**，**不是可移植的执行序**；换框架 / 模型 / 规模须重判
> （判据见 `train-aware-lossy-method.md` §6）。

- 少步蒸馏档上，**量化 / 稀疏照常生效**（可叠，见支持矩阵「训练感知」表）。
- **少步档 `Cache` 结构性失效**：该框架的 cache 预热步数参数会吃满少步档的全部步数预算 ⇒
  开 / 关两档由两次独立 serve 产出的 mp4 **md5 相同**（字节级等价判据，见方法文件 §4.3）；
  少步档应**直接整体剔除 Cache**，不要记成「收益近似为零」。
- 换入权重档落地后须按 `post-enable-review` 重审其余维度**是否仍生效**（不只是复测墙钟）。

## 6. 产物坐标与实测记录（不入 skills）

- 会话产物目录：`{run_results_dir}` —— 各档 `*.mp4`、`*_serve.log`（含适配器/解码器装载契约与
  逐阶段 profiler 计时）、质量与协同 JSON、报表与 `evidence.json`。
- 权重坐标：`{model_weight_dir}/MiniMax-H3/FL2VA`（BF16 主体）、
  蒸馏适配器 `{model_weight_dir}/flashgen-lora/*.safetensors`、
  预览级解码器 `{model_weight_dir}/h3-tae-official/taeh3.safetensors`（时序布局）。
- 代码落点（`[探针]`）：框架源码树内新增解码器模块 + pipeline env 分派；原文件 `.bak` 保留。
