# LightX2V 调优环境（MindIE-SD 接入实测环境）

> 定位：LightX2V 的部署形态与 vLLM-Omni 不同——**editable 源码 + `PLATFORM` 环境变量驱动**，
> 无独立服务进程。本文记录把该环境从零复现/更新到就绪的姿势（快照环境实测；读数与配置坐标见归档）。
> 边界：止于「环境就绪」（import 成功、版本配套、权重就位）；**运行入口与 `--config_json` 档位**见
> `framework-integration/references/run-entry-and-request-tiers.md` §2，
> 使能姿势见 `framework-integration/references/lightx2v-enablement.md`。

## 1. 环境形态（先理解再动手）

| 项 | 形态 |
|---|---|
| 代码 | 源码目录 + `pip install -e`（editable），远端常非 git 部署 |
| 运行时 | `docker exec` 进容器 → source CANN env → `torchrun` 启动推理（命令与档位见 §4 指针） |
| 模型 | 共享模型目录（容器与宿主机挂载一致） |

## 2. 版本配套（一次环境快照；换机型 / 换框架版本须按 §1 推导顺序重查）

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

## 4. 运行入口（不属本技能）

**止于环境就绪**：本技能不登记运行入口与档位。`torchrun` 运行命令、`--config_json` 档位语义
（并行 × compile × 时长/分辨率档）、请求/配置切换与档位报错回退 →
`framework-integration/references/run-entry-and-request-tiers.md` §2；
并行档位**形态怎么选**归 `dit-parallel-opt`（`parallel-form-selection-method.md`）；
使能细节（`compile_backend`、`hccl_eager`、kernel diff 方法）→
`framework-integration/references/lightx2v-enablement.md`。

环境侧**就绪判据**（本技能负责）：§3 的 `PLATFORM=ascend_npu` 下 `import lightx2v, mindiesd, torch_npu`
成功 + §2 版本配套就位 + §5 权重分区与完整性。

## 5. MiniMax-H3 权重（LightX2V t2av 视角）

- 路径 `{model_weight_dir}/MiniMax-H3`（完整 ~269G；下载姿势见「权重确认与下载」）
- **LightX2V t2av 实际需要**：`transformer`（62G）+ `text_encoder`（63G）+ `vae`（9.8G）+
  `audio_vae` / `audio_scheduler` / `scheduler` / `tokenizer` / `processor` / `configuration.json`
- **`FL2VA`（135G）不需要**——它是其他任务组件（vLLM-Omni 某些任务格式才需要）；
  为 LightX2V t2av 准备权重时跳过 FL2VA 可省约一半存储
- 校验：`{root}/transformer/model.safetensors.index.json`、`{root}/text_encoder/...` 存在，
  `find {root} -name '*.incomplete'` 为空

## 6. 共享机与卡组（不属本技能，只留判据）

调优机常有多租户：其他服务的推理进程可能占用部分卡（vllm / rtp_llm 等）。选卡、占用与进程归属
**不属本技能**：选空闲卡与进程查询姿势 → `remote-access/SKILL.md`「空闲卡选择」；
卡组可用性（`Health=OK` ≠ 组可用）、拓扑判定与**不要 kill 他人进程**的共享机纪律 →
`dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §1/§4。

> 对安装侧的唯一要求：装完/验证时就近记录**本次可用的卡组与卡状态**，供后续实验沿用同一卡组
> （口径一致性优先于卡编号）。

## 维护与更新

当 LightX2V 部署形态（editable + `PLATFORM` 环境变量）、版本配套或权重分区口径变化时，
按 dev-workflow 的复盘流程更新本文件；运行入口与档位更新到
`framework-integration/references/run-entry-and-request-tiers.md`，本文件只留环境就绪判据。
