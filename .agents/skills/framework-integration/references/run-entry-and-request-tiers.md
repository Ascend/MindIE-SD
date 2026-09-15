# 运行入口与档位选择（服务启动 / torchrun / 请求级 task）

> **定位**：模型**已部署**（依赖装好、权重就位）之后「**怎么起、起哪一档、请求级怎么切任务**」。
> 安装与权重归 `env-install`；特性档位**该不该开**（量化 / 稀疏 / 缓存 / offload）归 `dit-perf-opt`；
> 并行档位**形态怎么选**归 `dit-parallel-opt`；验收口径归 `perf-gate`。
> 本文件只承接**运行入口与档位切换本身**；逐框架启动 recipe 不在此复制（单点在 §1/§2 指出的文件）。
> 路径占位符：`{repo}`（代码仓根）、`{model_weight_dir}`（权重根）、`{container}`、`{out}`。

## 1. vLLM-Omni：服务启动与请求级 task 档位

**启动 recipe 单点**：`vllm-omni-enablement.md` §2.1（env 前置、`vllm serve` 命令、健康探针与端点选型）
与 `cache-dit-enablement.md` §2.1（4 卡启动示例）——本文件**不复制**，只登记**档位语义**。

- **服务侧任务档**：`vllm serve ... --task-type {t2va|fl2va|ref2va}` 决定服务的默认任务分区。
- **serve 目标二选一**：
  - serve **模型根目录**（`{model_weight_dir}/{模型名}`）：pipeline 按请求档自动识别
    `FL2VA/` / `Ref2VA/` 分区子目录；
  - serve **分区目录**（`{model_weight_dir}/{模型名}/{分区}`）：把该分区固定为唯一可用档。
  落位约定与分区口径（含「哪个任务只需哪个分区」）见 `env-install/references/weights-prep.md` §2.2/§2.3。
- **请求级档位切换**：请求体 `extra_params.task = "t2va" / "fl2va" / "ref2va"` —— 同进程内按请求选任务档，
  **不必重启服务换档**。切档 = 切权重分区 ⇒ 请求前先确认该分区权重已就位且完整
  （完整性判定归 `env-install`，见 `weights-prep.md` §5）。
- **切档后须复核「档位真的生效」**：换 task 档会换 pipeline / 分区 / 部分默认值，
  **HTTP 200 只说明服务活着**；参与真实性的判据是计数契约 + 三层证据（`../SKILL.md` §1.4），
  换档后按需重采一次 kernel diff / 日志证据。

## 2. LightX2V：torchrun 运行入口与 `--config_json` 档位

环境就绪（`PLATFORM=ascend_npu` 下 `import lightx2v` 成功、权重就位）由 `env-install` 判定
（`env-install/references/lightx2v-env.md` §2/§3）；本节只讲**怎么跑、跑哪一档**。

```bash
docker exec {container} bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export PLATFORM=ascend_npu                        # 必须先于 import lightx2v（否则 ERR99999 类异常）
  export ASCEND_RT_VISIBLE_DEVICES={cards}
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export HCCL_NPU_SOCKET_PORT_RANGE=20000-21000     # 默认端口段被占时
  export PYTHONPATH={repo}/lightx2v:$PYTHONPATH
  cd {repo}/lightx2v
  torchrun --standalone --nproc_per_node={n} -m lightx2v.infer \
    --model_cls minimax_h3 --task t2av \
    --model_path {model_weight_dir}/MiniMax-H3 \
    --config_json {repo}/lightx2v/configs/platforms/ascend_npu/minimax_h3_t2av_sp_compile_5s.json \
    --save_result_path {out}.mp4
'
```

- **`--config_json` 就是档位**：档位文件（路径名自带档位语义，如 `…_sp_compile_5s.json`
  ≈ 序列并行 × compile 开关 × 时长/分辨率档）一次性固定该次运行的
  并行（`parallel.{tensor_p_size,seq_p_size,seq_p_attn_type,seq_p_a2a_backend}`）、
  特性开关（`use_compile` / `compile_backend` / `rms_type` / `rope_type` / `attn_type`）与负载（步数/尺寸/帧数）。
  档位键全表与合入版配置示例见 `lightx2v-enablement.md` §4。
- **换档 = 换 config_json**（不改代码）；对照实验必须**同档同卡组同窗口**，跨窗口绝对值不可比（`perf-gate`）。
- **并行档位语义不在本文件**：卡数 / 序列并行形态（USP / 复合 AllGather-KV×Ulysses / CP）/ 拓扑选卡与
  「哪个形态更快」的判据 → `dit-parallel-opt`（`parallel-form-selection-method.md`、
  `ascend-topology-bandwidth-diag.md`）；本文件只负责把档位传进去。
- 运行/配置细节（`compile_backend`、`hccl_eager`、kernel diff 方法）→ `lightx2v-enablement.md`。

## 3. 使能异常时的回退姿势（怎么退回）

档位开启后报错 / 劣化 / 静默无效时，**回退姿势**在本技能，**选哪一档、退到哪一档**归 `dit-perf-opt`
（`dit-perf-opt/references/resource-fallback-tiers.md`）：

- 先用 `../SKILL.md` §1.5 定位根因类别（mindiesd 侧 / 自研算子部署侧 / 框架适配侧），
  再按 §1.2「何时停在 API 阶段即可」判回退：**落在噪声阈值内或改变数值语义 ⇒ 回退（默认关）**，
  不留半开状态。
- **回退到 eager 路径**：关掉该档的开关 / 不起用该 backend（框架侧保持默认关），
  **保留实现与开关，不在安装层删代码**；回退结论按「经验 vs 探针」分类留痕（`.agents/README.md` §7）。
- 自研算子产物在运行期不可见（`aclnnXxx … inferShape function does not exist`）**不是档位问题**，
  属部署校验 → `operator-dev/references/custom-op-runtime-deploy-verify.md`。

## 维护与更新

当某框架的启动入口 / `--config_json` 档位键 / 请求级字段（`extra_params.*`）变化，
或新增框架的运行入口时，按 dev-workflow 的复盘流程更新本文件；逐框架机制仍以对应
`{framework}-*-enablement.md` 为真源。
