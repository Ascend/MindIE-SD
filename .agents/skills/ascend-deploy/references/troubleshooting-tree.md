# 故障排查决策树

## 问题分类

### A. 部署失败

```text
部署失败
├─ SSH 认证失败 → 确认 IP/用户名/密码，检查网络连通性
├─ 编译错误
│   ├─ CANN 环境未 source → source /usr/local/Ascend/ascend-toolkit/set_env.sh
│   ├─ build_tik_ops.sh 失败 → 注释掉 build_ops.sh 中的 source build_tik_ops.sh 行
│   └─ 缺少编译依赖 → pip install build wheel cmake
└─ 文件缺失（首次部署）
    ├─ model/__init__.py 缺失 → 全量传输该目录
    └─ 新增 .py 文件未识别 → 重新执行 pip install -e .
```

### B. 运行时 OOM

```text
OOM
├─ CPU OOM（构造阶段）
│   └─ from_config 时内存不足 → 使用 meta→to_empty 构造方式
├─ NPU OOM（推理阶段）
│   ├─ 模型 > 显存 → 启用 CPU offload
│   ├─ 中间激活值过大 → 减少 batch / 降低分辨率
│   └─ CFG 双分支翻倍 → 关闭 CFG 使用 guidance_scale=1.0
└─ 排查步骤
    1. npu-smi info -t memory -i <device_id> 检查当前显存
    2. 确认是否有其他进程占用
    3. 选择空闲卡或启用 offload
```

### C. 输出异常（NaN / 黑图 / 花屏）

```text
输出异常
├─ NaN
│   ├─ 精度问题 → 检查 bf16 vs fp16，某层可能溢出
│   └─ 算子精度 → 检查是否有算子返回 inf/nan（npu_add_rms_norm_dynamic_quant 已知 crash）
├─ 黑图（全零输出）
│   ├─ VAE decode 未触发 → 检查 output_type 参数
│   └─ latent 全零 → 检查 transformer 输出
├─ 花屏 / 图像错乱
│   ├─ CFG 参数错误 → guidance_scale 是否正确传递
│   ├─ tokenizer 异常 → from_pretrained 返回 bool 而非 tokenizer（二进制文件被 CRLF 损坏）
│   └─ latent channel 数不匹配 → VAE config 与 transformer 输出不一致
└─ 排查步骤
    1. 先用 bf16 验证输出正确性
    2. 逐组件检查：text_encoder → transformer → VAE
    3. 对比 eager vs compiled 输出（cosine similarity）
```

### D. NPU 崩溃（HCCL 超时 / 算子 crash）

```text
NPU 崩溃
├─ 症状: 进程卡住无响应
│   ├─ npu-smi info -l 检查卡状态
│   ├─ 卡处于 ERROR 状态 → 需要复位
│   └─ HCCL 拓扑问题 → 检查多卡互联
├─ 症状: core dump / 进程退出
│   ├─ 算子 AC L error → 查询 CANN 错误码文档
│   ├─ triton 版本错误 → pip uninstall triton && pip install triton-ascend
│   └─ 特定算子 crash → 降级为 eager 路径、记录到 ascend-ops.md
├─ 症状: 输出随机错误（非 NaN）
│   └─ 显存越界 → npu-smi 检查显存，调整分辨率或启用 offload
└─ 排查步骤
    1. 查看容器日志: docker logs <container>
    2. 查看 CANN 日志: /var/log/npu/slog/
    3. 用单卡最小配置复现（bf16 / 最低分辨率 / 1 步）
    4. 若稳定复现 → 标记为算子兼容性问题
```

### E. 本地环境（Windows 开发机）

```text
本地环境问题
├─ git/curl HTTPS 握手失败: schannel: AcquireCredentialsHandle failed:
│   SEC_E_NO_CREDENTIALS (0x8009030e) —— 沙箱/受限进程下 schannel 拿不到凭据，
│   curl 与 git 均受影响；python urllib(openssl) 正常（可用此判断是否为 schannel 问题）
│   └─ 解决: git -c http.sslBackend=openssl fetch；或 git config --global http.sslBackend openssl
├─ deploy_to_remote.py 路径语义与仓库实际远端不一致
│   ├─ 脚本把文件传到 {workspace}/{local_root.name}（如 /home/<user>/code/MindIE-SD_compile）
│   │   但 build 却在 {workspace}/MindIE-SD 内执行 → 与本仓实际远端
│   │   （如 /home/<user>/code/mindie-sd-compile）不一致时不要直接用，改精准 SFTP 同步
│   └─ 注意: refs/ 等大目录不在脚本排除列表，全量传输会带上 profiling 数据
└─ 凭据安全: refs/ssh_helper.py 明文存 host/user/password —— 属历史遗留，
    新代码改用环境变量或 SSH key；报告/日志中不回显密码
```

## 工具速查

| 工具 | 用途 |
|------|------|
| `npu-smi info -l` | 列出所有 NPU 卡状态 |
| `npu-smi info -t memory -i 0` | 查看卡 0 显存 |
| `docker logs <container>` | 查看容器运行日志 |
| `python -c "import torch_npu; print(torch_npu.__version__)"` | 确认 TorchNPU 版本 |
| `npu-smi info -t usages -i 0` | 查看卡 0 使用率 |

## 维护与更新

当部署/编译故障模式变化或新增高危问题时，按 dev-workflow 的复盘流程更新本文件。
