# 性能分析工作流

## 端到端流程

```text
本地编码 → 部署到远端 → Profiling 采集 → 压缩回传 → 本地分析 → 经验归档
```

### Step 1: 部署代码到远端

使用 ascend-deploy skill 的 `deploy_to_remote.py`：

```bash
python ascend-deploy/scripts/deploy_to_remote.py
```

### Step 2: Profiling 采集 + 回传

使用 profiling-collection skill 的 `collect_profile.py`：

```bash
python profiling-collection/scripts/collect_profile.py \
    --host <IP> --user <用户名> --password <密码> \
    --container <容器名> --script wan_infer.py --device-id 0
```

自动完成：

1. SSH 连接远端昇腾设备
2. Docker exec 运行 profiling 脚本
3. Docker exec tar czf 压缩结果
4. SFTP 下载压缩包到本地

### Step 3: 本地分析

使用 `analyze_trace.py` 对 Profiling 数据做三层递进分析：

```bash
tar xzf profile_l1.tar.gz
python analyze_trace.py --profile-dir ./profile_l1 --output-dir ./
```

### Step 4: 经验归档

按 dev-workflow 复盘流程更新相关 skills。

## 数据流

```text
远端 NPU 容器
│
├── torch_npu.profiler (level=l1)
│   └── CANN Profiler 原始数据
│       ├── kernel_details.csv        # 每算子：Name, Start(us), Dur(us), Wait(us)
│       ├── trace_view.json           # Chrome Trace 格式 (Host + Device)
│       ├── step_trace_time.csv       # Step 级汇总
│       └── communication.json        # 通信详情（可选）
│
├── tar czf profile_l1.tar.gz
│
└── SFTP download
    │
    └── 本地 analyze_trace.py
        ├── profiling_report.md
        └── model_architecture_report.md
```

## 参考

- [ascend-profiling-anomaly](https://github.com/Ascend/agent-skills/tree/master/skills/ascend-profiling-anomaly): Bubble 检测、Anomaly 标签、Wait-Anchor 扫描、AICPU 分类
- [ascend-deploy](../../ascend-deploy/SKILL.md): 远端部署
- [profiling-collection](../../profiling-collection/SKILL.md): Profiling 数据采集
