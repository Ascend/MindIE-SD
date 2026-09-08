---
name: remote-access
compatibility: paramiko；远端 SSH（OpenSSH）；Docker（docker exec 容器内执行）；npu-smi（NPU 健康/占用查询）；SFTP/scp/rsync（文件传输）
description: >
  远程昇腾访问工具：为部署与验证提供访问远端昇腾服务器的通用执行通道，
  覆盖 SSH 连接复用、容器内命令执行、空闲卡选择与文件传输。
  当需要在远端昇腾设备执行命令、查询 NPU 状态、
  选择空闲卡或向远端传输文件时使用本技能；
  即使用户只提到「在服务器上跑个命令」而未说明 SSH，只要目标是远端昇腾环境也应触发；
  只提供远程执行通道，不负责安装/编译（env-install）与框架验证（framework-feature-enablement）。
  由 env-install 的部署流程与 dev-workflow 的部署阶段调用，亦可独立使用。
---

# 远程昇腾访问

## 定位与分工

本技能提供远程执行通道这一通用能力：SSH 连接复用、容器内命令执行、空闲卡选择与文件传输。
安装与编译由 env-install 负责，其 `scripts/deploy_to_remote.py` 完成「传输 + 远端编译安装」时会调用本技能的能力；
安装 / 编译细节（源码构建、依赖检查等）不在本技能范围。本技能由 env-install 的部署流程与 dev-workflow 的部署阶段调用，
亦可独立以 CLI 或 Python import 方式使用。

## 前置与参数收集

执行前向用户确认以下参数（无默认值，禁止猜测）：

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| 远端 IP | 昇腾服务器地址 | `<远端IP>` |
| 用户名 | SSH 登录用户 | `{用户名}` |
| 凭据 | 密码或 SSH key | 见下方凭据优先级 |
| 容器名 | Docker 容器名（容器内执行时需要） | `{容器名}` |
| 工作目录 | 远端命令 / 传输的基准路径 | `{远端工作目录}` |
| 卡数 | 参与空闲卡扫描的设备数 | `8`（与容器映射一致） |

- 凭据优先级：环境变量 `MINDIE_SSH_PASSWORD` > 交互式输入（不回显）> `--password`。
  避免明文密码出现在进程列表 / shell history；报告与日志中不回显密码。
- 主机密钥：连接默认不设 AutoAddPolicy，未知主机密钥被拒绝（防中间人）。
  首次连接报 host key 错误时，先核对主机指纹并将主机加入 `~/.ssh/known_hosts`。

### 连接复用原则

- 所有远端操作（执行、选卡、传输）复用同一个 SSH 连接：单个 `paramiko.SSHClient` 从建连到全部命令结束全程复用，最后才 `ssh.close()`。
- 严禁每个步骤独立 `ssh.connect()`——远端 SSH 有 `MaxStartups` 限制，短时间多次连接会触发拒绝访问。
- 一次 exec 内用 `;` / `&&` 串联多条命令，减少 bash login shell 初始化次数。

## 连接与执行

### ssh_helper.py：单连接命令执行器

`scripts/ssh_helper.py` 复用同一 SSH 连接逐条执行命令，CLI 参数为 `--host/--user/--password/--container/--cmd`：

```shell
python ssh_helper.py --host <远端IP> --user {用户名} --container {容器名} \
    --cmd "npu-smi info -l"
```

- 不带 `--container` 时命令直接在远端 shell 执行；带 `--container` 时自动包装为 `docker exec {容器名} bash -lc '{cmd}'`。
- 包装时用 `shlex.quote` 分别转义容器名与命令：远端 shell 与内层 bash 各只解一次引号，命令原样到达（含双引号、`$`、反引号），无需手写多层转义。
- 输出格式：stdout 直接打印，stderr 以 `[stderr]` 前缀打印，退出码以 `[exit code: N]` 结尾，脚本以退出码退出。
- stdout / stderr 由双线程并发读取，避免 paramiko 通道 64KB 管道缓冲填满时串行读取死锁；`--timeout` 默认 600s。

Python 复用（同一连接连续执行多条命令）：

```python
from ssh_helper import make_ssh, run

ssh = make_ssh("<远端IP>", "{用户名}", password="{密码}")
try:
    code, out, err = run(ssh, "npu-smi info -l")
    code, out, err = run(ssh, "ls {远端工作目录}")
finally:
    ssh.close()
```

### 多命令与容器

- 多条命令以 `;` 串联为一次 exec（如 `cmd1; cmd2; cmd3`）；有先后依赖时改用 `&&`。
- 需要 CANN 环境时在命令开头加 `source /usr/local/Ascend/ascend-toolkit/set_env.sh && ...`。
- 多容器场景：同一连接内为每个容器各执行一次 `docker exec`，不新建 SSH 连接。

### 容器内后台化（脱离 SSH）

- SSH 会话断开会终止未脱离会话的前台与宿主进程（现象：断连后任务消失），长任务必须后台化。
- `docker exec -d {容器名} bash -lc '{cmd}'`：detach 模式，exec 立即返回，进程在容器内继续，与 SSH 会话解耦。
- 命令内 `nohup {cmd} > {日志文件} 2>&1 &`：容器内进程脱离会话，不随 SSH 断开被杀。
- 输出重定向到文件便于事后排查（读日志文件或 `docker logs {容器名}`）；轮询 / 监控类进程自身也要后台化。

### Windows 脚本换行（CRLF → LF）

- 症状：Windows 编辑的 `.sh` / `.py` 上传后，在 Linux / 容器内报 `$'\r': command not found` 或编译失败。
- 核心原则：不在本地修改源码换行符（会产生大量无意义 git diff）；CRLF→LF 由 env-install 的 `deploy_to_remote.py`
  在传输阶段对远端临时工作副本自动处理，仅作用于 `.sh` / `.py` 文本。
- 手工上传临时脚本时在远端转换：`sed -i 's/\r$//' {file}`（或 `dos2unix {file}`）。
- 换行转换只针对文本：二进制文件（权重等）绝不经过 CRLF→LF，否则文件损坏（可能表现为 tokenizer / 权重加载异常）。

### 嵌套 shell 引号

- 避免手写 `docker exec ... bash -lc 'python -c "..."'` 等多层嵌套引号：每层 shell 都会解一次引号，可能吞掉 `%`、`$`、双引号。
- 复杂逻辑优先把脚本 SFTP 上传为 `.py` / `.sh` 后在远端执行，而不是堆叠内联引号。

### 长任务三段式（run / 轮询 / post）与结果回读

- 补测/扫描批推荐三段式：`run.sh`（每档一个配置，日志打结束标记如 `echo '=== XXX_DONE ==='`）
  与 `post.sh`（抽帧/质量/汇总，CPU 密集不占 NPU，可与后续 run 并行）分离；驱动脚本只做
  上传（SFTP，CRLF→LF）→ `docker exec -d ... nohup bash run.sh > log 2>&1 &` → 轮询标记 → 跑 post。
- **墙钟位置**：`curl -w time_total=...` 输出走 run 脚本 stdout（驱动 nohup 日志），**不在 serve.log**
  ——取墙钟 grep 驱动日志；serve.log 只有 vllm 服务输出（含 cache/量化等计数契约线索）。
- 轮询打印远端日志前，python 先 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`：
  远端日志常含 emoji/ANSI，Windows gbk 控制台会 `UnicodeEncodeError` 直接中断轮询进程
  （远端任务不受影响，但轮询/后处理会停）。
- 不要在 python f-string 里内嵌远端 shell 变量（如 `${t}` → `NameError`）；用字符串拼接，
  或把 `{{ }}` 转义成字面 `{}`。
- 结束释放卡：`pkill -9 -f "vllm-omni serve"` 后用 `npu-smi info` 进程段复核（0 进程）再交还。

## 空闲卡选择

### pick_free_device.py：按 HBM 占用选空闲卡

`scripts/pick_free_device.py` 扫描容器内 `npu-smi info -t usages -i 0-{num_cards-1}`，返回 HBM 占用率最低的卡。

CLI（独立建连）：

```shell
python pick_free_device.py --host <远端IP> --user {用户名} --password {密码} \
    --container {容器名} --num-cards 8
# 输出：{device_id} {hbm_usage_pct}
```

Python import（复用已有连接，避免重复握手）：

```python
from pick_free_device import pick_free_device

# ssh 为已连接的 paramiko.SSHClient（可用 ssh_helper.make_ssh 建立）
dev, usage = pick_free_device(ssh, container="{容器名}", num_cards=8)
```

要点：

- 选卡前先健康预检：`npu-smi info -l`（卡状态）与 `npu-smi info -t health`；Alarm / Warning / ERROR 状态卡不是空闲卡，直接避开。
- 解析按行结构进行：首列非设备号、末列无法解析为数值的异常行会被跳过，不进入候选。
- 占用对比须在同一卡组内：命令在指定容器内执行，`npu-smi` 只能看到该容器映射的设备，
  `--num-cards` 需与容器映射的设备数一致；不同容器、不同平台（如 910B vs A3）的占用数值不可直接对比。
- 多人共享环境可结合 `npu-smi info proc` 确认进程占用后再定卡。
- 安全提示：import 复用连接的路径沿用 ssh_helper 的主机密钥策略（未知主机拒绝）；独立 CLI 内部采用 AutoAddPolicy，仅适合可信环境。

## 文件传输

- 部署级传输（增量 + CRLF→LF + 远端编译安装）由 env-install 完成：其 `scripts/deploy_to_remote.py`
  复用本技能建立的 SSH / SFTP 通道做传输并触发容器内编译安装，本技能不重复实现安装流程。
- 增量原则：`sftp.stat` 逐个文件比对远端是主要瓶颈——先 `ls -l` 拉取远端文件清单，本地 diff 后仅传输变更文件。
- 复用连接：SFTP 会话挂在已建 SSH 连接上（`ssh.open_sftp()`），不为每个文件新建连接。
- 手工传输：`scp` 或 `rsync -avz -e ssh` 按需增量；首次传输先确认远端目录布局，目标路径须与远端实际构建 / 执行目录一致。
- 文本换行与二进制：`.sh` / `.py` 等文本走 CRLF→LF（见「Windows 脚本换行」）；权重等二进制文件不做任何换行转换。

### 上传同步纪律（同名冲突 / 内联 vs 脚本，2026-09 复盘实证）

- **同 basename 文件必须放 distinct dest 目录**：多任务 / 多版本迭代中频繁出现同名脚本
  （如 `probe.py`、`run.sh`）上传到同一远端目录 → 后传覆盖先传，远端执行的可能是**旧文件**，
  造成"改了代码但不生效 / 结果与本地不符"类错乱。上传前先 `mkdir -p` 按任务建目录
  （如 `{任务名}/{日期}/`），每次上传用绝对路径落到各自目录；同名文件在新目录重传时
  先确认远端确实更新（比对 mtime / hash）。
- **复杂逻辑一律先写脚本再上传执行，不堆 inline**：循环 / 变量 / heredoc / 多层引号的
  命令先写成 `.py` / `.sh` 上传（SFTP，CRLF→LF）再在远端执行；内联 python / heredoc 会被
  外层 shell（PowerShell → ssh → docker exec bash）逐层解引号吞掉 `$`、`|`、双引号，
  排错成本高于写脚本本身。
- 凭据 / 机密参数走环境变量（`MINDIE_SSH_PASSWORD`），不写进上传的脚本文件与命令行
  （脚本可能被他人阅读 / 留在远端）。

## 故障排查

SSH / 连接 / 容器 / 换行相关条目速查（完整决策树见 Reference Files）：

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| SSH 认证失败 | IP / 用户名 / 密码错误或网络不通 | 核对参数并检查网络连通性 |
| 报 "Server not found in known_hosts"（密钥已登记仍报） | **paramiko ≥ 4.0 不再自动加载用户 known_hosts**，`make_ssh` 需显式 `load_system_host_keys()` | ssh_helper.py 已修复；自写脚本建连时同样先 `ssh.load_system_host_keys()`，保持 RejectPolicy 语义 |
| 复杂命令内层引号（`python -c "…"`、`$`、反引号）经多层 shell 被吞/截断 | 每层 shell 解一次引号；Windows PowerShell 外层双引号遇内层 `"` 提前终止命令串 | 命令先 base64 编码，远端 `echo {b64} \| base64 -d \| bash` 执行（本会话验证可靠）；或用 ssh_helper 的 shlex.quote |
| 短时间多次连接被拒 | 远端 `MaxStartups` 限制 | 遵循连接复用原则，单连接跑完所有命令 |
| 报错 command not found（脚本含 CR 字符） | Windows 编辑的脚本未转 LF | 远端用 sed 去除行尾 CR，或由 deploy_to_remote.py 自动转换 |
| docker exec 内命令报语法错误或输出丢失 | 多层 shell 逐层解引号破坏命令 | 用 ssh_helper 的 shlex.quote 包装，或 SFTP 上传脚本再执行 |
| SSH 断开后任务消失 | 任务未脱离会话 | 容器内 `nohup ... &` 或 `docker exec -d` 后台化 |
| tokenizer / 权重加载异常 | 二进制文件被换行转换损坏 | 换行转换仅限文本，二进制绝不转换 |
| 凭据明文泄露 | `--password` 明文传递并留在进程列表 / 日志 | 改用 `MINDIE_SSH_PASSWORD` 环境变量或 SSH key；日志不回显 |
| 改了代码但不生效 / 远端结果与本地不符 | 同 basename 上传互相覆盖，远端执行的是旧文件 | 按任务建 distinct dest 目录（见「上传同步纪律」），上传后比对 mtime / hash |

> 其他 SSH / 容器 / 环境类问题（含 Windows 开发机 schannel、传输路径语义等）见 env-install 的故障排查决策树（归属 env-install，本技能只引用不复制全文）。

## Reference Files

- `../env-install/references/troubleshooting-env.md` — 加载时机: SSH 认证 / 连接 / 容器 / 换行等问题需系统排查定位根因时（该文件归属 env-install，仅引用不复制）

## Bundled Scripts

- `scripts/ssh_helper.py` — 单连接 SSH 命令执行器（连接复用；`--host/--user/--password/--container/--cmd`）
- `scripts/pick_free_device.py` — 按 HBM 占用率选空闲 NPU 卡（`--host/--user/--password/--container/--num-cards`；import 复用连接）

## 维护与更新

当远端 SSH 行为（MaxStartups、密钥策略）、docker exec 语义、npu-smi 输出格式变化，
或发现新的连接 / 换行 / 后台化 / 传输问题时，按 dev-workflow 的复盘流程更新本技能。
