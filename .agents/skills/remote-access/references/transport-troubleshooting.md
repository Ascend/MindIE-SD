# 传输域 / 本地开发机故障（SSH 侧补充单点）

> **加载时机**：SSH 连接类问题已按 `SKILL.md`「故障排查」表处置仍不收敛时；
> **Windows 开发机**上 git / curl 拉取失败时；或需要判断"这次失败是本地传输侧还是远端环境侧"时。
> **不重复的内容**：认证失败、CRLF、`MaxStartups` 连接复用、嵌套引号、`nohup` / `docker exec -d`
> 后台化、同 basename 覆盖、凭据优先级、二进制不做换行转换——单点在 `SKILL.md` 正文与
> 「故障排查」表，本文件只补**该表未覆盖**的条目。

## 1. Windows 开发机：git / curl HTTPS 握手失败（schannel）

- **症状**：`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS (0x8009030e)`；
  `git fetch/pull/clone`、`curl` 均受影响。
- **判据**：受限 / 沙箱进程下 **schannel（Windows 原生 TLS 后端）拿不到凭据**；
  同一 URL 用 **python `urllib`（openssl）能通** ⇒ 判定为 schannel 问题，不是网络或凭据本身错。
- **处置**：单次 `git -c http.sslBackend=openssl fetch`；长期 `git config --global http.sslBackend openssl`
  （只改 TLS 后端，不改远端地址与凭据）。
- **与部署的关系**：这是**本地开发机**故障，远端容器不受影响；若因此改用 `scp` / SFTP 通道传源码，
  仍遵守传输纪律（文本 CRLF→LF、二进制不转换、连接复用——见 `SKILL.md`）。

## 2. 传输范围与路径语义（跨技能指针）

部署级传输（增量 + CRLF→LF + 远端编译安装）由 `env-install/scripts/deploy_to_remote.py` 承担，
其**路径语义**（传到 `{workspace}/{local_root.name}` 而构建固定 `cd {workspace}/MindIE-SD`）、
**排除列表**（`refs/` 等大目录不在排除列表，全量传输会带上 profiling 数据）与
"远端布局不一致时改精准 SFTP 同步"的处置，单点在
`env-install/SKILL.md`「部署脚本」节——本技能不复制，只负责通道（SFTP 复用同一 SSH 连接）。
委派前先确认远端实际构建 / 执行目录与本地假定的目录一致，否则改精准同步而不是硬跑脚本。

## 3. 网络不可达时的定位顺序（先本地、后远端）

1. 本地开发机：目标 host/port 是否可达（发在**本地**的失败几乎都是 schannel / 代理 / DNS）。
2. 远端宿主：`ss -lunp` 看目标端口是否有监听；容器内 `curl -s localhost:{port}/health`。
3. 远端外网（拉 wheel / 镜像 / 权重）：代理与 registry mirror 是否配置；容器 DNS 抖动表现为
   可重试的 `NameResolutionError` / `ReadTimeoutError`（多数自动重试成功，**WARNING 不等于失败**）。

## 维护与更新

当本地 TLS 后端行为、远端 `MaxStartups` / 代理策略、或传输链路（SFTP 复用方式）变化，
或发现新的连接 / 传输类故障模式时，按 dev-workflow 的复盘流程更新本文件，
并把**可复用结论**回写 `SKILL.md`「故障排查」表（表是速查真源，本文件是补充单点）。
