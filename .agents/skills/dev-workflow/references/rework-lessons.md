# 返工教训

> **目录** · 1. 拒绝未实现功能的前置配置 · 2. 最小必要改动原则 · 3. 独立任务必须实际并行执行 · 4. 非代码仓内容不入库 · 5. PLAN.md 未随任务变更同步更新 · 6. triton vs triton-ascend 包名混淆 · 7. pip install -e . 新增文件未被索引 · 8. SSH 连接重复创建 · 9. dummy_run 门禁违规综合教训 · 10. Markdown 代码块未指定语言触发 MD040 · 11. 嵌套 Shell 引号转义失败 · 12. Profiling 结果回传与 GBK 编码 · 13. 通用分析脚本纳入 Skills · 14. meta→to_empty 构造后未注册 buffer · 15. CRLF→LF 转换破坏二进制文件 · 16. 远端 model 模块名冲突 · 17. Gated Model 配置下载 · 18. expandable_segments 池锁定误判 OOM · 19. 多模型 CLI 参数不一致 · 20. 首次部署未检查远端文件完整性 · 21. 过度抽象 · 22. 未请求的额外功能 · 23. Pattern 单元测试通过但全模型不命中 · 24. register_replacement 无法处理 get_attr · 25. Inductor freeze 不识别自定义 NPU ops · 26. 编译开销定位方法

以下问题均在 MindIE-SD 开发中实际发生并导致返工。

## 1. 拒绝未实现功能的前置配置

**问题**：功能未实现，但配置中提前添加了死配置字段。

**规则**：

- 配置字段必须与实现同步添加，禁止提前预留
- 每个配置字段必须有对应的已实现功能
- Review 时检查是否有未使用的新增配置

## 2. 最小必要改动原则

**问题**：为简单功能添加了大量不必要的框架改动。

**规则**：

- 实现功能前先确认：现有基础设施是否已经满足需求
- 每次改动前自问："不改这个，功能能运行吗？"
- 对于使用静态类接口的 pattern，不需要修改注册框架

## 3. 独立任务必须实际并行执行

**问题**：独立任务提前规划了并行，但实际执行时串行化了。多 NPU 卡的并行能力未被利用。

**规则**：

- 无代码依赖的独立任务直接启动并行闭环，不在同一线程串行排队
- 每个闭环独立：写测试 → 实现 → 部署 → 各自 pytest（不同卡）
- 共享文件的修改最后统一合并

## 4. 非代码仓内容不入库

**问题**：部署脚本、临时检查脚本误入代码仓目录。

**规则**：

- 部署脚本、临时检查脚本、一次性验证脚本不纳入代码仓
- 使用后立即删除或放在代码仓外独立目录
- 所有临时脚本统一放入 `tmp/` 目录，`.gitignore` 中追加 `tmp/` 屏蔽
- 合并临时脚本：避免散落多个独立脚本文件，合并为统一入口脚本

## 5. PLAN.md 未随任务变更同步更新

**问题**：任务被跳过或还原后，PLAN.md 中仍保留已废弃的条目。

**规则**：

- 每完成或跳过一个任务，立即更新 PLAN.md
- 删除 PLAN.md 中已废弃的任务条目和文件清单
- 任务粒度变化（如合并/拆分）同步刷新
- PLAN.md 内容必须与代码仓实际状态一致

## 6. triton vs triton-ascend 包名混淆

详见 [ascend-ops.md](ascend-ops.md)。

## 7. pip install -e . 新增文件未被索引

**问题**：首次部署后新增的 `.py` 文件存在于远端磁盘，但 import 报 `ModuleNotFoundError`。

**根因**：`pip install -e .` 在首次安装时扫描包目录并建立索引，后续新增的文件不会自动加入。

**规则**：

- 新增 Python 文件后，必须重新执行 `pip install -e .` 让 editable install 重新扫描
- 部署脚本的 `build_cmd` 中 `pip install -e .` 应在文件传输之后执行

## 8. SSH 连接重复创建

**问题**：多个独立脚本各自 `ssh.connect()` 新建 TCP/TLS 连接，加上 `sftp.stat` 逐个文件远端比对，以及独立的 `docker exec` 启动 bash login shell，累计产生大量无效等待。

**规则**：

- 所有远端操作（传输 + 编译 + 测试）合并为一个脚本，全程复用同一个 `ssh` 对象和 `sftp` 会话
- `docker exec` 命令用 `;` 串联，减少 login shell 初始化次数
- 仅传输本次变更文件，不做全量 `sftp.stat` 比对

```python
# 正例：长连接复用
ssh = paramiko.SSHClient()
ssh.connect(HOST, ...)
sftp = ssh.open_sftp()
for f in CHANGED_FILES:
    sftp.putfo(...)
sftp.close()
_run(ssh, "docker exec ... pip install -e .")
_run(ssh, "docker exec ... pytest tests/... -v")
ssh.close()

# 反例：每个操作独立 connect → 3 次连接，共浪费 6-15s
```

## 11. `examples/dummy_run` 门禁 11 项违规综合教训

**问题**：`examples/dummy_run/` 首次提交通过了 markdownlint 检查（门禁仅报 MD040），但后续完整门禁扫描报出 11 项违规，涉及代码风格、异常处理、参数设计等多方面。

**规则**（详见 `code-standards` skill）：

| 违规类型 | 规则 | 修复方式 |
|----------|------|----------|
| `avoid-import-method` | 禁止 `__import__()` | 使用模块级 `import` 或 `importlib.import_module()` |
| `avoid-using-exit` | 禁止在函数内 `sys.exit()` | 改为 `raise RuntimeError(...)` |
| `full-path-executable` | 禁止裸命令名 | 使用 `shutil.which()` 解析全路径 |
| `bare-except-pass` | 禁止无日志的 `except: pass` | 至少 `logger.debug(...)` |
| `too-many-arguments` | 参数 ≤ 5 | 移除调用方未使用的参数 / 合并 / 提取配置对象 |
| `comment-out-code` | 禁止注释掉的代码行 | 直接删除，或改为描述性注释 |
| `function-order` | 类方法排序 | 私有方法集中放在公共方法之后 |
| `duplicate-string` | 禁止重复字符串字面量 | 提取为类级/模块级常量 |

**关键认知**：

- `examples/` 目录与 `mindiesd/` 源目录受同一套门禁规则约束，不可放松标准
- 完整门禁扫描可能分阶段执行（先 markdownlint，后代码检查），首次通过不代表完全通过
- 提交前应全面运行门禁检查，不应依赖阶段性通过结果

## 10. Markdown 代码块未指定语言触发 MD040 门禁失败

**问题**：`examples/dummy_run/README.md` 中 3 处围栏代码块未指定语言标记（` ``` ` 裸写），CI markdownlint MD040 检查未通过。

**规则**：

- 所有围栏代码块必须指定语言或内容类型（`text`/`bash`/`python`/`shell`/`yaml`/`json`/`markdown` 等）
- 目录树、终端输出、日志等非可执行内容使用 `text`
- 提交前自检：`pre-commit run markdownlint --files <changed_file>.md`
- 详细规范见 `markdown-lint` skill

````markdown
<!-- 正例 -->
 ```shell
 npu-smi info -l
 ```

 ```text
 examples/
 ├── a.py
 └── b.py
 ```

<!-- 反例 -->
 ```
 examples/
 ├── a.py
 └── b.py
 ```
````

## 9. 嵌套 Shell 引号转义失败

**问题**：通过 paramiko `exec_command` 执行多层嵌套命令（Windows PowerShell → SSH → docker exec → bash -lc → python -c）时，内层 Python 代码中的 `%`、`$`、双引号被外层 shell 逐层转义，导致语法错误或输出静默丢弃。

具体来说，`$` 被 PowerShell 和 bash 各展开一次，`%` 被 bash printf-style 解释，" 的嵌套层次难以追踪。

**规则**：

- 避免 `docker exec ... python -c "..."` 嵌套引号。改用 SFTP 上传 `.py` 脚本文件后远端执行：

  ```python
  sftp.putfo(io.BytesIO(script.encode()), "/path/to/remote.py")
  _run(ssh, "docker exec container python3 /path/to/remote.py")
  ```

- 如需传递少量参数，使用 `sys.argv` 或环境变量，不在 shell 命令行中拼接 Python 代码。
- 示例反例（4 层嵌套转义失败）：

  ```python
  cmd = 'docker exec %s bash -lc "python3 -c \'import torch_npu; ...\'"'
  ```

  正例（SFTP 上传）：

  ```python
  script = "import torch_npu\nfor i in range(8):\n    print(...)"
  sftp.putfo(BytesIO(script.encode()), "/tmp/check.py")
  _run(ssh, "docker exec container python3 /tmp/check.py")
  ```

## 12. Profiling 结果回传与 GBK 编码

**问题**：远端 CANN Profiler 日志含 non-ASCII 字符，Windows GBK 终端 `print()` 输出报 `UnicodeEncodeError`。

**规则**：

- paramiko `exec_command` 返回的 stdout/stderr 统一以 UTF-8 解码（`errors="replace"`）
- 打印前用 `str.encode("utf-8", errors="replace").decode("utf-8", errors="replace")` 二次清洗
- 远端 profiling 日志不逐行打印到本地终端，改为保存到文件后 cat 前 N 行

## 13. 通用分析脚本纳入 Skills

**问题**：最初计划将 `deploy_and_profile.py` 和 `analyze_trace.py` 作为临时脚本。
但两个脚本的通用性强（参数化 IP/容器/密码，支持任意 ASCEND_PROFILER_OUTPUT 格式），
应作为可复用能力沉淀。

**规则**：

- 通用分析/部署脚本归入 skills 目录（`scripts/` 子目录），不作为一次性临时脚本
- 代码仓内容仅限 `examples/dummy_run/` 示例本身，不包含 profiling 产出的数据和报告
- 脚本参数化程度应支持不同环境复用

## 23. Pattern 单元测试通过但全模型不命中

**问题**：单元测试 model 使用 functional API（`weight` 作为函数输入，FX graph 中为 `placeholder` 节点），
全模型使用 `nn.Module`（`self.weight` 为 `get_attr` 节点），测试通过但全模型 graph 中 pattern 匹配静默失败。

**规则**：

- 单元测试 model 的 graph 结构必须与全模型完全一致，包括参数来源方式（functional vs modular）
- 单元测试通过是 pattern 验证的必要条件，但不是充分条件
- 如果 pattern 涉及 `nn.Module` 的参数（weight/bias），必须用全模型 profiling + kernel diff 做最终验证
- 全模型验证方法：采集 eager + compile profiling → `kernel_details.csv` diff → 确认融合 kernel 出现

## 24. `register_replacement` 无法处理 `get_attr` 参数

**问题**：`torch._inductor.pattern_matcher.register_replacement` 要求 pattern 所有参数在 traced graph
中为 `placeholder` 节点。全模型中 `nn.Module` 的 weight/bias 是 `get_attr` 节点，
两者 node 类型不同 → pattern matcher 静默跳过（无错误日志，match count 不变）。

**规则**：

- 创建 pattern 前先判断目标算子是否使用了 `nn.Module` 的参数
- 若 weight 来自模块参数（`get_attr`），`register_replacement` 的 placeholder-based pattern 无法匹配
- Fallback 方案：在 `PatternMatchPass` 中实现自定义 graph traversal pass，直接遍历 FX graph 节点
  并手动重写子图（见 `compilation-dev` skill §5 自定义 Graph Pass）

**判据**（kernel diff 中确认）：

- 全模型中 RMSNorm 的 weight 是 `get_attr(self.norm_q.weight)` → `register_replacement` 匹配失败
- 全模型中 GELU 无 learnable parameters → `register_replacement` 匹配成功
- 全模型中 RoPE 的 `apply_rotary_emb` 使用 `slice_scatter` 两阶段复制 → 与现有 `chunk/stack/flatten` 模式不匹配

## 25. Inductor freeze 阶段不识别自定义 NPU ops

**问题**：自定义 graph pass 插入的 `npu_rms_norm` 等非 aten op 节点，
在 `torch._inductor.freezing.freeze()` 的 `node_copy` 过程中 Crash（`KeyError: npu_rms_norm`）。

**根因**：`freeze()` 内部的图拷贝期望所有目标函数都在 Inductor 的 env dict 中注册。
NPU 自定义 op（`torch.ops.npu.*`）不在该 dict 中 → `node_copy` 失败。

**规则**：

- 涉及自定义 NPU op 的 graph pass 必须在 `graph_rewrite_after_freezing` 中执行，
  而非 `graph_rewrite_before_freezing`
- 即在 `MindieSDBackend.compile()` 中，调用 `patterns._rewrite_*_to_fused(fx_graph)` 的位置
  应在 `apply_decompose_auto_functionalized_pass` 之后

**代码示例**：

```python
def graph_rewrite_after_freezing(fx_graph, inputs):
    self.__class__.apply_redundant_node_elimination_pass(fx_graph, inputs)
    patterns._rewrite_rmsnorm_to_fused(fx_graph)  # ← 在 freeze 之后
    self.__class__.apply_decompose_auto_functionalized_pass(fx_graph)
    return fx_graph
```

## 26. 编译开销定位方法

**问题**：compile 推理比 eager 慢 9%（Wan2.2: 7000ms vs 7624ms），无法从推理时间差异定位原因。

**方法**（kernel 级 diff 分析法）：

1. 分别在 eager 和 compile 模式下执行 profiling：

   ```bash
   python wan_infer.py --profile                    # eager
   python wan_infer.py --compile --profile          # compile
   ```

2. 从 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中按 kernel 名称聚合耗时

3. 对同名 kernel 计算 `compile_time - eager_time`，按差值绝对值排序

4. 定位开销来源：
   - `ViewCopy` 569ms → 1137ms (+568ms, +100%) ← 最大开销源
   - `TensorMove` 0 → 40ms（新增）
   - `StridedSliceCopy` 0 → 25ms（新增）
   - RMSNorm 融合节省约 9ms（Pow+Mean → RmsNorm）

5. 确认 Custom Pattern 生效：搜索 compile 独有的 `RmsNorm` kernel（16ms）
