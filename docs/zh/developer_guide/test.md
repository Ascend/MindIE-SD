# 测试

MindIE SD 的测试套件支持在有或无昇腾 NPU 硬件的环境下运行。无 NPU 硬件时也可执行 CPU 兼容的测试。

## 测试环境说明

测试按是否依赖设备严格分为两类，分别放在两个目录下：

| 类型 | 说明 | 是否需要 NPU | 目录 |
|------|------|:---:|------|
| CPU 兼容测试（UT） | 配置解析、工具函数、量化参数校验、编译逻辑等 | 否 | `tests/UT/` |
| NPU 依赖测试（算子用例） | 自定义算子精度、Flash Attention、模型张量运算等 | 是 | `tests/ops/` |

`tests/conftest.py` 会校验这条边界：`tests/UT/` 下的用例不得访问 NPU，`tests/ops/` 下的用例缺少 NPU 时必须跳过。因此只有真正依赖设备的用例才放在 `tests/ops/`，其余一律进 `tests/UT/`，由 CI 在提交 PR 时执行。

## 统一测试入口

所有测试都从 `tests/scripts/run_suites.py` 这一个入口进入，区别只是传入的命令行参数。该脚本按模式（`CPU` / `NPU`）分别拉起独立的子进程执行 pytest，再合并覆盖率结果。

```bash
python -m pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
python -m pip install -r requirements-test.txt
```

### 常用命令

| 场景 | 命令 |
|------|------|
| 全量测试（CPU + NPU） | `python tests/scripts/run_suites.py` |
| 仅 CPU 兼容测试（无 NPU 环境） | `python tests/scripts/run_suites.py --cpu_only` |
| 仅 NPU 依赖测试（需 NPU 环境） | `python tests/scripts/run_suites.py --npu_only` |
| 指定测试文件或目录 | `python tests/scripts/run_suites.py tests/UT/layers` |
| 只跑本次改动影响的用例 | `python tests/scripts/run_suites.py --incremental` |
| 先看增量选出了哪些用例 | `python tests/scripts/run_suites.py --incremental --dry-run` |

### 参数说明

| 参数 | 说明 |
|------|------|
| 无模式参数 | 依次执行 CPU 与 NPU 两套用例 |
| `--cpu_only` | 只执行 `tests/UT/`（CPU 模式） |
| `--npu_only` | 只执行 `tests/ops/`（NPU 模式），执行前会校验 `torch.npu.is_available()` |
| `--all` | 显式指定全量，与不传模式参数等价 |
| `test_files ...` | 位置参数，指定仓库内 `tests/` 下的文件或目录 |
| `--incremental` | 基于 git 改动与依赖关系自动选出受影响的用例 |
| `--base-branch` | 增量分析的基线分支，默认 `dev` |
| `--no-staged` / `--no-unstaged` | 增量分析时忽略已暂存 / 未暂存改动 |
| `--dry-run` | 只输出增量选中的用例，不执行（需与 `--incremental` 同用） |

### 测试产物与覆盖率门禁

产物目录由环境变量 `MINDIE_TEST_RESULTS` 指定，未设置时位于系统临时目录下的 `mindiesd-test-results/`，每次运行生成一个 `run-*` 子目录，包含：

- `cpu.xml` / `npu.xml`：JUnit 报告
- `combined.coverage`、`coverage.json`：合并后的覆盖率数据
- `file-coverage.json`：逐文件覆盖率结果

当以全量 CPU 方式运行（未做显式用例筛选）时，`tests/scripts/check_file_coverage.py` 会校验 `mindiesd/` 下每个 Python 文件的分支覆盖率不低于 80%，不达标则本次运行返回非零退出码。

### CI 专用入口

仓库里只剩一个 shell 包装 `tests/run_UT_test.sh`，它只服务于 CI 的 CPU 流水线：选好 Python 解释器后以 `--cpu_only tests/UT` 调用 `run_suites.py`。流水线会删掉该脚本的最后一行再用 `grep OK` 判定结果，因此不要用它跑本地测试，日常请直接调用 `run_suites.py`。

```bash
bash tests/run_UT_test.sh   # 仅 CI 使用，等价于 run_suites.py --cpu_only tests/UT
```

## LA 单算子精度测试

本章节介绍 MindIE SD 仓中 LA 算子的精度自测方式。

1. 如需切换已安装版本，可先卸载当前 MindIE SD：

   ```bash
   python -m pip uninstall mindiesd
   ```

2. 修改 `tests/ops/plugin/la_acc_prof.py` 文件，选择 Option 1 或 Option 2，通过加载 `test_la.csv` 或 `enumerated_cases.csv` 文件，测试 LA 算子在所设置 shape 下的精度。

   - `tests/ops/plugin/test_la.csv`：设置了常用 SD 模型的输入 shape。
   - `enumerated_cases.csv`：枚举的各种 shape。

3. 完成修改后在仓库根目录执行以下命令：

   ```bash
   python tests/ops/plugin/la_acc_prof.py
   ```

运行成功后会在仓库目录下生成结果文件，记录 LA 和 FAScore 的相似度，可据此查看算子在目标 shape 下的精度表现。

## 测试编写规范

### 测试组织原则

测试按是否依赖设备分目录存放：

| 源码模块 | 对应 CPU 测试目录 | 对应 NPU 测试目录 |
|----------|-------------|-------------|
| `mindiesd/layers/` | `tests/UT/layers/` | `tests/ops/layers/` |
| `mindiesd/cache/` | `tests/UT/cache/` | — |
| `mindiesd/compilation/` | `tests/UT/compilation/` | — |
| `mindiesd/quantization/` | `tests/UT/quantization/` | `tests/ops/quantization/` |

新增代码时，检查对应测试目录是否已有测试文件，并在其中追加或新建测试覆盖新功能。判断标准只有一条：用例实际调用了 NPU 算子或设备接口时才放在 `tests/ops/`，否则放在 `tests/UT/`。算子用例还需要与设备型号对齐——只在特定设备上成立的用例按该设备标记，不得让其他型号设备借用同一条用例通过。

### 测试性能规范

- 单个测试文件运行超过 500 秒时，应拆分为多个文件
- 优先复用已有的设备/模型加载，减少重复初始化开销
- CPU 友好的测试提交前可用 `python tests/scripts/run_suites.py --cpu_only` 验证
