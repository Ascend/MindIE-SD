# 测试

## CPU 友好单元测试

推荐优先使用仓库当前提供的 CPU 友好 UT 入口，生成覆盖率与测试产物。

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
bash tests/run_UT_test.sh
```

默认生成的产物位于 `tests/UT/` 目录，包括：

- `run_UT.log`
- `final.xml`
- `coverage.xml`
- `htmlcov/`

仓库中的 `tests/scripts/check_coverage.py` 用于在 CI 中校验新增 Python 文件的覆盖率门禁。

## 全量测试入口

当环境具备 Ascend/NPU 运行栈时，可使用现有包装脚本执行全量或按模式测试：

```bash
bash tests/run_test.sh --all
```

可选参数：

- `--cpu_only`
- `--npu_only`
- `--all`

## LA 单算子精度测试

本章节介绍 MindIE SD 仓中 LA 算子的精度自测方式。

1. 如需切换已安装版本，可先卸载当前 MindIE SD：

   ```bash
   python -m pip uninstall mindiesd
   ```

2. 修改 `tests/plugin/la_acc_prof.py` 文件，选择 Option 1 或 Option 2，通过加载 `test_la.csv` 或 `enumerated_cases.csv` 文件，测试 LA 算子在所设置 shape 下的精度。

   - `./tests/plugin/test_la.csv`：设置了常用 SD 模型的输入 shape。
   - `enumerated_cases.csv`：枚举的各种 shape。

3. 完成修改后执行以下命令：

   ```bash
   cd tests
   python plugin/la_acc_prof.py
   ```

运行成功后会在仓库目录下生成结果文件，记录 LA 和 FAScore 的相似度，可据此查看算子在目标 shape 下的精度表现。

## 测试编写规范

### 测试组织原则

`tests/` 下的测试目录与 `mindiesd/` 模块层次对应：

| 源码模块 | 对应测试目录 |
|----------|-------------|
| `mindiesd/layers/` | `tests/layers/` |
| `mindiesd/cache/` | `tests/cache/` |
| `mindiesd/compilation/` | `tests/compilation/` |
| `mindiesd/quantization/` | `tests/quantization/` |

新增代码时，检查对应测试目录是否已有测试文件，并在其中追加或新建测试覆盖新功能。

### 测试性能规范

- 单个测试文件运行超过 500 秒时，应拆分为多个文件
- 优先复用已有的设备/模型加载，减少重复初始化开销
- CPU 友好的测试应优先使用 `bash tests/run_UT_test.sh` 验证
