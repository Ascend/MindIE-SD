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

## 常见异常说明

在使用 MindIE SD 进行推理业务时，模型相关文件（权重、配置、模型代码等）的安全性需由用户保证，常见的异常如下：

- 如用户在模型初始化时修改模型配置的默认参数，可能影响模型接口；若权重文件过大或配置文件参数过大，可能会触发内存不足报错，例如：`RuntimeError: NPU out of memory. Tried to allocate xxx GiB.`。
- 使用模型推理时，若输入的 tensor shape 过大，也可能触发类似的内存不足报错。
- 在使用 MindIE SD 进行生成业务时，如果出现错误输入、环境不适配等问题，代码中会抛出异常，需要用户在上层进行异常捕获处理。常见异常类型如下：

   | 异常类型 | 说明 |
   | -- | -- |
   | ZeroDivisionError | 除 0 异常。 |
   | ValueError | 参数值异常。 |
