# 测试

MindIE SD 的测试套件支持在有或无昇腾 NPU 硬件的环境下运行。无 NPU 硬件时也可执行 CPU 兼容的测试。

## 测试环境说明

测试分为两类：

| 类型 | 说明 | 是否需要 NPU |
|------|------|:---:|
| CPU 兼容测试 | 配置解析、工具函数、量化参数校验、编译逻辑等 | 否 |
| NPU 依赖测试 | 自定义算子精度、Flash Attention、模型张量运算等 | 是 |

## 测试入口

### 入口一：CPU 友好单元测试（推荐无 NPU 用户使用）

`run_UT_test.sh` 始终以 CPU 模式运行，适合无 NPU 硬件的开发环境。

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

### 入口二：全量测试（支持三种模式）

`run_test.sh` 通过参数控制测试范围，支持以下三种模式：

**1. 全量测试（默认）**

执行所有 CPU 兼容测试和 NPU 依赖测试：

```bash
cd tests/
bash run_test.sh --all
```

不传参数时默认执行全量测试：

```bash
cd tests/
bash run_test.sh
```

**2. 仅执行 CPU 兼容测试（无需 NPU 硬件）**

```bash
cd tests/
bash run_test.sh --cpu_only
```

**3. 仅执行 NPU 依赖测试（需要 NPU 硬件）**

```bash
cd tests/
bash run_test.sh --npu_only
```

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

- 单个测试文件运行超过500秒时，应拆分为多个文件
- 优先复用已有的设备/模型加载，减少重复初始化开销
- CPU 友好的测试应优先使用 `bash tests/run_UT_test.sh` 验证
