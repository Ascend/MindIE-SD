# CPU 覆盖率补充

本报告对应 [Issue #269](https://gitcode.com/Ascend/MindIE-SD/issues/269)。新增 26 项 CPU 单元测试，覆盖日志配置、SoC 识别和 EPLB 状态管理。

## 测试范围

- `EnvVar`：组件配置优先级、日志关闭、控制字符、轮转参数边界。
- `get_npu_device`：支持的 SoC 区间、未知 SoC、驱动异常、查询缓存。
- `TaskPayload`：类型校验、必填字段、独立默认字典。
- `ExpertLoadCollector`：累积、读取后复位、任务通知。
- `DynamicDispatcher`：专家映射、全 rank 就绪判定、权重与布局更新的消费状态。

测试只替换硬件边界：NPU token 计数用 CPU `bincount` 表示，锁页分配用普通 CPU 张量表示，跨 rank 的 `all_reduce` 用就绪/未就绪结果表示。真实状态管理代码保持原样。这些测试验证 CPU 契约，未验证 NPU kernel 或多卡 EP 通信。

## 覆盖率结果

统计使用仓库 `.coveragerc`：`source=mindiesd`，`branch=True`。基线与补充后执行相同的 `tests/UT tests/utils tests/eplb` 范围。分母包含整个 `mindiesd`，未过滤未命中的模块。

下表是 coverage 的行与分支综合覆盖率，即 `(覆盖语句 + 覆盖分支) / (总语句 + 总分支)`。

| 模块 | 基线 | 补充后 |
| --- | ---: | ---: |
| `utils/env.py` | 78.31% | 100% |
| `utils/get_platform.py` | 43.18% | 100% |
| `eplb/task_payload.py` | 75.00% | 100% |
| `eplb/collector.py` | 0% | 100% |
| `eplb/dispatcher.py` | 0% | 100% |
| 总计 | 16.78% | 18.26% |

总语句覆盖率从 21.07% 升至 22.64%。语句分母保持 8862，分支分母保持 2888。基线为 217 passed、5 skipped；补充后为 243 passed、5 skipped。

## 复现

在仓库根目录运行。先准备仓库依赖和 `requirements-test.txt` 中的测试依赖。

```bash
export ASCEND_RT_VISIBLE_DEVICES=-1
export MINDIE_TEST_MODE=CPU
export MINDIE_LOG_TO_FILE=false
export MINDIE_LOG_TO_STDOUT=false
export COVERAGE_FILE=/path/to/task/coverage.data
python -m coverage run --rcfile .coveragerc -m pytest tests/UT tests/utils tests/eplb \
    --junitxml=/path/to/task/tests.xml
python -m coverage json --rcfile .coveragerc -o /path/to/task/coverage.json
python -m coverage report --rcfile .coveragerc
```

将基线和补充后的报告分别保存。两次使用同一环境、命令、配置和源码文件集合。不要将缩小测试范围后的覆盖率当成完整 NPU 回归结果。

## 环境与限制

- 源码基线：`44fbd543133ea8ced34b63c7579e78776c236108`，`dev`。
- openEuler、aarch64、Python 3.11.6、torch/torch_npu 2.9.0、CANN 9.1.0。
- 机器为单卡 910B3；本轮 CPU 测试隐藏 NPU。
- 五个既有 `TestEplbScheduler` 测试因 `skipIf(True, ...)` 固定跳过，未计入通过数。
- Triton-Ascend 缺失产生一个既有告警。未执行完整 NPU 回归或多卡 EPLB 对比。
- 首次未隐藏 NPU 时，导入因缺少 `libPTAExtensionOPS.so` 失败。该记录保留，不作为覆盖率基线。

## 静态检查结果

改动文件的适用 pre-commit hooks、手动 Markdownlint、双语 Sphinx 严格构建和 diff 检查通过。原生算子 hooks 无适用文件，未计为通过。

基线与补充后的全仓 pre-commit 在独立副本中均退出 1。两边有相同的 11 类失败：trailing whitespace、EOF fixer、ruff check、ruff format、codespell、pylint、bandit、typos、profiling-collect wiring selftest、clang-format、clang-tidy。其中 clang-tidy 工具缺失。该比较核对失败 hook 集合，不声称逐条诊断完全一致。

全仓检查产生的自动修改只保存在检查副本中。提交源码保留本次测试和文档的最小改动。
