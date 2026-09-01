# 仓库结构

## 顶层目录

| 目录 | 定位 |
|------|------|
| `mindiesd/` | 核心 Python 包，包含层算子、编译、缓存、并行、量化等模块 |
| `csrc/` | C/C++ 源码（AscendC / triton 算子实现） |
| `tests/` | 测试套件，按模块与 mindiesd/ 层次对应 |
| `examples/` | 模型推理示例（dummy_run、cache、service、wan） |
| `docs/` | Sphinx 文档源码（中英双语） |
| `docker/` | 开发容器与镜像定义 |
| `build/` | 构建产物 |
| `benchmarks/` | 性能基准数据 |
| `pre-commit/` | Pre-commit 补充配置（pyproject.toml、typos.toml） |

## mindiesd/ 关键模块

| 模块 | 说明 |
|------|------|
| `layers/` | 对外加速 API（attn、moe、quant 等 layer 实现） |
| `kernel/` | 昇腾高性能 kernel 封装 |
| `cache/` | 以存代算（Attention Cache、DiT Cache） |
| `parallelism/` | 多卡并行（CFG、USP、EPLB） |
| `quantization/` | 量化自动使能 |
| `compilation/` | torch.compile 自定义融合 pass |

## tests/ 测试架构

| 子目录 | 测试对象 |
|--------|----------|
| `tests/cache/` | 缓存模块 |
| `tests/compilation/` | 编译模块（含 pattern 测试） |
| `tests/layers/` | Layer 模块（含 flash_attn 测试） |
| `tests/plugin/` | 自定义 plugin 算子精度 |
| `tests/quantization/` | 量化模块 |
| `tests/eplb/` | 动态专家负载均衡 |
| `tests/UT/` | CPU 友好单元测试 |
| `tests/scripts/` | 测试辅助脚本（覆盖率检查等） |

测试入口详见 [test.md](test.md)。

## docs/ 文档架构

| 路径 | 说明 |
|------|------|
| `docs/zh/` | 中文文档 |
| `docs/en/` | 英文文档 |
| `docs/zh/developer_guide/` | 开发者指南（本文所在目录） |
| `docs/zh/features/` | 加速特性说明 |
| `contributing.md` | 贡献流程与治理规则 |

## 关键配置文件

| 文件 | 说明 |
|------|------|
| `setup.py` / `pyproject.toml` | Python 包构建与分发 |
| `version.py` | 版本号定义 |
| `requirements.txt` | 核心依赖 |
| `requirements-test.txt` | 测试依赖 |
| `requirements-lint.txt` | Lint 依赖 |
| `.pre-commit-config.yaml` | Pre-commit hook 配置 |
| `.clang-format` | C++ 代码格式 |
| `.coveragerc` | 覆盖率配置 |
| `.readthedocs.yaml` | Read the Docs 构建 |
