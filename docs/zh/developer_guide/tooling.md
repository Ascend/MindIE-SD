# 开发工具与文档

本文档说明 MindIE SD 本地开发中常用的文档生成、开发镜像构建与提交前检查命令。

## 文档生成

文档构建依赖位于 `docs/requirements-docs.txt`。本地生成 HTML 文档可使用以下命令：

```bash
python -m pip install -r docs/requirements-docs.txt
# 构建中英文
python docs/build_docs.py
# 仅构建中文
SPHINX_LANGUAGE=zh sphinx-build -b html -c docs docs/zh docs/_build/zh/html
# 仅构建英文
SPHINX_LANGUAGE=en sphinx-build -b html -c docs docs/en docs/_build/en/html
```

本地预览方法：

```bash
python -m http.server 8080 --directory docs/_build
```

http://localhost:8080 → 自动跳转中文版

## 开发镜像构建

仓库提供了基于Atlas 800I A2 推理服务器的AArch64环境的开发镜像定义文件 `docker/Dockerfile_910b_aarch64.ubuntu`。本地构建镜像可使用：

```bash
docker build --network=host -f docker/Dockerfile_910b_aarch64.ubuntu -t mindiesd:910b-aarch64-head .
```

## Lint 与提交前检查

Lint 相关依赖位于 `requirements-lint.txt`。首次在本地开发前，建议先安装并启用 `pre-commit`：

```bash
python -m pip install -r requirements-lint.txt
pre-commit install
pre-commit run --all-files
```

`pre-commit install` 会将仓库 hook 写入 `.git/hooks/pre-commit`。完成一次安装后，后续 `git commit` 会自动执行当前仓库默认启用的检查。

如果需要显式执行 Markdown 文档检查，请额外运行：

```bash
pre-commit run markdownlint --all-files --hook-stage manual
```

只有在明确需要绕过检查时，才使用 `git commit --no-verify`。
