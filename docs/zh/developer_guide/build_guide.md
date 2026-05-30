# 编译指导

## 编译说明

本文档介绍如何从源码编译MindIE-SD。安装已编译的 `.whl` 包请参见[安装指导](../installation.md#源码安装)。

## 编译安装

1. 使用以下命令拉取代码。

    ```bash
    git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
    python -m build --wheel --no-isolation
    ```

    >[!NOTE]说明
    >若环境中没有wheel等依赖，请用户使用以下命令自行安装。
    >
    >```bash
    >pip install build wheel
    >```

2. 安装生成的 `.whl` 包请参见[安装指导](../installation.md#源码安装)。

   如需可编辑模式安装（适合开发者调试，可通过环境变量 `MINDIE_SD_VERSION_OVERRIDE` 修改版本号），可使用：

   ```bash
   pip install -e .
   ```
