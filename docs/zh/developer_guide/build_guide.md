# 编译指导

## 编译说明

本文档介绍如何从源码编译MindIE-SD，生成 `.whl` 包，安装与运行。

## 环境准备

## 镜像安装方式

MindIE镜像获取请参见[镜像安装方式](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/image_usage_guide.md#%E8%8E%B7%E5%8F%96mindie%E9%95%9C%E5%83%8F)。

## 容器/物理机安装方式

1. 容器/物理机安装方式，需要准备的软件包和依赖请参见[准备软件包和依赖](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/preparing_software_and_dependencies.md)。
2. 容器/物理机安装方式，软件包和依赖的安装请参见[安装软件包和依赖](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/installing_software_and_dependencies.md)。

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

2. 安装MindIE SD。
   - 方式一：常规安装（使用默认版本号）

       ```bash
       cd dist
       pip install mindiesd-*.whl
       ```

   - 方式二：开发者可编辑模式安装（可通过环境变量MINDIE_SD_VERSION_OVERRIDE修改版本号）

       ```bash
       pip install -e .
       ```

## 升级

详情请参见[升级](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/upgrade.md)章节。

## 卸载

详情请参见[卸载](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/uninstallation.md)章节。
