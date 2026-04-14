# Build Guide

## Overview

This document describes how to build MindIE-SD from source, generate a `.whl` package, install it, and run it locally.

## Environment Preparation

## Image-Based Installation

For MindIE image acquisition, refer to the [image installation guide](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/image_usage_guide.md#%E8%8E%B7%E5%8F%96mindie%E9%95%9C%E5%83%8F).

## Container or Bare-Metal Installation

1. For container or bare-metal installation, refer to [preparing software packages and dependencies](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/preparing_software_and_dependencies.md).
2. For dependency installation steps, refer to [installing software packages and dependencies](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/installing_software_and_dependencies.md).

## Build and Install

1. Clone the repository and build the wheel:

   ```bash
   git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
   python -m build --wheel --no-isolation
   ```

   > **Note**
   > If `wheel` or related build dependencies are missing, install them first:
   >
   > ```bash
   > pip install build wheel
   > ```

2. Install MindIE SD.

   - Option 1: Standard installation with the default version

     ```bash
     cd dist
     pip install mindiesd-*.whl
     ```

   - Option 2: Editable installation for development

     ```bash
     pip install -e .
     ```

## Upgrade

For upgrade instructions, refer to the [upgrade guide](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/upgrade.md).

## Uninstall

For uninstall instructions, refer to the [uninstallation guide](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/install/source/uninstallation.md).
