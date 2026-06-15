# Build Guide

## Build Instructions

This document describes how to build MindIE-SD from source. For installing the compiled `.whl` package, see [Installation Guide](../installation.md#source-build).

## Build and Install

1. Clone the code using the following command:

    ```bash
    git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
    python -m build --wheel --no-isolation
    ```

    > [!NOTE] Note
    > If dependencies such as wheel are not available in the environment, install them manually:
    >
    > ```bash
    > pip install build wheel
    > ```

2. For installing the generated `.whl` package, see [Installation Guide](../installation.md#source-build).

    For editable-mode installation (suitable for developer debugging; the version number can be overridden via the `MINDIE_SD_VERSION_OVERRIDE` environment variable):

    ```bash
    pip install -e .
    ```
