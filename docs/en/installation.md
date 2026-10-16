# Installation Guide

## Python Package Installation

MindIE SD is a Python package built on PyTorch that can be easily integrated into Python applications.

### Dependencies

* OS: Linux
* Python: >=3.10
* PyTorch: 2.6, 2.7, 2.8, 2.9, 2.10
* TorchNPU: 2.6, 2.7, 2.8, 2.9, 2.10
* CANN: 9.0.0
* triton: 3.5.0
* triton-ascend: 3.2.1

> **Note**: `triton-ascend 3.2.1` is not published on PyPI. Download the wheel matching your Python version and architecture from the [GitCode release page](https://gitcode.com/Ascend/triton-ascend/releases) and install it (it pins `triton==3.5.0`). Without 3.2.1, the SparseLinearAttention (FA sparse) feature is disabled.

#### CANN Installation

MindIE SD depends on the CANN Toolkit development package and CANN ops operator package. See the [CANN Software Installation Guide](https://gitcode.com/cann/ops-cv/blob/master/docs/zh/install/quick_install.md) for installation instructions. Choose the installation scenario based on your installation method and operating system, then click "Start Reading" and follow the "Install CANN" section.

After installation, run the following command to set environment variables (using the default installation path as an example):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### Notes

1. MindIE SD primarily depends on the TorchNPU version and will try to meet the CANN and Python version requirements mandated by TorchNPU.
2. After CANN installation, the installation path provides a process-level environment variable setup script `set_env.sh` to automatically configure environment variables. This script includes LD_LIBRARY_PATH and ASCEND_CUSTOM_OPP_PATH as shown in [Table 1 Environment Variables](#table_environment0001). These settings automatically expire when the user process ends.

**Table 1** Environment Variables<a id="table_environment0001"></a>

| Environment Variable | Description |
| -- | -- |
| LD_LIBRARY_PATH | Dynamic library search path. |
| ASCEND_CUSTOM_OPP_PATH | Custom operator package installation path for inference engine. |
| ASCEND_RT_VISIBLE_DEVICES | Specifies the logical IDs of Ascend AI processors used by the current process. Configure as needed.<br>Example: "0,1,2" or "0-2"; Ascend AI processor logical IDs are separated by "," and consecutive IDs use "-". |

### Quick Install

Currently the simplest way is to install via pip. Our package is named `mindiesd`, which differs from the repository name.

```bash
pip install mindiesd
```

### Source Build

In some cases, you may need to install MindIE SD from source to try the latest features or customize the library for your specific needs.

Follow these steps to install MindIE SD from source:

1. Clone the repository and enter the project:

   ```bash
   git clone https://github.com/MindIE-SD/MindIE-SD.git && cd MindIE-SD
   ```

2. [Optional] Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build and install:

   ```bash
   python setup.py bdist_wheel
   cd dist
   pip install mindiesd-*.whl
   ```

> **Dependency Layering**
>
> Dependencies are split by purpose in the repository; install the ones you need:
>
> * `requirements.txt`: core runtime dependencies (minimal install; only torch/TorchNPU pinned, others loose).
> * `examples/dummy_run/requirements.txt`: dependencies for the `dummy_run` model inference example (diffusers/transformers/etc.).
> * `examples/service/requirements.txt`: serving-example dependencies (ray, fastapi, uvicorn, pydantic, Pillow).
> * Testing, linting, and docs-build dependencies are in `requirements-test.txt`, `requirements-lint.txt`, and `docs/requirements-docs.txt` respectively (see the developer guide).

### Nightly Build Installation

Nightly builds are available for testing the latest features:

Coming soon...

## Image Installation (vLLM-Omni)

The Docker image combines **vLLM-Omni + MindIE SD**. MindIE-SD 3.1.0 provides three product variants:

| Supported Product | Image Tag | Base Image |
| --- | --- | --- |
| Atlas 800I A2 inference server | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0` |
| Atlas 800I A3 SuperPoD Server | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0-a3` |
| 950PR&950DT products | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0-a5` |

Pull a published `mindiesd` image from the [MindIE image repository](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f), or build locally from the `dev` branch. The current base images provide `linux/amd64` only.

<!-- npu="910b" id1 -->
```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
git checkout dev && cd docker/omni
docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 -f Dockerfile.a2.ubuntu .
```
<!-- end id1 -->

<!-- npu="A3" id2 -->
```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
git checkout dev && cd docker/omni
docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 -f Dockerfile.a3.ubuntu .
```
<!-- end id2 -->

<!-- npu="950" id3 -->
```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
git checkout dev && cd docker/omni
docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12 -f Dockerfile.a5.ubuntu .
```
<!-- end id3 -->

See the [vLLM-Omni image overview](../../docker/omni/OVERVIEW.md) for runtime options and hardware requirements. The full image Tag list is in [Supported Tags](../../docker/supported_tags.md).
