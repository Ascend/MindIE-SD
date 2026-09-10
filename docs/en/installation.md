# Installation Guide

## Python Package Installation

MindIE SD is a Python package built on PyTorch that can be easily integrated into Python applications.

Python package installation works for most use cases, but requires manual CANN setup. To skip this step, consider using the container image approach: pull the image directly from the Ascend community and start a container.

### Dependencies

* OS: Linux
* Python: >=3.10
* PyTorch: 2.6, 2.7, 2.8, 2.9, 2.10
* TorchNPU: 2.6, 2.7, 2.8, 2.9, 2.10
* CANN: 9.0.1

#### CANN Installation

MindIE SD depends on the CANN Toolkit development package and CANN ops operator package. See the  <a href="https://gitcode.com/cann/ops-cv/blob/master/docs/zh/install/quick_install.md" target="_blank" rel="noopener noreferrer">CANN Software Installation Guide</a>.

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

Currently the simplest way is to install via pip. Our package is named `mindiesd`, which differs from the repository name. Before installing `mindiesd`, install the required Python dependencies:

> The `requirements.txt` file is located in the root directory of this repository. To obtain it, either clone the repository (`git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD`) or download `requirements.txt` separately from the repository.

```bash
pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
```

Then install `mindiesd`:

```bash
pip install mindiesd
```

### Source Build

In some cases, you may need to install MindIE SD from source to try the latest features or customize the library for your specific needs.

Follow these steps to install MindIE SD from source:

1. Clone the repository and enter the project.

   ```bash
   git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD && git checkout dev
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
   ```

   > [!NOTE]
   > Some operators in MindIE SD depend on `triton-ascend==3.2.1`, which is currently only available at <https://triton-ascend.osinfra.cn/pypi/simple>.

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
> - `requirements.txt`: core runtime dependencies (minimal install; only torch/TorchNPU pinned, others loose).
> - `examples/dummy_run/requirements.txt`: dependencies for the `dummy_run` model inference example (diffusers/transformers/etc.).
> - `examples/service/requirements.txt`: serving-example dependencies (ray, fastapi, uvicorn, pydantic, Pillow).
> - Testing, linting, and docs-build dependencies are in `requirements-test.txt`, `requirements-lint.txt`, and `docs/requirements-docs.txt` respectively (see the developer guide).

## Installation via Image (vLLM-Omni)

In addition to the Python package installation, we provide a Docker image that bundles **vLLM-Omni + MindIE-SD**, enabling simultaneous multimodal LLM inference and Stable Diffusion image generation on Ascend NPUs. The image is built upon the `quay.io/ascend/vllm-omni` base image and offers two variants:

| Product | Image Tag | Base Image |
|--|--|--|
| Atlas 800I A2 Inference Server | `v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0` |
| Atlas 800I A3 SuperPoD Server | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0-a3` |

**Obtaining the Image (Two Methods):**

* (Recommended) Pull the pre-built `mindiesd` image from the [MindIE Image Registry](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f).
* Build locally: clone the repository and navigate to the `docker/omni` directory, then build using the Dockerfile for your product:

  ```bash
  git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD/docker/omni
  # For Atlas 800I A2 Inference Server
  docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64 -f Dockerfile.a2.ubuntu .
  # For Atlas 800I A3 SuperPoD Server
  docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64 -f Dockerfile.a3.ubuntu .
  ```

For details on runtime parameters, hardware requirements, and development, refer to the [vLLM-Omni Image Guide](../../docker/omni/OVERVIEW.md).
