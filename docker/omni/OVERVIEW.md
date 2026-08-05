# MindIE-SD

## Quick Reference

| Item | Value |
|------|-------|
| **Image** | `mindiesd` |
| **Tags** | `v3.0.0-cann8.5.1-torch_npu2.9.0-910b-ubuntu22.04-py3.11-aarch64` |
| | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` |
| **Base Images** | Atlas 800I A2 inference server: `quay.io/ascend/vllm-omni:v0.20.0` |
| | Atlas 800I A3 SuperPoD Server: `quay.io/ascend/vllm-omni:v0.20.0-a3` |
| **Architecture** | `linux/arm64` (aarch64) |
| **OS** | Ubuntu 22.04 |
| **Python** | 3.11 |
| **CANN** | 8.5.1 |
| **TorchNPU** | 2.9.0 |
| **License** | Mulan PSL v2 |

This image is maintained by the [MindIE community](https://www.hiascend.com/en/developer/software/mindie).

Get help:

- [MindIE image repository](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
- [MindIE-SD documentation](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/en/index.md)
- [Atlas Developer Community](https://www.hiascend.com/developer)
- [Issue feedback](https://gitcode.com/Ascend/MindIE-SD/issues)

## Image Overview

This image combines **vLLM-Omni** and **MindIE-SD** (Mind Inference Engine Stable Diffusion) into a single container, enabling both multi-modal LLM inference and Stable Diffusion image generation on Atlas NPUs.

It is built on top of the `quay.io/ascend/vllm-omni` base image (which includes CANN 8.5.1, torch, TorchNPU, vllm, and vllm_ascend). Two variants are provided for different NPU series:

- **Atlas 800I A2 inference server**: based on `quay.io/ascend/vllm-omni:v0.20.0`, for Atlas 800I A2 inference server
- **Atlas 800I A3 SuperPoD Server**: based on `quay.io/ascend/vllm-omni:v0.20.0-a3`, for Atlas 800I A3 SuperPoD Server

Both variants add the following Atlas tuning and debugging tools:

| Component | Version | Description |
|-----------|---------|-------------|
| mindiesd | latest | MindIE Stable Diffusion inference engine |
| msprobe | 0.1.4 | Precision debugging tool |
| msmodelslim | 8.2.1 | Model compression and quantization tool |
| msprof-analyze | 26.0.0 | MindStudio Profiler analysis tool |
| msprof | *(bundled with CANN)* | NPU profiling tool |

## Image Tags & Dockerfile Path

### Tag Naming Convention

```text
{version}-{cann version}-{torch_npu version}-{supported product}-{os}-{python version}-{architecture}-{others}
```

| Series | Example Tag | Base Image |
|--------|-------------|------------|
| Atlas 800I A2 inference server | `v3.0.0-cann8.5.1-torch_npu2.9.0-910b-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0` |
| Atlas 800I A3 SuperPoD Server | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0-a3` |

### v3.0.0 Version Dockerfile Directory

Each series has a dedicated Dockerfile, archived in the `docker/omni` directory of the MindIE-SD source repository:

| Series | Example Tag | Dockerfile |
|--------|-------------|------------|
| Atlas 800I A2 inference server | `v3.0.0-cann8.5.1-torch_npu2.9.0-910b-ubuntu22.04-py3.11-aarch64` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/master/docker/omni/Dockerfile.a2.ubuntu) |
| Atlas 800I A3 SuperPoD Server | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/master/docker/omni/Dockerfile.a3.ubuntu) |

## Quick Start

### Pull Base Image

Browse all available tags at [quay.io/ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni?tab=tags).

Atlas 800I A2 inference server:

```bash
docker pull quay.io/ascend/vllm-omni:v0.20.0
```

Atlas 800I A3 SuperPoD Server:

```bash
docker pull quay.io/ascend/vllm-omni:v0.20.0-a3
```

> **Tip:** You can also use `podman pull` in place of `docker pull` if you prefer Podman as your container runtime.

### Run the Container

Atlas 800I A2 inference server:

```bash
docker run -it --rm --name=mindiesd \
    --privileged \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-910b-ubuntu22.04-py3.11-aarch64 \
    bash
```

Atlas 800I A3 SuperPoD Server:

```bash
docker run -it --rm --name=mindiesd \
    --privileged \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64 \
    bash
```

> **Note:** The `--privileged` flag and device mappings are required for NPU access. The host must mount driver libraries (`/usr/local/Ascend/driver/lib64`), driver version info, DCMI, `npu-smi`, and Atlas install info into the container.

### Build Locally

Clone the MindIE-SD repository and build from the `docker/omni` directory:

Atlas 800I A2 inference server:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD/docker/omni

docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-910b-ubuntu22.04-py3.11-aarch64 \
    -f Dockerfile.a2.ubuntu .
```

Atlas 800I A3 SuperPoD Server:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD/docker/omni

docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64 \
    -f Dockerfile.a3.ubuntu .
```

### Customize (Secondary Development)

To add your own dependencies or application code, create a new Dockerfile based on this image:

```dockerfile
FROM mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-910b-ubuntu22.04-py3.11-aarch64

# Add your custom packages
RUN pip install --no-cache-dir your-package

# Copy your application
COPY ./your-app /workspace/your-app
WORKDIR /workspace/your-app
```

## Hardware Support

| Item | Requirement |
|------|-------------|
| **NPU** | Atlas 800I A2 inference server |
| | Atlas 800I A3 SuperPoD Server |
| **Driver** | Atlas NPU driver must be installed on the host |
| **Host Mounts** | `/usr/local/dcmi`, `/usr/local/bin/npu-smi`, `/usr/local/Ascend/driver/lib64/`, `/usr/local/Ascend/driver/version.info`, `/etc/ascend_install.info`, `/root/.cache` |

## Compatibility Changes

Refer to the [MindIE-SD documentation](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/en/index.md) for the latest release notes and compatibility information.

## License & Disclaimer

This image is licensed under the **Mulan Permissive Software License, Version 2 (Mulan PSL v2)**. See the [LICENSE](https://gitcode.com/Ascend/MindIE-SD/blob/master/LICENSE.md) file for the full text.

By pulling and using this container image, you accept the terms and conditions of the Huawei Container License Agreement. A copy of the license is available at: https://www.hiascend.com/en/legal/ascendhub-download

You agree and undertake that when using Huawei or third-party software in this image, you will comply with the license agreement of the corresponding Huawei or third-party software.
