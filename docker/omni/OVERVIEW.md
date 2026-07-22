# vLLM-Omni + MindIE-SD (Ubuntu)

## Quick Reference

| Item | Value |
|------|-------|
| **Image** | `mindiesd` |
| **Tags** | `v3.0.0-A2-ubuntu22.04-py3.11-aarch64` |
| | `v3.0.0-A3-ubuntu22.04-py3.11-aarch64` |
| **Base Images** | A2: `quay.io/ascend/vllm-omni:v0.20.0` |
| | A3: `quay.io/ascend/vllm-omni:v0.20.0-a3` |
| **Architecture** | `linux/arm64` (aarch64) |
| **OS** | Ubuntu 22.04 |
| **Python** | 3.11 |
| **CANN** | 8.5.1 |
| **License** | Mulan PSL v2 |

## Image Overview

This image combines **vLLM-Omni** and **MindIE-SD** (Mind Inference Engine Stable Diffusion) into a single container, enabling both multi-modal LLM inference and Stable Diffusion image generation on Ascend NPUs.

It is built on top of the `quay.io/ascend/vllm-omni` base image (which includes CANN 8.5.1, torch, torch_npu, vllm, and vllm_ascend). Two variants are provided for different NPU series:

- **A2**: based on `quay.io/ascend/vllm-omni:v0.20.0`, for Ascend A2 series NPUs
- **A3**: based on `quay.io/ascend/vllm-omni:v0.20.0-a3`, for Ascend A3 series NPUs

Both variants add the following Ascend tuning and debugging tools:

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
{version}-cann{cann_version}-{series}-{os}-py{python_version}-{architecture}
```

| Series | Example Tag | Base Image |
|--------|-------------|------------|
| A2 | `v3.0.0-A2-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0` |
| A3 | `v3.0.0-A3-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0-a3` |

### Dockerfile Path

The Dockerfiles are archived in the MindIE-SD source repository at:

```text
docker/omni/Dockerfile.a2.ubuntu   # A2 series
docker/omni/Dockerfile.a3.ubuntu   # A3 series
```

## Quick Start

### Pull Base Image

Browse all available tags at [quay.io/ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni?tab=tags).

A2:

```bash
docker pull quay.io/ascend/vllm-omni:v0.20.0
```

A3:

```bash
docker pull quay.io/ascend/vllm-omni:v0.20.0-a3
```

> **Tip:** You can also use `podman pull` in place of `docker pull` if you prefer Podman as your container runtime.

### Run the Container

A2:

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
    mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64 \
    bash
```

A3:

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
    mindiesd:v3.0.0-A3-ubuntu22.04-py3.11-aarch64 \
    bash
```

> **Note:** The `--privileged` flag and device mappings are required for NPU access. The host must mount driver libraries (`/usr/local/Ascend/driver/lib64`), driver version info, DCMI, `npu-smi`, and Ascend install info into the container.

### Build Locally

Clone the MindIE-SD repository and build from the `docker/omni` directory:

A2:

```bash
git clone https://github.com/Ascend/MindIE-SD.git
cd MindIE-SD/docker/omni

docker build -t mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64 \
    -f Dockerfile.a2.ubuntu .
```

A3:

```bash
git clone https://github.com/Ascend/MindIE-SD.git
cd MindIE-SD/docker/omni

docker build -t mindiesd:v3.0.0-A3-ubuntu22.04-py3.11-aarch64 \
    -f Dockerfile.a3.ubuntu .
```

### Customize (Secondary Development)

To add your own dependencies or application code, create a new Dockerfile based on this image:

```dockerfile
FROM mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64

# Add your custom packages
RUN pip install --no-cache-dir your-package

# Copy your application
COPY ./your-app /workspace/your-app
WORKDIR /workspace/your-app
```

## Hardware Support

| Item | Requirement |
|------|-------------|
| **NPU** | Ascend A2 series (e.g., Atlas 300I Duo, Atlas A2 training series) |
| | Ascend A3 series |
| **Driver** | Ascend NPU driver must be installed on the host |
| **Host Mounts** | `/usr/local/dcmi`, `/usr/local/bin/npu-smi`, `/usr/local/Ascend/driver/lib64/`, `/usr/local/Ascend/driver/version.info`, `/etc/ascend_install.info`, `/root/.cache` |

## Compatibility Changes

Refer to the [MindIE-SD documentation](../../docs/en/index.md) for the latest release notes and compatibility information.

## License & Disclaimer

This image is licensed under the **Mulan Permissive Software License, Version 2 (Mulan PSL v2)**. See the [LICENSE](../../LICENSE.md) file for the full text.

By pulling and using this container image, you accept the terms and conditions of the Huawei Container License Agreement. A copy of the license is available at: https://www.hiascend.com/en/legal/ascendhub-download

You agree and undertake that when using Huawei or third-party software in this image, you will comply with the license agreement of the corresponding Huawei or third-party software.
