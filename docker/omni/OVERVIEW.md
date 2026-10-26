# MindIE SD

## Quick Reference

| Item | Value |
|------|-------|
| **Image** | `mindiesd` |
| **Tags** | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` |
| | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` |
| | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` |
| **Base Images** | Atlas 800I A2 inference server: `quay.io/ascend/vllm-omni:v0.28.0` |
| | Atlas 800I A3 SuperPoD Server: `quay.io/ascend/vllm-omni:v0.28.0-a3` |
| | 950PR&950DT products: `quay.io/ascend/vllm-omni:v0.28.0-a5` |
| **Architecture** | `linux/amd64` (x86_64) |
| **OS** | Ubuntu 22.04 |
| **Python** | 3.12 |
| **CANN** | 9.1.0 |
| **TorchNPU** | 2.10.0.post4 |
| **License** | Mulan PSL v2 |

This image is maintained by the [MindIE community](https://www.hiascend.com/en/developer/software/mindie).

Get help:

- [MindIE image repository](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
- [MindIE SD documentation](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/en/index.md)
- [Atlas Developer Community](https://www.hiascend.com/developer)
- [Issue feedback](https://gitcode.com/Ascend/MindIE-SD/issues)

## Image Overview

This image combines **vLLM-Omni** and **MindIE SD** (Mind Inference Engine Stable Diffusion) into a single container, enabling both multi-modal LLM inference and Stable Diffusion image generation on Atlas NPUs.

It is built on top of the `quay.io/ascend/vllm-omni` base image (which includes CANN 9.1.0, torch, TorchNPU, vllm, and vllm_ascend). Three variants are provided for different NPU series:

- **Atlas 800I A2 inference server**: based on `quay.io/ascend/vllm-omni:v0.28.0`, for Atlas 800I A2 inference server
- **Atlas 800I A3 SuperPoD Server**: based on `quay.io/ascend/vllm-omni:v0.28.0-a3`, for Atlas 800I A3 SuperPoD Server
- **950PR&950DT products**: based on `quay.io/ascend/vllm-omni:v0.28.0-a5`, for 950PR&950DT products

All three base tags publish `linux/amd64` only.

All three variants add the following Atlas tuning and debugging tools:

| Component | Version | Description |
|-----------|---------|-------------|
| mindiesd | 3.1.0 | MindIE Stable Diffusion inference engine |
| msprobe | 0.1.4 | Precision debugging tool |
| msmodelslim | 8.2.1 | Model compression and quantization tool |
| msprof-analyze | 26.0.0 | MindStudio Profiler analysis tool |
| msprof | *(bundled with CANN)* | NPU profiling tool |

## Image Tags & Dockerfile Path

### Tag Naming Convention

```text
{version}-{cann version}-{torch_npu version}-{supported product}-{os}-{python version}
```

| Series | Example Tag | Base Image |
|--------|-------------|------------|
| Atlas 800I A2 inference server | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0` |
| Atlas 800I A3 SuperPoD Server | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0-a3` |
| 950PR&950DT products | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0-a5` |

### v3.1.0 Version Dockerfile Directory

Each series has a dedicated Dockerfile, archived in the `docker/omni` directory of the MindIE SD source repository:

| Series | Example Tag | Dockerfile |
|--------|-------------|------------|
| Atlas 800I A2 inference server | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/omni/Dockerfile.a2.ubuntu) |
| Atlas 800I A3 SuperPoD Server | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/omni/Dockerfile.a3.ubuntu) |
| 950PR&950DT products | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/omni/Dockerfile.a5.ubuntu) |

Released image tags, including v3.0.0, are listed in [Supported Tags](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/supported_tags.md).

## Quick Start

### Pull Base Image

Browse all available tags at [quay.io/ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni?tab=tags).

<!-- npu="910b" id1 -->
Atlas 800I A2 inference server:

```bash
docker pull quay.io/ascend/vllm-omni:v0.28.0
```
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas 800I A3 SuperPoD Server:

```bash
docker pull quay.io/ascend/vllm-omni:v0.28.0-a3
```
<!-- end id2 -->

<!-- npu="950" id3 -->
950PR&950DT products:

```bash
docker pull quay.io/ascend/vllm-omni:v0.28.0-a5
```
<!-- end id3 -->

> **Tip:** You can also use `podman pull` in place of `docker pull` if you prefer Podman as your container runtime.

### Run the Container

<!-- npu="910b" id4 -->
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
    mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 \
    bash
```
<!-- end id4 -->

<!-- npu="A3" id5 -->
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
    mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 \
    bash
```
<!-- end id5 -->

<!-- npu="950" id6 -->
950PR&950DT products:

```bash
docker run -it --rm --name=mindiesd \
    --privileged \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12 \
    bash
```
<!-- end id6 -->

> **Note:** The `--privileged` flag and device mappings are required for NPU access. The host must mount driver libraries (`/usr/local/Ascend/driver/lib64`), driver version info, DCMI, `npu-smi`, and Atlas install info into the container.

### Build Locally

Clone the MindIE SD repository and build from the `docker/omni` directory:

<!-- npu="910b" id7 -->
Atlas 800I A2 inference server:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git checkout dev
cd docker/omni

docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 \
    -f Dockerfile.a2.ubuntu .
```
<!-- end id7 -->

<!-- npu="A3" id8 -->
Atlas 800I A3 SuperPoD Server:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git checkout dev
cd docker/omni

docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 \
    -f Dockerfile.a3.ubuntu .
```
<!-- end id8 -->

<!-- npu="950" id9 -->
950PR&950DT products:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git checkout dev
cd docker/omni

docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12 \
    -f Dockerfile.a5.ubuntu .
```
<!-- end id9 -->

### Customize (Secondary Development)

To add your own dependencies or application code, create a new Dockerfile based on this image:

```dockerfile
FROM mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12

# Add your custom packages
RUN pip install --no-cache-dir your-package

# Copy your application
COPY ./your-app /workspace/your-app
WORKDIR /workspace/your-app
```

## Basic Verification Flow (950PR&950DT products)

After the container starts, complete basic verification in three steps, all inside the container. This flow was verified on a physical Ascend 950PR machine (host: Ubuntu 24.04, driver 25.7.rc1.10).

> **Note on the verified image tag**: verification used the tag actually published on the Ascend image registry, `v3.1.0-A5-ubuntu22.04-py3.11` (`swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd`), which currently differs from the formal tags in the tables above; the `py3.11` in that tag also does not match the Python 3.12.13 actually shipped inside the image. The discrepancy has been reported to maintainers in [issue #267](https://gitcode.com/Ascend/MindIE-SD/issues/267); always pull images per the [Ascend image registry](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f) page.

### 1. NPU environment check

```bash
npu-smi info
```

Expected output (excerpt from the Ascend 950PR test):

```text
| NPU ID | Name        | Health | Power(W) | Temp(C) | HBM-Usage(MB)  |
|--------|-------------|--------|----------|---------|----------------|
| 0      | Ascend950PR | OK     | 211.7    | 55      | 5236 / 131072  |
```

### 2. CANN / TorchNPU self-check

```bash
python3 -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('npu available:', torch.npu.is_available())"
```

Measured output (Ascend 950PR):

```text
torch: 2.10.0+cpu
torch_npu: 2.10.0.post4
npu available: True
```

### 3. MindIE-SD weight-free smoke inference

Use the weight-free entry points in `examples/dummy_run`; no model weights are downloaded (only KB-sized config files are fetched automatically). The example below uses Wan2.2 (FLUX.1-dev is a gated repository and requires HuggingFace access approval; `flux_infer.py` and other entries share the same arguments):

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
pip install -r examples/dummy_run/requirements.txt
python examples/dummy_run/wan_infer.py --device_id 0 --num_layers 2
```

Measured output excerpt (Ascend 950PR, with per-phase timing and memory statistics):

```text
INFERENCE                  Time(ms)   Mem(GB)   Peak(GB)
-- Timed --
text_encoder                  5.36      8.63      12.08
transformer                 1051.80      8.65      12.08
OVERALL TOTAL               2718.37                12.08
Wan2.2 dummy weight verification PASSED
```

## Known Limitations (950PR&950DT products)

- The base image tags are provided for `linux/amd64` (x86_64) only.
- The `/dev/devmm_svm` device does not exist on the tested Ascend 950PR environment (it does exist on 910B/A3 environments). The 950PR&950DT products run command above has removed that device mapping accordingly; do not mount it in custom launch commands, otherwise `docker run` fails immediately because the device is missing.
- The `/dev/davinciN` device mappings in the run command must be adjusted to the actual number of NPUs: on the tested single-card 950PR environment only `/dev/davinci0` exists, and copying the multi-card example with `davinci1`~`davinci3` verbatim fails because those devices are missing.
- The base image defaults to `SOC_VERSION=ascend950dt_9582` (950DT); it also works on Ascend 950PR through runtime adaptation, as verified on a physical machine.
- The host must install an NPU driver matching CANN 9.1.0 (25.7.rc1.10 was tested) and Ascend Docker Runtime, and mount the driver libraries and device files as shown above.
- The formal tag naming in this repository (for example `v3.1.0-...-950-...-py3.12`) differs from the tags actually published on the Ascend image registry (`v3.1.0-A5-ubuntu22.04-py3.11`, whose image actually ships Python 3.12.13). Always pull images per the [Ascend image registry](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f) page; the mapping is pending maintainer confirmation (see [issue #267](https://gitcode.com/Ascend/MindIE-SD/issues/267)).
- In restricted environments without a container runtime (for example development containers where creating namespaces is forbidden), an equivalent verification can be done via `skopeo copy` to fetch the image, `umoci unpack` to extract the rootfs, and `chroot` into it. See [issue #267](https://gitcode.com/Ascend/MindIE-SD/issues/267) for the full record.

## Compatibility Changes

Refer to the [MindIE SD documentation](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/en/index.md) for the latest release notes and compatibility information.

## License & Disclaimer

This image is licensed under the **Mulan Permissive Software License, Version 2 (Mulan PSL v2)**. See the [LICENSE](https://gitcode.com/Ascend/MindIE-SD/blob/master/LICENSE.md) file for the full text.

By pulling and using this container image, you accept the terms and conditions of the Huawei Container License Agreement. A copy of the license is available at: <https://www.hiascend.com/en/legal/ascendhub-download>

You agree and undertake that when using Huawei or third-party software in this image, you will comply with the license agreement of the corresponding Huawei or third-party software.
