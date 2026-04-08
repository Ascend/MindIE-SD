# Installation Guide

This document describes how to prepare a MindIE SD environment in container and bare-metal setups. If you want to run an example end to end, read this guide together with [Quick Start](quick_start.md).

## Option 1: Container image installation

This section describes how to prepare and use a MindIE container image.

1. Install the driver and firmware.

   Make sure the host already has the NPU driver and firmware installed. If not, see the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit) and choose the appropriate installation scenario.

   - Installation method: install on a physical machine.
   - Operating system: choose the operating system used in your environment.
   - Service scenario: choose training, inference, and development/debugging.

   Install Docker on the host yourself. Docker 24.x.x or later is recommended. Make sure the environment can access the network before configuring package sources.

2. Obtain the MindIE container image.

   - Open the [Ascend image repository page](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f).
   - Sign in with a Huawei account.
   - On the image version tab, select the image that matches your hardware form factor and download it.
   - Follow the download instructions shown by the site.

3. Use the image.

   The following options usually need to be adjusted for your environment:

   | Item | Location in command | Description |
   | ------ | ------------ | ---- |
   | Container name | `--name <container-name>` and `docker exec <container-name>` | Replace with your custom container name. |
   | Image name | `mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts` at the end of the command | Replace with the local image name and tag shown by `docker images`. |
   | Mount paths | The host-side path of each `-v` option, such as `/path-to-weights` | Replace with the real paths on the host machine. |

   Start the container with a command like this:

   ```bash
   docker run -it -d --net=host --shm-size=1g \
       --name <container-name> \
       --device=/dev/davinci_manager:rwm \
       --device=/dev/hisi_hdc:rwm \
       --device=/dev/devmm_svm:rwm \
       --device=/dev/davinci0:rwm \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware:ro \
       -v /usr/local/sbin:/usr/local/sbin:ro \
       -v /path-to-weights:/path-to-weights:ro \
       mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts bash
   ```

   > **Note**
   > The image name and tag above are only examples. Use `docker images` on the host to inspect the images already available on your machine.
   >
   > The `--device` mounts are set to `rwm` rather than `rw` or `r`. This avoids known runtime and `npu-smi` failures on Atlas 800I A2 and Atlas 800I A3 systems when NPUs are mounted with insufficient permissions.

   Enter the container:

   ```bash
   docker exec -it <container-name> bash
   ```

4. Install other dependencies.

   1. Install the model-specific dependencies before running inference. For Wan2.1, for example:

      ```bash
      git clone https://modelers.cn/MindIE/Wan2.1.git
      cd Wan2.1
      pip install -r requirements.txt
      ```

   2. Install `gcc` and `g++` if they are missing from the container, then export the include path:

      ```bash
      yum install gcc g++ -y
      export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
      ```

## Option 2: Bare-metal installation

This section describes how to prepare a full development environment on a physical machine, including the driver and firmware, CANN, PyTorch, Torch NPU, and MindIE SD installation paths.

1. Install the driver and firmware.

   Make sure the host already has the NPU driver and firmware installed. If not, follow the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit) and choose the matching installation scenario.

   - Installation method: install on a physical machine.
   - Operating system: choose the operating system used in your environment.
   - Service scenario: choose training, inference, and development/debugging.

2. Install CANN.

   Required CANN packages include:

   - the CANN Toolkit development package
   - the CANN Kernels operator package

   Follow the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit) and choose the physical-machine CANN installation path.

3. Install PyTorch and Torch NPU.

   Required packages include:

   - the PyTorch wheel package, version 2.1.0
   - the `torch_npu` wheel package

   - Install PyTorch by following the [Ascend Extension for PyTorch installation guide](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html).
   - Install `torch_npu` by following the optional extension module guide in the same documentation set.

   > **Note**
   > If `gcc` and `g++` are missing, install them and export the include path:
   >
   > ```bash
   > yum install gcc g++ -y
   > export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
   > ```

4. Install other environment dependencies.

   Install the model-specific dependencies required for inference:

   ```bash
   pip install -r requirements.txt
   ```

5. Install MindIE SD.

   MindIE SD does not need to be installed separately when you install the full MindIE package. To install the MindIE package:

   1. Upload the MindIE package to any path on the target machine, for example `/home/package`, then grant execute permission:

      ```bash
      cd /home/package
      chmod +x Ascend-mindie_<version>_linux-<arch>_<abi>.run
      ```

   2. Export the `ascend-toolkit` environment variables, for example with the default root installation path:

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

   3. Install the package:

      ```bash
      ./package-name.run --install --quiet
      ```

   4. Verify the installation:

      ```bash
      python3 -c "import torch, torch_npu, mindiesd; print(torch_npu.npu.is_available())"
      ```
