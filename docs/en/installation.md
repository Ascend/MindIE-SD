# Installation Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:13:13.950Z pushedAt=2026-06-08T09:29:13.227Z -->

## Python Package Installation

MindIE SD is a Python package built on PyTorch, which can be easily integrated into Python applications.

### Installation Dependencies

| Version | Dependencies |
| ---- | ---- |
| dev  | <ul><li>OS: Linux</li><li>Python: >=3.10</li><li>PyTorch: 2.6, 2.7, 2.8, 2.9, 2.10</li><li>torch-npu: 2.6, 2.7, 2.8, 2.9, 2.10</li><li>CANN: 9.0.0</li><li>triton: 3.5.0</li><li>triton-ascend: 3.2.1</li></ul>|
| master | <ul><li>OS: Linux</li><li>Python: >=3.10</li><li>PyTorch: 2.6, 2.7, 2.8, 2.9</li><li>torch-npu: 2.6, 2.7, 2.8, 2.9</li><li>CANN: 8.5.1</li></ul>|

#### **Precautions**

1. MindIE SD primarily depends on the `torch-npu` version. Ensure that the required versions of CANN and Python are met accordingly.

2. After the CANN version is installed, the installation directory provides a process-level environment variable script `set_env.sh` to automatically configure the environment variables. This script includes `LD_LIBRARY_PATH` and `ASCEND_CUSTOM_OPP_PATH` as shown in [Table 1 Environment variables](#table_environment0001), and it automatically becomes invalid after the user process ends.

**Table 1** Environment variables<a id="table_environment0001"></a>

|Environment Variable|Description|
|--|--|
|LD_LIBRARY_PATH|Search path for dynamic libraries|
|ASCEND_CUSTOM_OPP_PATH|Installation path for the inference engine's custom operator package|
|ASCEND_RT_VISIBLE_DEVICES|The logical IDs of the Ascend AI processors used by the current process. Configure this if needed.<br>Configuration example: `0,1,2` or `0-2`; logical IDs of Ascend AI processors are separated by "," and a range is indicated by "-".|

### Quick Installation

The easiest way now is to install via pip source. The package name is `mindiesd`, which is slightly different from the repository name.

```bash
pip install mindiesd
```

### Source Code Installation

In some cases, you may need to install MindIE SD from source code to try the latest features or customize the library according to your specific needs.

You can install MindIE SD from source code by following these steps:

1. Clone the repository and enter the project.

   ```bash
   git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
   ```

2. [Optional] Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Build and install the package.

   ```bash
   python setup.py bdist_wheel
   cd dist
   pip install mindiesd-*.whl
   ```

### Daily Build and Installation

The daily build version is available for testing the latest features:

To be provided...
