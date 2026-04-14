# MindIE SD Docker

## 910B aarch64

`docker/Dockerfile_910b_aarch64.ubuntu` is based directly on the CI image definition in `MindIE-CI/env/version/Dockerfile.py311.arm._2.9.0`.

The main CI build chain is kept in place:

- `ubuntu:24.04`
- Python `3.11.4`
- CANN `8.5.1`
- `torch 2.9.0`
- matching `torch_npu`

The SD-specific changes are limited to:

- copying the `MindIE-SD` source tree into the image
- installing the workspace `requirements.txt`
- running `python3.11 -m build --wheel --no-isolation`
- exposing `/workspace/MindIE-SD` as the working directory

Build example:

```bash
cd MindIE-SD
docker build --network=host -f docker/Dockerfile_910b_aarch64.ubuntu -t mindiesd:910b-aarch64-head .
```

Run example:

```bash
docker run -itd \
  --name mindiesd-910b-test \
  --privileged \
  --ipc=host \
  --net=host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  mindiesd:910b-aarch64-head
```

Run test:

```bash
docker exec -it mindiesd-910b-test bash -lc 'python3 -m pip install coverage && cd /workspace/MindIE-SD && bash tests/run_test.sh --all'
```
