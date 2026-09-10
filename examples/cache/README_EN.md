# DiTCache Acceleration Feature

This document uses the `Qwen-Image-Edit-2509` model as an example to demonstrate how to use the `DiTCache` acceleration feature for model optimization.

## Prerequisites

1. Download the weights.

   - The original weights are available from [HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

   - For users in China, the weights are available [here](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509)

2. Run the following commands in any path (for example, /home/{username}/example/) to download the model code and install the required dependencies.

    ```shell
    git clone https://modelers.cn/MindIE/Qwen-Image-Edit-2509.git && cd Qwen-Image-Edit-2509
    pip install -r requirements.txt
    ```

3. Copy the [cache.py](cache.py) file from the `examples/cache` directory to the Qwen-Image-Edit-2509 directory.

    For more information about this model, see [Modelers Community](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509).

## Enabling DiTCache

Run the following commands to enable cache optimization and perform inference. Observe the acceleration effect by comparing the average model inference time before and after enabling the cache.

```shell
export COND_CACHE=1
export UNCOND_CACHE=1

python cache.py  \
--model_path /mnt/data/Qwen-Image-Edit-2509  \
--device_id 0  \
--img_paths ./yarn-art-pikachu.png
```

Parameter description:

- `model_path`: Path to the model weights.

- `device_id`: ID of the device used for model inference.

- `img_paths`: Path to the input image. Separate multiple images with commas, for example, `img1,img2`.

Note: To disable cache acceleration, set the environment variables `COND_CACHE` and `UNCOND_CACHE` to `0`.
