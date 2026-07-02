# DiTCache Acceleration

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:15:29.039Z pushedAt=2026-06-09T06:01:31.150Z -->

Using the `Qwen-Image-Edit-2509` model as an example, this document demonstrates how to leverage the `DiTCache` acceleration feature for model optimization.

## Prerequisites

1. Download weights.

   - The original weights are from [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).

   - In this document, weights are downloaded from [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509).

2. Use the following commands to download the model code and install the required dependencies in any path (for example: `/home/{username}/example/`).

    ```shell
    git clone https://modelers.cn/MindIE/Qwen-Image-Edit-2509.git && cd Qwen-Image-Edit-2509
    pip install -r requirements.txt
    ```

3. Copy the [cache.py](cache.py) file from the `examples/cache` directory to the `Qwen-Image-Edit-2509` directory.

For more details about this model, see [Modelers Community](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509).

## Enabling DiTCache

Run the following command to enable cache optimization and perform inference. Compare the average inference time before and after enabling cache to observe the speedup.

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

- `device_id`: Device ID for model inference.

- `img_paths`: Path to the input images. For multiple images, separate them with commas, e.g., `img1,img2`.

**NOTE**: To disable cache acceleration, set the environment variables `COND_CACHE` and `UNCOND_CACHE` to `0`.
