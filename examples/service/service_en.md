# Serving Acceleration Features

## Serving scheduling

Serving refers to starting an HTTP-based service (for example, a service that generates videos from text). Users can send requests to the backend through URLs to complete end-to-end model inference.

The Wan 2.2 model is used as an example. This model can generate videos from text or images. The generated videos can be directly returned to users or saved to a specified disk location. For example, you can run the following command to start an HTTP service, with the model path being `./Wan2.2-I2V-A14B/`, fsdp enabled for DIT and T5, and the number of Ulysses parallelism being 8 and the VAE parallelism policy used to reduce the graphics memory usage.
`server.py` is the script for starting the serving. Before starting the service, install the dependencies required for serving (including ray, fastapi, uvicorn, pydantic, and Pillow). For details, see `examples/service/requirements.txt`.

Ensure that the serving can access Wan by referring to the model link [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2).

```shell
# Install the dependencies required for serving (ray, fastapi, uvicorn, pydantic, and Pillow)
pip install -r examples/service/requirements.txt

model_base="/Wan2.2-I2V-A14B"

export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

python server.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--vae_parallel \
--sample_steps 40 \
--use_rainfusion \
--sparsity 0.64 \
--sparse_start_step 15 \
--base_seed 0 \
--rainfusion_type v2
```

The following is an example of a request for generating a video from images. After the service is started, you can send an HTTP request to generate a video. The `save_disk_path` parameter is optional. If this parameter is not set, the request result is directly returned. If this parameter is set, the generated video will be saved to the specified directory. The sample_guide_scale and sample_shift parameters are passed to the configuration of the corresponding task.

```shell
curl -X POST "http://localhost:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "task": "i2v-A14B",
           "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline'\''s intricate details and the refreshing atmosphere of the seaside.",
           "image": "examples/i2v_input.JPG",
           "sample_steps": 40,
           "base_seed": 0,
           "save_disk_path": "test_i2v.mp4",
           "size": "1280*720",
           "sample_guide_scale": [3.5, 3.5],
           "sample_shift": 5.0
         }'
```
