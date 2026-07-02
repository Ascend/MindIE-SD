# Serving Acceleration

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:16:05.158Z pushedAt=2026-06-09T02:19:02.979Z -->

## Serving Scheduling

Serving refers to launching an HTTP-based service (e.g., a text-to-video generation service), where users send requests to the backend via a URL to perform end-to-end model inference.

For example, the Wan2.2 model generates videos from text or images. The resulting video can be returned directly to the user or saved to a specified drive location. You can start an HTTP service with the following commands. The model path is `./Wan2.2-I2V-A14B/`.
Enable FSDP for both DiT and T5 to reduce memory usage. Set Ulysses parallel size to 8 and apply VAE parallelism.
`server.py` is the serving startup script. Start the service and install the required dependencies first.

For the reference model link [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2), ensure that the service can access wan.

```shell
pip install fastapi
pip install ray
pip install uvicorn

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

The following request is an example of generating a video from an image. After starting the service, users can generate videos by sending HTTP requests. The `save_disk_path` parameter is optional. If this parameter is not set, the request result will be returned directly; if it is set, the generated video will be saved to the specified directory. Pass `sample_guide_scale` and `sample_shift` to the corresponding task configuration.

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
