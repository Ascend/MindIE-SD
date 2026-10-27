# Service Deployment Acceleration Feature

## Service Scheduling

Service deployment means starting an HTTP-based service (for example, a text-to-video service); users send URL requests to the backend, which completes end-to-end model inference.

Taking the Wan2.2 model as an example, the model can generate videos from text or images, and the generated video can be returned directly to the user or saved to a specified disk location. For example, the following command starts an HTTP service, where the model path is `./Wan2.2-I2V-A14B/`, FSDP is enabled for DiT and for T5 to reduce memory usage, the Ulysses degree is 8, and the VAE parallel strategy is used.
`server.py` is the script that starts the service. Install the service dependencies (see `examples/service/requirements.txt`, which includes ray, fastapi, uvicorn, pydantic, and Pillow) before starting the service.

See the model link [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) to ensure that the service can access Wan.

```shell
# Install the service dependencies first (ray/fastapi/uvicorn/pydantic/Pillow)
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

The following request is an example of generating a video from an image. After the service is started, users can send HTTP requests to generate videos. The `save_disk_path` parameter is optional: if it is not set, the request result is returned directly; if it is set, the generated video is saved to the specified directory. Pass the configuration of the corresponding task for sample_guide_scale and sample_shift

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
