# 服务化加速特性

## 服务化调度

服务化是指启动一个基于Http的服务（例如文本生成视频服务），用户通过URL请求后端，完成模型的端到端推理。

以Wan2.2模型为例，该模型可以根据文字或者图片生成视频，生成的视频可以直接返回给用户，也可以保存到指定的磁盘位置。例如可以通过以下命令启动一个Http服务，模型路径为`./Wan2.2-I2V-A14B/`，dit使能fsdp，t5使能fsdp用以降低显存占用，ulysses并行数为8，使用vae并行策略。
`server.py`是启动服务化的脚本，需要先启动服务，并安装服务化所需依赖。

参考模型链接[Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)确保服务化可以访问到wan

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

下面请求为图片生成视频的请求示例。在启动好服务后，用户可以通过发送Http请求来生成视频。其中`save_disk_path`参数是可选的，如果未设置该参数，将直接返回请求结果；如果设置了该参数，生成的视频将会保存到指定的目录下。sample_guide_scale和sample_shift传入对应任务的配置

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
