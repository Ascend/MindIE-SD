# Lightweight Algorithm Acceleration Features

## Linear Quantization

- Quantization usually applies lower-bit representations to model weights and activations so that the final model becomes lighter, reducing storage footprint and transfer latency while improving compute efficiency and inference performance.

  Depending on whether retraining is required, quantization can be divided into post-training quantization (PTQ) and quantization-aware training (QAT).

- This section focuses on PTQ and covers three common modes:

  - Dynamic quantization: only weights are quantized offline, while activation quantization factors are computed dynamically during inference.
  - Static quantization: both weights and activations are quantized offline.
  - Time-aware quantization: the quantization strategy is adjusted dynamically across the timestep dimension.

  The figure below shows an INT8 quantization example in which FP32 values are mapped to INT8 values. `[-max(|xf|), max(|xf|)]` is the floating-point range before quantization, and `[-128, 127]` is the value range after quantization.

  ![](../../zh/figures/INT8-image.png)

- Constraints: this feature is currently supported only on Atlas 800I A2 inference servers.
- Optimization workflow:

  - MindIE SD quantization first exports the weights with a tool and then performs quantized inference through the inference framework interface.
  - Install the large-model compression tool. See the [msmodelslim repository](https://gitee.com/ascend/msit/tree/master/msmodelslim).
  - For algorithms that include activation quantization, refer to the relevant [example](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/multimodal_sd/README.md) and [API reference](https://gitee.com/ascend/msit/tree/master/msmodelslim/docs/Python-API%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/%E5%A4%9A%E6%A8%A1%E6%80%81%E7%94%9F%E6%88%90%E7%BB%9F%E4%B8%80%E9%87%8F%E5%8C%96%E6%8E%A5%E5%8F%A3) to export the quantized weights. For weight-only quantization, refer to the relevant [example](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/w8a16%E7%B2%BE%E5%BA%A6%E8%B0%83%E4%BC%98%E7%AD%96%E7%95%A5.md) and [PyTorch API reference](https://gitee.com/ascend/msit/tree/master/msmodelslim/docs/Python-API%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%8E%A5%E5%8F%A3/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96%E6%8E%A5%E5%8F%A3/PyTorch) to export the quantized weights.

    > **Note**
    > - For single-card quantized weight export, the tool’s default naming for weights and descriptors can be used directly.
    > - For multi-card parallel quantization, the inference framework imposes naming rules.
    > - Quantized weight files use the pattern `quant_model_weight_{quant_algo.lower()}_{rank}.safetensors`.
    > - Descriptor files use the pattern `quant_model_description_{quant_algo.lower()}_{rank}.json`.

  - Use the `quantize` interface to convert the floating-point model. This interface handles quantized weights and graph rewriting:

    ```python
    from mindiesd import quantize
    model = from_pretrain()
    model = quantize(model, "quant json path exported in step 2")
    ```

    > **Note**
    > - The model loads its original weights and completes instance initialization itself. `quantize` is provided by the plugin and performs the layer-level quantized conversion inside the interface.
    > - The model can be moved to NPU after `quantize` completes if desired.

  - If timestep-aware quantization is used, pass `TimestepPolicyConfig` to `quantize`. After quantized conversion, use `TimestepManager` to set timestep information in the model:

    ```python
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            # -----------new code-----------
            from mindiesd import TimestepManager
            TimestepManager.set_timestep_idx(i)
            # -----------new code-----------
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
    ```

---

## FA sparsity

RainFusion is a sparse method that takes advantage of the temporal-spatial similarity inherent in videos. It adaptively classifies attention patterns and performs sparse attention computation, reducing compute cost and improving inference speed.

Its core ideas are:

- Offline feature mining: attention in DiT diffusion generation contains temporal and spatial redundancy. Attention heads can be divided into three sparse types, corresponding to three static attention masks.

  ![](../../zh/figures/RainFusion-image-1.png)

  - Spatial head: attends to all tokens in the current or key frame, focusing on spatial consistency within a single frame.
  - Temporal head: attends to corresponding local regions across multiple frames, focusing on periodic sparsity in long sequences.
  - Textural head: attends to high-level semantic information and input-related details, focusing on semantic consistency.

- Online classification: a lightweight online decision module called ARM is introduced to classify the sparse type of each head at runtime.

  ![](../../zh/figures/RainFusion-image-2.png)
