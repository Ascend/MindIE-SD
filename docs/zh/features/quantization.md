# 量化

## Linear量化

- 量化：一般是对模型的权重（weight）和激活值（activation）进行低比特处理，让最终生成的网络模型更加轻量化，从而达到节省网络模型存储空间、降低传输时延、提高计算效率，提高性能的目标。

    量化根据是否需要重训练，分为训练后量化（Post-Training Quantization, PTQ）和量化感知训练（Quantization-Aware Training，QAT）。

- 本章节以PTQ量化为主，主要分为以下三种类型：

  - 动态量化：仅离线量化权重，在推理时动态计算激活值的量化因子。
  - 静态量化：权重和激活值都是离线量化。
  - Time-Aware量化：根据时间维度动态调整量化策略。

  下图展示了INT8量化示例，将FP32（32位浮点数）映射到INT8（8位整数）。其中[-max(|xf|), max(|xf|)]是量化前浮点数值的数据范围，[-128,127]是量化后的数据范围。

  ![](../figures/INT8-image.png)

- 限制与约束：仅Atlas 800I A2 推理服务器支持此特性。
- 优化流程：

    - MindIE SD量化需要先使用工具导出权重，再使用推理框架的接口进行量化推理。
    - 安装大模型压缩工具，详情请参见[链接](https://gitee.com/ascend/msit/tree/master/msmodelslim)。
    - 对于包含激活值的量化算法，参考对应的[示例](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/multimodal_sd/README.md)和[接口](https://gitee.com/ascend/msit/tree/master/msmodelslim/docs/Python-API%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/%E5%A4%9A%E6%A8%A1%E6%80%81%E7%94%9F%E6%88%90%E7%BB%9F%E4%B8%80%E9%87%8F%E5%8C%96%E6%8E%A5%E5%8F%A3)，对量化权重进行导出。对于仅权重量化，参考对应的[示例](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/w8a16%E7%B2%BE%E5%BA%A6%E8%B0%83%E4%BC%98%E7%AD%96%E7%95%A5.md)和[接口](https://gitee.com/ascend/msit/tree/master/msmodelslim/docs/Python-API%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%8E%A5%E5%8F%A3/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96%E6%8E%A5%E5%8F%A3/PyTorch)，对量化权重进行导出。
      >[!NOTE]说明
      >- 对于单卡进行量化权重导出，使用工具默认量化权重和描述符命名即可。
      >- 对于多卡并行量化，推理框架限制命名规则。
      >- 量化权重命名规则为quant_model_weight_{quant_algo.lower()}_{rank}.safetensors。
      >- 描述符命名规则为quant_model_description_{quant_algo.lower()}_{rank}.json。

    - 使用quantize接口，对浮点模型进行量化转换，该接口会处理量化权重和修改计算图，示例如下所示。

        ```python
        from mindiesd import quantize
        model = from_pretrain()
        model = quantize(model, "步骤2导出的quant json path")
        ```

      >[!NOTE]说明
      >- 模型自行加载原始权重，并完成实例初始化，quantize由插件提供，在接口中对相应层进行量化转换。
      >- 模型可以选择在quantize转换完后再使用to npu。
    - 如果使用时间步量化，在quantize中还需要传入TimestepPolicyConfig，量化转换后还需要使用TimestepManager在模型中设置时间步信息，示例如下：

        ```python
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # -----------新增代码-----------
                from mindiesd import TimestepManager
                TimestepManager.set_timestep_idx(i)
                # -----------新增代码-----------
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
        ```

## FA量化

待补充...
