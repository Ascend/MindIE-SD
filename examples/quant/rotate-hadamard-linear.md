# MXFP4 RHL 加速特性

MXFP4 Rotate Hadamard Linear(RHL) 特性是指对多模态大模型线性层使用随机哈达玛矩阵旋转后再进行MXFP4量化。

以`HunyuanImage-3.0-Instruct`模型为例，展示如何使用`MXFP4 RHL`加速特性，进行模型优化。该模型的推理瓶颈在MoE部分的两个GMM，通过对GMM使能MXFP4 RHL特性，端到端加速比可达1.33x。

具体步骤：

1. 生成随机哈达玛矩阵 GWISE，对GMM部分的权重进行旋转再量化，这一步可离线进行，参考代码中的 AscendMxfp4RhlQuantizedFusedMoEMethod 。

2. 使用步骤1生成的 GWISE，对GMM部分的激活进行旋转再量化，这步需要推理时在线进行，参考代码中的 unquant_apply_mlp 。

```python
class AscendMxfp4RhlQuantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        self.GWISE = generate_hadamard_matrix(block_size=32, device='npu', transpose=True, seed=42) # 用于旋转权重
        self.GWISE_BF16 = self.GWISE.to(torch.bfloat16).contiguous()     # 用于旋转激活


    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        # Keep expert-aware loaders attached for later online weight updates.
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        replace_parameter(layer, "w13_weight", w13_data)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        replace_parameter(layer, "w2_weight", w2_data)


        enable_fused_mc2 = get_ascend_config().enable_fused_mc2
        if enable_fused_mc2:
            if _MEGA_MOE_SUPPORTED:
                layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
                layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]
            else:
                layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
                layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
                if enable_fused_mc2 == 1 and self.dynamic_eplb:
                    layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
                    layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]
                    del layer.w13_weight
                    del layer.w2_weight
                    torch.npu.empty_cache()
        else:
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
        # MXFP4 RHL start
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        scale_dtype = getattr(torch, "float8_e8m0fnu", None)
        w13_data = layer.w13_weight.data
        w13_data = apply_hadamard_weight(w13_data, self.GWISE)    # 旋转权重，转置放到初始化
        w13_data, w1_scale = torch_npu.npu_dynamic_mx_quant(w13_data, axis=1, dst_type=torch_npu.float4_e2m1fn_x2, block_size=32)

        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        layer.w1_scale = w1_scale

        w2_data = layer.w2_weight.data
        w2_data = apply_hadamard_weight(w2_data, self.GWISE)    # 旋转权重，转置放到初始化
        w2_data, w2_scale = torch_npu.npu_dynamic_mx_quant(w2_data, axis=1, dst_type=torch_npu.float4_e2m1fn_x2, block_size=32)
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)
        layer.w2_scale = w2_scale
        # MXFP4 RHL end

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = getattr(layer, "activation", "silu")
        if getattr(layer, "swigluoai_uninterleave", False):
            activation = "swigluoai_uninterleave"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        w13_weight_list = getattr(layer, "w13_weight_list", None)
        w2_weight_list = getattr(layer, "w2_weight_list", None)
        has_split_weight_lists = isinstance(w13_weight_list, list) and isinstance(w2_weight_list, list)
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            if _MEGA_MOE_SUPPORTED:
                w1 = w13_weight_list if isinstance(w13_weight_list, list) else [layer.w13_weight]
                w2 = w2_weight_list if isinstance(w2_weight_list, list) else [layer.w2_weight]
                w1_scale = None
                w2_scale = None
                w1_scale_bias = None
                w2_scale_bias = None
            else:
                if self.dynamic_eplb and not has_split_weight_lists:
                    logger.warning_once(
                        "FUSED_MC2 is enabled with dynamic EPLB, but unquantized MoE weights are not split into "
                        "tensor lists. This may cause accuracy issues or communication hangs."
                    )
                w1 = w13_weight_list if isinstance(w13_weight_list, list) else [layer.w13_weight]
                w2 = w2_weight_list if isinstance(w2_weight_list, list) else [layer.w2_weight]
                w1_scale = [torch.tensor([], dtype=torch.int64)]
                w2_scale = [torch.tensor([], dtype=torch.int64)]
                w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
                w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight # MXFP4 RHL
            w1_scale = getattr(layer, 'w1_scale', None)                                     # MXFP4 RHL
            w2 = w2_weight_list if isinstance(w2_weight_list, list) else layer.w2_weight    # MXFP4 RHL
            w2_scale = getattr(layer, 'w2_scale', None)                                     # MXFP4 RHL
            w1_scale_bias = None
            w2_scale_bias = None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
                GWISE=self.GWISE_BF16,  # MXFP4 RHL
            )
        )
        return final_hidden_states

def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | MoEActivation | None = None,
    activation_situ_beta: float | None = None,
    activation_situ_linear_beta: float | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    lora_context=None,
    expanded_row_idx: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    GWISE: torch.Tensor | None = None,
    dynamic_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    hqx, hqx_scale = torch_npu.npu_rotate_quant(hidden_states, GWISE.to(torch.bfloat16), alpha=None, dst_dtype =torch_npu.float4_e2m1fn_x2, axis = -1)  # 旋转输入 后 量化

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if fp4_dtype is not None:
        hqx = hqx.view(fp4_dtype)
        if hqx_scale is not None:
            hqx_scale = hqx_scale.view(scale_dtype)
        w1qx = w1qx.view(fp4_dtype)
        if w1qx_scale is not None:
            w1qx_scale = w1qx_scale.view(scale_dtype)
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hqx],
        weight=[w1qx],
        scale=[w1qx_scale],
        per_token_scale=[hqx_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
        scale_dtype=scale_dtype,
        per_token_scale_dtype=scale_dtype,
    )[0]

    gate_up_out = torch_npu.npu_swiglu(gate_up_out)
    w1qx, w1qx_scale = w2, w2_scale

    hqx, hqx_scale = torch_npu.npu_rotate_quant(gate_up_out, GWISE.to(torch.bfloat16), alpha=None, dst_dtype =torch_npu.float4_e2m1fn_x2, axis = -1)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if fp4_dtype is not None:
        hqx = hqx.view(fp4_dtype)
        if hqx_scale is not None:
            hqx_scale = hqx_scale.view(scale_dtype)
        w1qx = w1qx.view(fp4_dtype)
        if w1qx_scale is not None:
            w1qx_scale = w1qx_scale.view(scale_dtype)

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hqx],
        weight=[w1qx],
        scale=[w1qx_scale],
        per_token_scale=[hqx_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
        scale_dtype=scale_dtype,
        per_token_scale_dtype=scale_dtype,
    )[0]
    return hidden_states, None
```
