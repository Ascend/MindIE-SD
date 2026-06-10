/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*!
 * \file quant_flash_attn_def.cpp
 * \brief QuantFlashAttn算子定义（训练推理归一，仅非量化）
 *        输入数据类型仅支持FLOAT16和BFLOAT16。
 *        支持BSND/BNSD/TND三种layout，支持分页KV缓存（PA_ND/PA_Nz）。
 */

#include "register/op_def_registry.h"
#include <cstdlib>
#include <string>

namespace ops {
namespace {
bool IsSocEnabled(const char *socVersion) {
    const char *computeUnit = std::getenv("ASCEND_COMPUTE_UNIT");
    if (computeUnit == nullptr || computeUnit[0] == '\0') {
        return true;
    }

    std::string units(computeUnit);
    std::string soc(socVersion);
    size_t start = 0;
    while (start <= units.size()) {
        size_t end = units.find(';', start);
        std::string item = units.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (item == soc) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}
} // namespace

class QuantFlashAttn : public OpDef {
  public:
    explicit QuantFlashAttn(const char *name) : OpDef(name) {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("q_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("v_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("q_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("k_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("v_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("quant_block_size_qs").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_block_size_ks").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_block_size_vs").AttrType(OPTIONAL).Int(0);
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("win_left").AttrType(OPTIONAL).Int(-1);
        this->Attr("win_right").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_kv").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_out").AttrType(OPTIONAL).String("BSND");
        this->Attr("softmax_precision").AttrType(OPTIONAL).Int(0);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config_95;
        aicore_config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "quant_flash_attn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        if (IsSocEnabled("ascend950")) {
            this->AICore().AddConfig("ascend950", aicore_config_95);
        }
    }
};

OP_ADD(QuantFlashAttn);

} // namespace ops
