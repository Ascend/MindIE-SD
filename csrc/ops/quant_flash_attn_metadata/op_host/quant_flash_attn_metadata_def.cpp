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

class QuantFlashAttnMetadata : public OpDef {
  public:
    explicit QuantFlashAttnMetadata(const char *name) : OpDef(name) {
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
        this->Output("metadata")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("batch_size").AttrType(OPTIONAL).Int(0);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_kv").AttrType(OPTIONAL).Int(-1);
        this->Attr("num_heads_q").AttrType(REQUIRED).Int(0);
        this->Attr("num_heads_kv").AttrType(REQUIRED).Int(0);
        this->Attr("head_dim").AttrType(REQUIRED).Int(0);
        this->Attr("q_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("k_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("v_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("q_dtype").AttrType(REQUIRED).Int(0);
        this->Attr("k_dtype").AttrType(REQUIRED).Int(0);
        this->Attr("v_dtype").AttrType(REQUIRED).Int(0);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("win_left").AttrType(OPTIONAL).Int(-1);
        this->Attr("win_right").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_out").AttrType(OPTIONAL).String("BSND");
        this->Attr("custom_soc_version").AttrType(REQUIRED).String("");
        this->Attr("aic_core_num").AttrType(REQUIRED).Int(36);
        this->Attr("aiv_core_num").AttrType(REQUIRED).Int(72);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AICPU")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        if (IsSocEnabled("ascend950")) {
            this->AICore().AddConfig("ascend950", aicore_config);
        }
    }
};

OP_ADD(QuantFlashAttnMetadata);

} // namespace ops
