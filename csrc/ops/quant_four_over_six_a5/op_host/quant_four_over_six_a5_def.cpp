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
 * \file quant_four_over_six_a5_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_BLOCK_SIZE = 32;
static constexpr int32_t DEFAULT_DST_TYPE = 40;
// T9b: 仅保留 46 自适应路径——默认值按 46 语义（scale_alg=2, dst_type_max=4.0）
static constexpr int32_t DEFAULT_SCALE_ALG = 2;
static constexpr float DEFAULT_DST_TYPE_MAX = 4.0;
static constexpr uint32_t ATTR_VERSION = 2;
class QuantFourOverSixA5 : public OpDef {
public:
    explicit QuantFourOverSixA5(const char* name) : OpDef(name)
    {
        // T9b: 仅保留 46 自适应组合（bf16 -> FLOAT4_E2M1, E8M0 mxscale）；其余 7 组合
        // （fp16 输入、e1m2/fp8 输出）已随非 46 路径清理删除。
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT4_E2M1})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("blocksize").AttrType(OPTIONAL).Int(DEFAULT_BLOCK_SIZE);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(DEFAULT_SCALE_ALG);
        this->Attr("dst_type_max").AttrType(OPTIONAL).Version(ATTR_VERSION).Float(DEFAULT_DST_TYPE_MAX);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(QuantFourOverSixA5);
} // namespace ops