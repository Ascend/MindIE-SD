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

namespace ops {
class SigmoidMulDynamicMxQuant : public OpDef {
  public:
    explicit SigmoidMulDynamicMxQuant(const char *name) : OpDef(name) {
        for (const char *input : {"x", "gate"}) {
            Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
        }
        Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Output("mxscale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Attr("precision_mode").AttrType(OPTIONAL).Int(0);
        Attr("scale_alg").AttrType(OPTIONAL).Int(0);
        AICore().AddConfig("ascend950");
    }
};
OP_ADD(SigmoidMulDynamicMxQuant);
} // namespace ops
