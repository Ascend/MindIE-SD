/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

class EagleBlockSparseAttention : public OpDef {
public:
    explicit EagleBlockSparseAttention(const char *name)
        : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("blockSparseMask").ParamType(OPTIONAL).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND});
        this->Input("attenMask").ParamType(OPTIONAL).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND});
        this->Input("blockShape").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengths").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengthsKv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("blockTable").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("qDequantScaleOptional")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("kDequantScaleOptional")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("vDequantScaleOptional")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("pQuantScaleOptional").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmaxLse").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        this->Attr("qInputLayout").AttrType(OPTIONAL).String("TND");
        this->Attr("kvInputLayout").AttrType(OPTIONAL).String("TND");
        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(1);
        this->Attr("maskType").AttrType(OPTIONAL).Int(0);
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(0);
        this->Attr("blockSize").AttrType(OPTIONAL).Int(0);
        this->Attr("preTokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("nextTokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("softmaxLseFlag").AttrType(OPTIONAL).Int(0);
        this->Attr("quantMode").AttrType(OPTIONAL).Int(0);
        this->Attr("dstTypeMax").AttrType(OPTIONAL).Float(0.0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(EagleBlockSparseAttention);

} // namespace ops
