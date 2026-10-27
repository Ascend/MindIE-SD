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

#include <string>
#include "register/op_def_registry.h"

namespace ops {
class EagleSplitNormRopeConcat : public OpDef {
  public:
    explicit EagleSplitNormRopeConcat(const char *name) : OpDef(name) {
        for (const char *input :
            {"qkv", "head_counts", "norm_query_weight", "norm_key_weight", "rope_cos", "rope_sin"}) {
            if (std::string(input) == "head_counts") {
                this->Input(input)
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64, ge::DT_INT64})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                    .ValueDepend(REQUIRED);
            } else {
                const bool isWeight =
                    std::string(input) == "norm_query_weight" || std::string(input) == "norm_key_weight";
                this->Input(input)
                    .ParamType(isWeight ? OPTIONAL : REQUIRED)
                    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            }
        }
        for (const char *output : {"query", "key", "value"}) {
            this->Output(output)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Attr("norm_type").AttrType(REQUIRED).Int();
        this->Attr("rope_type").AttrType(REQUIRED).Int();
        // CANN attributes use 0 for an omitted rotary_dim; PyTorch exposes None.
        this->Attr("rotary_dim").AttrType(OPTIONAL).Int(0);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5);
        this->Attr("output_bnsd").AttrType(OPTIONAL).Bool(true);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(EagleSplitNormRopeConcat);
} // namespace ops
