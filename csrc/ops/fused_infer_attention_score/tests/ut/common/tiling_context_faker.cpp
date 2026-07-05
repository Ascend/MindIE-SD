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

#include "tiling_context_faker.h"

namespace gert {

TilingContextFaker &TilingContextFaker::SetOpType(const std::string opType) {
    OpTilingContextBuilder::OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

TilingContextFaker &TilingContextFaker::NodeIoNum(size_t inputNum, size_t outputNum) {
    OpTilingContextBuilder::IONum(inputNum, outputNum);
    return *this;
}

TilingContextFaker &TilingContextFaker::IrInstanceNum(
    const std::vector<uint32_t> &inputInstanceNum, const std::vector<uint32_t> &outputInstanceNum) {
    OpTilingContextBuilder::IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

TilingContextFaker &TilingContextFaker::NodeInputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat) {
    return *this;
}

TilingContextFaker &TilingContextFaker::NodeOutputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat) {
    return *this;
}

TilingContextFaker &TilingContextFaker::InputTensors(const std::vector<Tensor *> &inputTensors) {
    OpTilingContextBuilder::InputTensors(inputTensors);
    return *this;
}

TilingContextFaker &TilingContextFaker::OutputTensors(const std::vector<Tensor *> &outputTensors) {
    OpTilingContextBuilder::OutputTensors(outputTensors);
    return *this;
}

TilingContextFaker &TilingContextFaker::CompileInfo(const void *compileInfo) {
    OpTilingContextBuilder::CompileInfo(compileInfo);
    return *this;
}

TilingContextFaker &TilingContextFaker::PlatformInfo(const void *platformInfo) {
    OpTilingContextBuilder::PlatformInfo(platformInfo);
    return *this;
}

TilingContextFaker &TilingContextFaker::DeterministicInfo(int32_t deterministicInfo) {
    OpTilingContextBuilder::Deterministic(deterministicInfo);
    return *this;
}

TilingContextFaker &TilingContextFaker::TilingData(const void *tilingData) {
    OpTilingContextBuilder::TilingData(static_cast<const gert::TilingData *>(tilingData));
    return *this;
}

TilingContextFaker &TilingContextFaker::Workspace(const ContinuousVector *workspace) {
    OpTilingContextBuilder::Workspace(workspace);
    return *this;
}

ContextHolder<TilingContext> TilingContextFaker::Build() { return OpTilingContextBuilder::Build(); }

} // namespace gert
