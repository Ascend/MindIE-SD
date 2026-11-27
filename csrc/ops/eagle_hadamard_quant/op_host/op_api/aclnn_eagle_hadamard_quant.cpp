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

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "aclnn_eagle_hadamard_quant.h"
#include "eagle_hadamard_quant.h"
#include "../eagle_hadamard_quant_params.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
// The ACL descriptors can use different base pointers into the same allocation.
// Compare device byte addresses, including the view offset and strided extent.
struct ByteRange {
    uintptr_t begin;
    uintptr_t end;
};

bool DescribeView(const aclTensor *tensor, uintptr_t elementBytes, bool writable, ByteRange &range) {
    const auto &shape = tensor->GetViewShape();
    const auto &strides = tensor->GetViewStrides();
    std::array<std::pair<uintptr_t, uintptr_t>, 5> axes{};
    size_t count = 0;
    const auto base = reinterpret_cast<uintptr_t>(tensor->GetStorageAddr());
    const auto offset = tensor->GetViewOffset();
    if (base == 0 || offset < 0 || shape.GetDimNum() > axes.size() ||
        static_cast<uintptr_t>(offset) > (UINTPTR_MAX - base) / elementBytes) {
        return false;
    }
    range.begin = base + static_cast<uintptr_t>(offset) * elementBytes;
    for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
        if (shape.GetDim(axis) > 1) {
            const auto stride = strides[axis];
            if (stride < 0) {
                return false;
            }
            axes[count++] = {static_cast<uintptr_t>(stride), static_cast<uintptr_t>(shape.GetDim(axis) - 1)};
        }
    }
    std::sort(axes.begin(), axes.begin() + count);
    uintptr_t span = 1;
    for (size_t i = 0; i < count; ++i) {
        const auto [stride, steps] = axes[i];
        if ((writable && stride < span) || stride > (UINTPTR_MAX - span) / steps) {
            return false;
        }
        span += stride * steps;
    }
    if (span > (UINTPTR_MAX - range.begin) / elementBytes) {
        return false;
    }
    range.end = range.begin + span * elementBytes;
    return true;
}

bool Intersects(const ByteRange &left, const ByteRange &right) {
    return left.begin < right.end && right.begin < left.end;
}

aclTensor *CreateNdView(const aclTensor *input, aclOpExecutor *executor) {
    const auto &shape = input->GetViewShape();
    auto *view = executor->CreateView(input, shape, input->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageShape(shape);
        view->SetStorageFormat(op::Format::FORMAT_ND);
        view->SetOriginalFormat(op::Format::FORMAT_ND);
        view->SetOriginalShape(shape);
        view->SetViewShape(shape);
    }
    return view;
}

bool CopyOutputBytes(const aclTensor *source, const aclTensor *destination, aclOpExecutor *executor) {
    // Both quantized outputs use one byte per stored element. ViewCopy does not
    // support E8M0 on all backends, so preserve the bytes and strides as UINT8.
    auto *sourceBytes = executor->CreateView(
        source, source->GetViewShape(), source->GetStorageShape(), source->GetViewStrides(), source->GetViewOffset());
    auto *destinationBytes = executor->CreateView(destination, destination->GetViewShape(),
        destination->GetStorageShape(), destination->GetViewStrides(), destination->GetViewOffset());
    if (sourceBytes == nullptr || destinationBytes == nullptr) {
        return false;
    }
    sourceBytes->SetDataType(op::DataType::DT_UINT8);
    destinationBytes->SetDataType(op::DataType::DT_UINT8);
    return l0op::ViewCopy(sourceBytes, destinationBytes, executor) != nullptr;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *signH, int64_t blockSize, const char *dstType,
    bool isSignHNz, const aclTensor *y, const aclTensor *scale) {
    CHECK_COND(x != nullptr && signH != nullptr && y != nullptr && scale != nullptr && dstType != nullptr,
        ACLNN_ERR_PARAM_NULLPTR, "x, sign_h, dst_type, y and scale must be provided");
    CHECK_COND(x->GetDataType() == op::DataType::DT_BF16 && signH->GetDataType() == op::DataType::DT_BF16,
        ACLNN_ERR_PARAM_INVALID, "x and sign_h must be BF16");
    const auto p =
        EagleHadamardQuant::CheckShape(x->GetViewShape(), signH->GetViewShape(), blockSize, dstType, isSignHNz);
    CHECK_COND(p.error == nullptr, ACLNN_ERR_PARAM_INVALID, "%s", p.error);
    const auto yType = p.format == 0 ? op::DataType::DT_UINT8 : op::DataType::DT_FLOAT8_E4M3FN;
    CHECK_COND(y->GetDataType() == yType && scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0,
        ACLNN_ERR_PARAM_INVALID, "y dtype must match dst_type and scale must be FLOAT8_E8M0");
    const auto m = x->GetViewShape().GetDim(0);
    CHECK_COND(y->GetViewShape() == op::Shape({m, p.outputColumns}) &&
            scale->GetViewShape() == op::Shape({m, p.scalePairs, 2}),
        ACLNN_ERR_PARAM_INVALID, "y or scale shape does not match x and dst_type");
    for (const auto *tensor : {x, signH, y, scale}) {
        CHECK_COND(
            tensor->GetStorageFormat() < op::Format::FORMAT_END && !op::IsPrivateFormat(tensor->GetStorageFormat()),
            ACLNN_ERR_PARAM_INVALID, "x, sign_h, y and scale must use non-private storage formats");
    }
    if (x->IsEmpty()) {
        return ACLNN_SUCCESS;
    }
    // Validate the current addresses before executor-cache lookup.
    ByteRange xRange{}, rotationRange{}, yRange{}, scaleRange{};
    CHECK_COND(DescribeView(x, sizeof(uint16_t), false, xRange) &&
            DescribeView(signH, sizeof(uint16_t), false, rotationRange) && DescribeView(y, 1, true, yRange) &&
            DescribeView(scale, 1, true, scaleRange),
        ACLNN_ERR_PARAM_INVALID, "invalid tensor byte range or overlapping output strides");
    CHECK_COND(!Intersects(yRange, scaleRange) && !Intersects(yRange, xRange) && !Intersects(yRange, rotationRange) &&
            !Intersects(scaleRange, xRange) && !Intersects(scaleRange, rotationRange),
        ACLNN_ERR_PARAM_INVALID, "each output byte range must be disjoint from both inputs and the other output");
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnEagleHadamardQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *signH, int64_t blockSize,
    const char *dstType, bool isSignHNz, const aclTensor *y, const aclTensor *scale, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        "workspaceSize and executor must be provided");
    const auto status = CheckParams(x, signH, blockSize, dstType, isSignHNz, y, scale);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    L2_DFX_PHASE_1(aclnnEagleHadamardQuant, DFX_IN(x, signH, blockSize, dstType, isSignHNz), DFX_OUT(y, scale));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const auto *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    const auto *signHContiguous = l0op::Contiguous(signH, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr && signHContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *xNd = CreateNdView(xContiguous, uniqueExecutor.get());
    auto *signHNd = CreateNdView(signHContiguous, uniqueExecutor.get());
    const bool yContiguous = op::IsContiguous(y);
    const bool scaleContiguous = op::IsContiguous(scale);
    auto *yResult = yContiguous
        ? CreateNdView(y, uniqueExecutor.get())
        : uniqueExecutor->AllocTensor(y->GetViewShape(), y->GetDataType(), op::Format::FORMAT_ND);
    auto *scaleResult = scaleContiguous
        ? CreateNdView(scale, uniqueExecutor.get())
        : uniqueExecutor->AllocTensor(scale->GetViewShape(), scale->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(
        xNd != nullptr && signHNd != nullptr && yResult != nullptr && scaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto launch = l0op::EagleHadamardQuant(
        xNd, signHNd, blockSize, dstType, isSignHNz, yResult, scaleResult, uniqueExecutor.get());
    CHECK_RET(launch == ACLNN_SUCCESS, launch);
    if (!yContiguous) {
        CHECK_RET(CopyOutputBytes(yResult, y, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    }
    if (!scaleContiguous) {
        CHECK_RET(CopyOutputBytes(scaleResult, scale, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEagleHadamardQuant(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnEagleHadamardQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
