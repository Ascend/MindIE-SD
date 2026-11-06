#include "aclnn_eagle_quant_batch_matmul_swiglu_weight_nz.h"

#include <limits>
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

namespace {
bool ShapeSize(const op::Shape &shape, int64_t &size) {
    size = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const auto dim = shape.GetDim(i);
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        size *= dim;
    }
    return true;
}

bool GetWeightNzShape(const aclTensor *weight, op::Shape &physical) {
    const auto &logical = weight->GetViewShape();
    const auto rank = logical.GetDimNum();
    int64_t logicalSize = 0;
    if (rank < 2 || rank > 6 || !ShapeSize(logical, logicalSize)) {
        return false;
    }
    for (size_t i = 0; i + 2 < rank; ++i) {
        physical.AppendDim(logical.GetDim(i));
    }
    // The transpose attribute changes matmul semantics, not the stored matrix axes.
    const auto rows = logical.GetDim(rank - 2);
    const auto cols = logical.GetDim(rank - 1);
    physical.AppendDim(cols / 32 + (cols % 32 != 0));
    physical.AppendDim(rows / 16 + (rows % 16 != 0));
    physical.AppendDim(16);
    physical.AppendDim(32);
    int64_t expectedSize = 0;
    int64_t storageSize = 0;
    return ShapeSize(physical, expectedSize) && ShapeSize(weight->GetStorageShape(), storageSize) &&
        expectedSize == storageSize;
}

aclTensor *CreateTensorView(
    const aclTensor *input, const op::Shape &storageShape, op::Format format, aclOpExecutor *executor) {
    auto view = executor->CreateView(input, storageShape, input->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageShape(storageShape);
        view->SetStorageFormat(format);
        view->SetOriginalFormat(op::Format::FORMAT_ND);
        view->SetOriginalShape(input->GetViewShape());
        view->SetViewShape(input->GetViewShape());
    }
    return view;
}
} // namespace

namespace l0op {
OP_TYPE_REGISTER(EagleQuantBatchMatmulSwiglu);

aclnnStatus EagleQuantBatchMatmulSwiglu(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
    const aclTensor *pertokenScale, int64_t dtype, bool transposeX1, bool transposeX2, int64_t groupSize,
    aclTensor *out, aclOpExecutor *executor) {
    L0_DFX(EagleQuantBatchMatmulSwiglu, x1, x2, scale, pertokenScale, dtype, transposeX1, transposeX2, groupSize);
    // Keep the L0 binary configuration aligned with the OpDef opFile.
    [[maybe_unused]] static const bool configRegistered = []() {
        op::BinConfigJsonDict::UpdateConfigJsonPath(
            EagleQuantBatchMatmulSwigluOpTypeId(), "eagle_quant_batch_matmul_swiglu.json");
        return true;
    }();
    const aclTensor *absent = nullptr;
    const auto expectedShape = out->GetViewShape();
    auto ret = INFER_SHAPE(EagleQuantBatchMatmulSwiglu, OP_INPUT(x1, x2, scale, absent, absent, pertokenScale),
        OP_OUTPUT(out), OP_ATTR(dtype, transposeX1, transposeX2, groupSize));
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (out->GetViewShape() != expectedShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SwiGLU output shape does not match inferred [batch, M, N/2].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // The output view aliases caller memory; no intermediate result or ViewCopy is needed.
    return ADD_TO_LAUNCHER_LIST_AICORE(EagleQuantBatchMatmulSwiglu,
        OP_INPUT(x1, x2, scale, absent, absent, pertokenScale), OP_OUTPUT(out),
        OP_ATTR(dtype, transposeX1, transposeX2, groupSize));
}
} // namespace l0op

aclnnStatus aclnnEagleQuantBatchMatmulSwigluWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
    const aclTensor *scale, const aclTensor *offset, const aclTensor *bias, const aclTensor *pertokenScale,
    int64_t dtype, bool transposeX1, bool transposeX2, int64_t groupSize, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnEagleQuantBatchMatmulSwigluWeightNz,
        DFX_IN(x1, x2, scale, offset, bias, pertokenScale, dtype, transposeX1, transposeX2, groupSize), DFX_OUT(out));
    if (x1 == nullptr || x2 == nullptr || scale == nullptr || pertokenScale == nullptr || out == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (bias != nullptr || offset != nullptr || groupSize != 0 || x1->GetDataType() != op::DataType::DT_INT8 ||
        x2->GetDataType() != op::DataType::DT_INT8 || scale->GetDataType() != op::DataType::DT_FLOAT ||
        pertokenScale->GetDataType() != op::DataType::DT_FLOAT ||
        (out->GetDataType() != op::DataType::DT_BF16 && out->GetDataType() != op::DataType::DT_FLOAT16) ||
        dtype != static_cast<int64_t>(out->GetDataType()) || x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "WeightNz SwiGLU requires INT8 ND/NZ inputs, FP32 scales, BF16/FP16 output, "
            "group_size=0 and no bias/offset.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto *tensor : {x1, x2, scale, pertokenScale, static_cast<const aclTensor *>(out)}) {
        int64_t size = 0;
        if (!op::IsContiguous(tensor) || !ShapeSize(tensor->GetViewShape(), size) ||
            (tensor != x2 && tensor->GetStorageFormat() != op::Format::FORMAT_ND &&
                tensor->GetStorageFormat() != op::Format::FORMAT_NCL &&
                tensor->GetStorageFormat() != op::Format::FORMAT_NCHW &&
                tensor->GetStorageFormat() != op::Format::FORMAT_NCDHW)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "WeightNz SwiGLU requires nonempty contiguous tensors and ND activations.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    op::Shape weightShape;
    if (!GetWeightNzShape(x2, weightShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "INT8 NZ weight storage size does not match its logical shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto *exec = uniqueExecutor.get();
    auto *a = CreateTensorView(x1, x1->GetViewShape(), op::Format::FORMAT_ND, exec);
    auto *b = CreateTensorView(x2, weightShape, op::Format::FORMAT_FRACTAL_NZ, exec);
    auto *channel = CreateTensorView(scale, scale->GetViewShape(), op::Format::FORMAT_ND, exec);
    auto *token = CreateTensorView(pertokenScale, pertokenScale->GetViewShape(), op::Format::FORMAT_ND, exec);
    auto *output = CreateTensorView(out, out->GetViewShape(), op::Format::FORMAT_ND, exec);
    if (a == nullptr || b == nullptr || channel == nullptr || token == nullptr || output == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto ret = l0op::EagleQuantBatchMatmulSwiglu(
        a, b, channel, token, dtype, transposeX1, transposeX2, groupSize, output, exec);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    *workspaceSize = exec->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEagleQuantBatchMatmulSwigluWeightNz(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnEagleQuantBatchMatmulSwigluWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
