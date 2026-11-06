#ifndef ACLNN_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_WEIGHT_NZ_H_
#define ACLNN_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_WEIGHT_NZ_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnEagleQuantBatchMatmulSwigluWeightNzGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *scale, const aclTensor *offset, const aclTensor *bias,
    const aclTensor *pertokenScale, int64_t dtype, bool transposeX1, bool transposeX2, int64_t groupSize,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnEagleQuantBatchMatmulSwigluWeightNz(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
