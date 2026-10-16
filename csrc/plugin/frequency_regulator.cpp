/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <limits>
#include <memory>
#include <string_view>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

#include "frequency_regulator.h"
#include "pytorch_npu_helper.h"

namespace {
constexpr std::string_view FREQUENCY_REGULATOR_OP_NAME = "aclnnFrequencyRegulator";
constexpr int64_t FREQUENCY_REGULATOR_OUTPUT_NUMEL = 1;

using FrequencyRegulatorGetWorkspaceSizeFunc = int (*)(uint32_t, AclTensor *, uint64_t *, AclOpExecutor **);
using FrequencyRegulatorFunc = int (*)(FunctionPtr<>, uint64_t, AclOpExecutor *, aclrtStream);

struct AclTensorDeleter {
    void operator()(AclTensor *tensor) const {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
};

struct AclWorkspaceDeleter {
    void operator()(void *workspace) const {
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
    }
};

struct HugeMemSession {
    void *unInitMemAddr = nullptr;

    HugeMemSession(void *initMemAddr, void *unInitMemAddr) : unInitMemAddr(unInitMemAddr) {
        InitHugeMemCustom(initMemAddr);
    }

    ~HugeMemSession() { UnInitHugeMem(unInitMemAddr); }
};

struct HugeMemReleaseGuard {
    void *releaseMemAddr = nullptr;

    explicit HugeMemReleaseGuard(void *releaseMemAddr) : releaseMemAddr(releaseMemAddr) {}

    ~HugeMemReleaseGuard() { ReleaseHugeMemResource(releaseMemAddr); }
};

std::unique_ptr<AclTensor, AclTensorDeleter> CreateFrequencyRegulatorOutAclTensor(const at::Tensor &out) {
    TORCH_CHECK(out.scalar_type() == c10::ScalarType::Int, "frequency_regulator out must be INT32-backed, but got ",
        out.scalar_type());
    TORCH_CHECK(out.dim() == 1 && out.numel() == FREQUENCY_REGULATOR_OUTPUT_NUMEL,
        "frequency_regulator out must be a one-dimensional single-element tensor, but got dim=", out.dim(),
        " and numel=", out.numel());

    auto aclOut = aclCreateTensor(out.sizes().data(), out.sizes().size(), ACL_UINT32, nullptr, 0, ACL_FORMAT_ND,
        out.sizes().data(), out.sizes().size(), const_cast<void *>(out.storage().data()));
    TORCH_CHECK(aclOut != nullptr, "aclCreateTensor failed for frequency_regulator out");
    return std::unique_ptr<AclTensor, AclTensorDeleter>(aclOut);
}

uint32_t CopyFrequencyRegulatorStatusToHost(const at::Tensor &out) {
    uint32_t status = 0;
    auto ret = aclrtMemcpy(
        &status, sizeof(status), const_cast<void *>(out.storage().data()), sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpy failed for frequency_regulator status, ret=", ret,
        ", detail:", aclGetRecentErrMsg());
    return status;
}

at::Tensor CreateFrequencyRegulatorResultTensor(uint32_t status) {
    auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(c10::ScalarType::Long);
    at::Tensor result = at_npu::native::empty_with_format({FREQUENCY_REGULATOR_OUTPUT_NUMEL}, options, ACL_FORMAT_ND);
    int64_t statusValue = static_cast<int64_t>(status);
    auto ret = aclrtMemcpy(const_cast<void *>(result.storage().data()), sizeof(statusValue), &statusValue,
        sizeof(statusValue), ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpy failed for frequency_regulator result, ret=", ret,
        ", detail:", aclGetRecentErrMsg());
    return result;
}

std::unique_ptr<void, AclWorkspaceDeleter> AllocateFrequencyRegulatorWorkspace(uint64_t workspaceSize) {
    if (workspaceSize == 0) {
        return std::unique_ptr<void, AclWorkspaceDeleter>(nullptr);
    }

    void *workspaceAddr = nullptr;
    auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMalloc failed for frequency_regulator workspace, ret=", ret,
        ", workspaceSize=", workspaceSize);
    return std::unique_ptr<void, AclWorkspaceDeleter>(workspaceAddr);
}
} // namespace

at::Tensor frequency_regulator_impl_npu(int64_t freq) {
    TORCH_CHECK(freq >= 0 && freq <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "freq must be in range [0, UINT32_MAX], but got ", freq);

    auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(c10::ScalarType::Int);
    at::Tensor aclOutStorage =
        at_npu::native::empty_with_format({FREQUENCY_REGULATOR_OUTPUT_NUMEL}, options, ACL_FORMAT_ND);

    auto aclOut = CreateFrequencyRegulatorOutAclTensor(aclOutStorage);
    auto workspaceSizeApiStr = GetWorkspaceSizeApiName<FREQUENCY_REGULATOR_OP_NAME>();
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspaceSizeApiStr.c_str());
    static const auto opApiFuncAddr = GetOpApiFuncAddr(FREQUENCY_REGULATOR_OP_NAME.data());
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");

    ValidateApiAddresses(getWorkspaceSizeFuncAddr, opApiFuncAddr, FREQUENCY_REGULATOR_OP_NAME,
        std::string_view(workspaceSizeApiStr.c_str(), workspaceSizeApiStr.length()));

    HugeMemSession hugeMemSession(initMemAddr, unInitMemAddr);
    uint64_t workspaceSize = 0;
    AclOpExecutor *executor = nullptr;
    auto getWorkspaceSizeFunc =
        FunctionPointerConverter<FrequencyRegulatorGetWorkspaceSizeFunc, void *>::Convert(getWorkspaceSizeFuncAddr);
    auto workspaceStatus = getWorkspaceSizeFunc(static_cast<uint32_t>(freq), aclOut.get(), &workspaceSize, &executor);
    TORCH_CHECK(workspaceStatus == 0, "call ", workspaceSizeApiStr, " failed, detail:", aclGetRecentErrMsg());

    auto workspace = AllocateFrequencyRegulatorWorkspace(workspaceSize);
    void *workspaceAddr = workspace.get();
    auto aclStreamObj = c10_npu::getCurrentNPUStream().stream(false);
    auto aclCall = [workspaceAddr, workspaceSize, aclStreamObj, executor]() -> int {
        HugeMemReleaseGuard releaseGuard(releaseMemAddr);
        auto opApiFunc = FunctionPointerConverter<FrequencyRegulatorFunc, void *>::Convert(opApiFuncAddr);
        auto apiRet = opApiFunc(workspaceAddr, workspaceSize, executor, aclStreamObj);
        TORCH_CHECK(apiRet == 0, "call ", FREQUENCY_REGULATOR_OP_NAME.data(), " failed, detail:", aclGetRecentErrMsg());
        auto syncRet = aclrtSynchronizeStream(aclStreamObj);
        TORCH_CHECK(syncRet == ACL_SUCCESS, "aclrtSynchronizeStream failed after ", FREQUENCY_REGULATOR_OP_NAME.data(),
            ", ret=", syncRet, ", detail:", aclGetRecentErrMsg());
        return apiRet;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(FREQUENCY_REGULATOR_OP_NAME.data());
    cmd.SetCustomHandler(aclCall);
    cmd.Run();

    auto status = CopyFrequencyRegulatorStatusToHost(aclOutStorage);
    return CreateFrequencyRegulatorResultTensor(status);
}
