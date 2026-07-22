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

#include "tiling_case_executor.h"
#include <unordered_set>
#include <stdexcept>
#ifdef FIA_ARCH35_STANDALONE_UT
#else
#include <gtest/gtest.h>
#endif
#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifdef FIA_ARCH35_STANDALONE_UT
#define FIA_UT_ASSERT_EQ(actual, expected) \
    do { \
        auto actualValue = (actual); \
        auto expectedValue = (expected); \
        if (!(actualValue == expectedValue)) { \
            std::ostringstream oss; \
            oss << "ASSERT_EQ failed at " << __FILE__ << ":" << __LINE__ << ", actual=" << actualValue \
                << ", expected=" << expectedValue; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)
#define FIA_UT_ASSERT_LE(actual, expected) \
    do { \
        auto actualValue = (actual); \
        auto expectedValue = (expected); \
        if (!(actualValue <= expectedValue)) { \
            std::ostringstream oss; \
            oss << "ASSERT_LE failed at " << __FILE__ << ":" << __LINE__ << ", actual=" << actualValue \
                << ", expected=" << expectedValue; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)
#define FIA_UT_EXPECT_EQ(actual, expected) FIA_UT_ASSERT_EQ(actual, expected)
#define FIA_UT_REQUIRE_NOT_NULL(value, name) \
    do { \
        if ((value) == nullptr) { \
            throw std::runtime_error(std::string(name) + " is null"); \
        } \
    } while (0)
#else
#define FIA_UT_ASSERT_EQ(actual, expected) ASSERT_EQ(actual, expected)
#define FIA_UT_ASSERT_LE(actual, expected) ASSERT_LE(actual, expected)
#define FIA_UT_EXPECT_EQ(actual, expected) EXPECT_EQ(actual, expected)
#define FIA_UT_REQUIRE_NOT_NULL(value, name) \
    do { \
        if ((value) == nullptr) { \
            throw std::runtime_error(std::string(name) + " is null"); \
        } \
    } while (0)
#endif

static bool IsFiaUtTraceEnabled() {
    const char *trace = std::getenv("FIA_UT_TRACE");
    return trace != nullptr && std::string(trace) != "0";
}

static void FiaUtTrace(const std::string &step) {
    if (IsFiaUtTraceEnabled()) {
        std::cerr << "[TRACE] " << step << std::endl;
    }
}

#define DO_TILING(tilingPara) \
    FiaUtTrace(std::string("build tensors: ") + tilingContextPara.opName_); \
    auto contextFaker = gert::TilingContextFaker(); \
    /* 1. input/output information */ \
    size_t inputNum = tilingContextPara.inputTensorDesc_.size(); \
    size_t outputNum = tilingContextPara.outputTensorDesc_.size(); \
    std::vector<uint32_t> inputIrInstance = {}; \
    std::vector<uint32_t> outputIrInstance = {}; \
    std::vector<gert::Tensor *> inputTensors = {}; \
    std::vector<gert::Tensor *> outputTensors = {}; \
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorsKeepAlive = {}; \
    std::vector<std::unique_ptr<gert::Tensor>> outputTensorsKeepAlive = {}; \
    for (size_t index = 0; index < inputNum; index++) { \
        if (tilingContextPara.inputTensorDesc_[index].shape_.GetStorageShape().GetDimNum() == 0) { \
            inputIrInstance.push_back(0); \
        } else { \
            inputIrInstance.push_back(1); \
            std::unique_ptr<gert::Tensor> curTensor = \
                std::make_unique<gert::Tensor>(tilingContextPara.inputTensorDesc_[index].shape_, \
                    gert::StorageFormat(tilingContextPara.inputTensorDesc_[index].format_, \
                        tilingContextPara.inputTensorDesc_[index].format_, gert::ExpandDimsType()), \
                    gert::TensorPlacement::kOnHost, tilingContextPara.inputTensorDesc_[index].dtype_, \
                    tilingContextPara.inputTensorDesc_[index].isConst_ \
                        ? tilingContextPara.inputTensorDesc_[index].constValue_ \
                        : nullptr); \
            inputTensors.push_back(curTensor.get()); \
            inputTensorsKeepAlive.push_back(std::move(curTensor)); \
        } \
    } \
    for (size_t index = 0; index < outputNum; index++) { \
        if (tilingContextPara.outputTensorDesc_[index].shape_.GetStorageShape().GetDimNum() == 0) { \
            outputIrInstance.push_back(0); \
        } else { \
            outputIrInstance.push_back(1); \
            std::unique_ptr<gert::Tensor> curTensor = \
                std::make_unique<gert::Tensor>(tilingContextPara.outputTensorDesc_[index].shape_, \
                    gert::StorageFormat(tilingContextPara.outputTensorDesc_[index].format_, \
                        tilingContextPara.outputTensorDesc_[index].format_, gert::ExpandDimsType()), \
                    gert::TensorPlacement::kOnHost, tilingContextPara.outputTensorDesc_[index].dtype_, \
                    tilingContextPara.outputTensorDesc_[index].isConst_ \
                        ? tilingContextPara.outputTensorDesc_[index].constValue_ \
                        : nullptr); \
            outputTensors.push_back(curTensor.get()); \
            outputTensorsKeepAlive.push_back(std::move(curTensor)); \
        } \
    } \
    if (tilingContextPara.inputInstanceNum_.size() != 0 || tilingContextPara.outputInstanceNum_.size() != 0) { \
        contextFaker.IrInstanceNum(tilingContextPara.inputInstanceNum_, tilingContextPara.outputInstanceNum_); \
    } else { \
        contextFaker.IrInstanceNum(inputIrInstance, outputIrInstance); \
    } \
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors); \
    contextFaker.DeterministicInfo(tilingContextPara.deterministicInfo_); \
    for (auto &attrInfo : tilingContextPara.attrs_) { \
        switch (attrInfo.attr_.type_) { \
        case Ops::Transformer::AnyValue::ValueType::VT_BOOL: { \
            contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_INT: { \
            contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_FLOAT: { \
            contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_STRING: { \
            contextFaker.Attr(attrInfo.attrName_, \
                ge::AscendString(reinterpret_cast<std::string *>(attrInfo.attr_.valuePtr_.get())->c_str())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_LIST_BOOL: { \
            contextFaker.Attr( \
                attrInfo.attrName_, *reinterpret_cast<std::vector<bool> *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_LIST_INT: { \
            contextFaker.Attr( \
                attrInfo.attrName_, *reinterpret_cast<std::vector<int64_t> *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_LIST_LIST_INT: { \
            contextFaker.Attr(attrInfo.attrName_, \
                *reinterpret_cast<std::vector<std::vector<int64_t>> *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        case Ops::Transformer::AnyValue::ValueType::VT_LIST_FLOAT: { \
            contextFaker.Attr( \
                attrInfo.attrName_, *reinterpret_cast<std::vector<float> *>(attrInfo.attr_.valuePtr_.get())); \
            break; \
        } \
        default: \
            std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType " << attrInfo.attr_.type_ \
                      << "is not supported!" << std::endl; \
        } \
    } \
    /* 2. base information */ \
    fe::PlatFormInfos platformInfo; \
    platformInfo.Init(); \
    auto tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_); \
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096); \
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get()); \
    auto contextHolder = contextFaker.SetOpType(tilingContextPara.opName_.c_str()) \
                             .CompileInfo(tilingContextPara.compileInfo_) \
                             .PlatformInfo(reinterpret_cast<char *>(&platformInfo)) \
                             .TilingData(tilingData.get()) \
                             .Workspace(ws_size) \
                             .Build(); \
    FiaUtTrace("context faker build done"); \
    string compileInfoStringPrefix = \
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": )"; \
    string compileInfoStringMiddle = \
        R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": )"; \
    string compileInfoStringSuffix = std::string(", \"socVersion\":\"") + tilingContextPara.socVersion_ + "\"} }"; \
    string compileInfoString = compileInfoStringPrefix + std::to_string(tilingContextPara.ubSize_) + \
        compileInfoStringMiddle + std::to_string(tilingContextPara.coreNum_) + compileInfoStringSuffix; \
    if (tilingContextPara.socInfoString_ != "") { \
        compileInfoString = tilingContextPara.socInfoString_; \
    } \
    map<string, string> socInfos; \
    map<string, string> aicoreSpec; \
    map<string, string> intrinsics; \
    map<string, string> versions; \
    string version = tilingContextPara.socVersion_; \
    if (isNpuArchString(version)) { \
        map<string, string> archToSoc = {{"3510", "Ascend950"}}; \
        versions = {{"NpuArch", tilingContextPara.socVersion_}, \
            {"Short_SoC_version", archToSoc[tilingContextPara.socVersion_]}}; \
    } else { \
        map<string, string> socToArch = { \
            {"Ascend310P", "2002"}, {"Ascend910B", "2201"}, {"Ascend910_93", "2201"}, {"Ascend950", "3510"}}; \
        versions = {{"NpuArch", socToArch[tilingContextPara.socVersion_]}, \
            {"Short_SoC_version", tilingContextPara.socVersion_}}; \
    } \
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics); \
    auto tilingContext = contextHolder.GetContext(); \
    FIA_UT_REQUIRE_NOT_NULL(tilingContext, "tilingContext"); \
    auto platformInfoPtr = tilingContext->GetPlatformInfo(); \
    FIA_UT_REQUIRE_NOT_NULL(platformInfoPtr, "platformInfo"); \
    FiaUtTrace("set platform resources"); \
    platformInfoPtr->SetPlatformRes("SoCInfo", socInfos); \
    platformInfoPtr->SetPlatformRes("AICoreSpec", aicoreSpec); \
    platformInfoPtr->SetCoreNumByCoreType("AICore"); \
    platformInfoPtr->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics); \
    platformInfoPtr->SetPlatformRes("version", versions); \
    /* 3. get tiling func */ \
    FiaUtTrace("lookup tiling func"); \
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(); \
    FIA_UT_REQUIRE_NOT_NULL(spaceRegistry, "spaceRegistry"); \
    auto opImpl = spaceRegistry->GetOpImpl(tilingContextPara.opName_.c_str()); \
    FIA_UT_REQUIRE_NOT_NULL(opImpl, "opImpl"); \
    auto tilingFunc = opImpl->tiling; \
    FIA_UT_REQUIRE_NOT_NULL(tilingFunc, "tilingFunc"); \
    /* 4. check tiling func */ \
    FiaUtTrace("call tiling func"); \
    auto tilingRet = tilingFunc(tilingContext);

static bool isNpuArchString(string version) {
    if (version.empty()) {
        return false;
    }

    for (char c : version) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

static std::string to_string_fnv_hash(const void *buf, size_t size) {
    const uint8_t *data = static_cast<const uint8_t *>(buf);
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL; // FNV偏移基础值
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL; // FNV质数
    constexpr uint64_t FNV_OUTPUT_LEN = 16; // FNV输出数据长度

    // FNV-1a算法
    uint64_t hash = FNV_OFFSET;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= FNV_PRIME;
    }

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(FNV_OUTPUT_LEN) << hash;
    return ss.str();
}

template <typename T> static string to_string(void *buf, size_t size, unordered_set<size_t> mask = {}) {
    string result;
    const T *data = reinterpret_cast<const T *>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += mask.find(i) == mask.end() ? std::to_string(data[i]) : "*";
        result += " ";
    }
    return result;
}

static unordered_set<size_t> GetMask(const string &expectTilingData) {
    unordered_set<size_t> mask;
    size_t index = 0;
    for (auto c : expectTilingData) {
        if (c == ' ') {
            ++index;
            continue;
        }
        if (c == '*') {
            mask.emplace(index);
        }
    }
    return mask;
}

static bool ExtractJsonValue(const string &jsonStr, const string &key, string &value) {
    const string quotedKey = "\"" + key + "\"";
    size_t keyPos = jsonStr.find(quotedKey);
    if (keyPos == string::npos) {
        return false;
    }

    size_t colonPos = jsonStr.find(':', keyPos + quotedKey.size());
    if (colonPos == string::npos) {
        return false;
    }

    size_t valueStart = jsonStr.find_first_not_of(" \t\r\n", colonPos + 1);
    if (valueStart == string::npos) {
        return false;
    }

    if (jsonStr[valueStart] == '"') {
        size_t valueEnd = jsonStr.find('"', valueStart + 1);
        if (valueEnd == string::npos) {
            return false;
        }
        value = jsonStr.substr(valueStart + 1, valueEnd - valueStart - 1);
        return true;
    }

    size_t valueEnd = jsonStr.find_first_of(",}\r\n", valueStart);
    if (valueEnd == string::npos) {
        valueEnd = jsonStr.size();
    }
    value = jsonStr.substr(valueStart, valueEnd - valueStart);
    size_t valueLast = value.find_last_not_of(" \t\r\n");
    if (valueLast == string::npos) {
        value.clear();
    } else {
        value.erase(valueLast + 1);
    }
    return !value.empty();
}

static void GetPlatFormInfos(const char *compileInfoStr, map<string, string> &socInfos, map<string, string> &aicoreSpec,
    map<string, string> &intrinsics) {
    string default_hardward_info = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432,
                          "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                          "CORE_NUM": 32}})";
    string compileInfoJson = compileInfoStr == nullptr || string(compileInfoStr).find("hardware_info") == string::npos
        ? default_hardward_info
        : string(compileInfoStr);

    map<string, string> socInfoKeys = {{"ai_core_cnt", "CORE_NUM"}, {"l2_size", "L2_SIZE"},
        {"cube_core_cnt", "cube_core_cnt"}, {"vector_core_cnt", "vector_core_cnt"},
        {"core_type_list", "core_type_list"}};

    for (auto &t : socInfoKeys) {
        string value;
        if (ExtractJsonValue(compileInfoJson, t.second, value)) {
            socInfos[t.first] = value;
        }
    }

    if (socInfos.find("cube_core_cnt") != socInfos.end() && socInfos.find("vector_core_cnt") != socInfos.end()) {
        socInfos["core_type_list"] = "CubeCore,VectorCore";
    } else {
        socInfos["core_type_list"] = "AICore";
    }

    map<string, string> aicoreSpecKeys = {{"ub_size", "UB_SIZE"}, {"l0_a_size", "L0A_SIZE"}, {"l0_b_size", "L0B_SIZE"},
        {"l0_c_size", "L0C_SIZE"}, {"l1_size", "L1_SIZE"}, {"bt_size", "BT_SIZE"},
        {"load3d_constraints", "load3d_constraints"}};
    aicoreSpec["cube_freq"] = "cube_freq";
    for (auto &t : aicoreSpecKeys) {
        string value;
        if (ExtractJsonValue(compileInfoJson, t.second, value)) {
            aicoreSpec[t.first] = value;
        }
    }

    std::string intrinsicsKeys[] = {"Intrinsic_data_move_l12ub", "Intrinsic_data_move_l0c2ub",
        "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz", "Intrinsic_matmul_ub_to_ub",
        "Intrinsic_conv_ub_to_ub", "Intrinsic_data_move_l12bt"};
    for (string key : intrinsicsKeys) {
        string value;
        if (ExtractJsonValue(compileInfoJson, key, value) && value == "true") {
            intrinsics[key] = "float16";
            if (key.find("Intrinsic_data_move_l12bt") != string::npos) {
                intrinsics[key] = "bf16";
            }
        }
    }
}

void ExecuteTestCase(const gert::TilingContextPara &tilingContextPara, ge::graphStatus expectResult,
    uint64_t expectTilingKey, const string &expectTilingData, const std::vector<size_t> &expectWorkspaces,
    uint64_t tilingDataReservedLen, bool useHashTilingData) {
    DO_TILING(tilingContextPara);

    // check tiling func
    FIA_UT_ASSERT_EQ(tilingRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check workspace
    if (!expectWorkspaces.empty()) {
        size_t workspaceCount = tilingContext->GetWorkspaceNum();
        if (workspaceCount > 0) {
            auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
            for (size_t i = 0; i < workspaceCount; i++) {
                FIA_UT_ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i]);
            }
        }
    }

    // check tiling key
    constexpr uint64_t SKIP_TILING_KEY_VALIDATION = UINT64_MAX;
    if (expectTilingKey != SKIP_TILING_KEY_VALIDATION) {
        auto tilingKeyResult = tilingContext->GetTilingKey();
        FIA_UT_ASSERT_EQ(tilingKeyResult, expectTilingKey);
    }

    // check tiling data
    if (expectTilingData != "") {
        auto rawTilingData = tilingContext->GetRawTilingData();
        auto tilingDataReservedSize = tilingDataReservedLen * sizeof(uint64_t);
        FIA_UT_ASSERT_LE(tilingDataReservedSize, rawTilingData->GetDataSize());
        auto tilingDataResult = useHashTilingData
            ? to_string_fnv_hash(rawTilingData->GetData() + tilingDataReservedSize,
                  rawTilingData->GetDataSize() - tilingDataReservedSize)
            : to_string<int64_t>(rawTilingData->GetData() + tilingDataReservedSize,
                  rawTilingData->GetDataSize() - tilingDataReservedSize, GetMask(expectTilingData));
        FIA_UT_EXPECT_EQ(tilingDataResult, expectTilingData);
    }
}

bool ExecuteTiling(const gert::TilingContextPara &tilingContextPara, TilingInfo &tilingInfo) {
    DO_TILING(tilingContextPara);

    if (tilingRet != ge::GRAPH_SUCCESS) {
        return false;
    }

    tilingInfo.tilingKey = tilingContext->GetTilingKey();
    tilingInfo.blockNum = tilingContext->GetBlockDim();
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        auto workSpaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            tilingInfo.workspaceSizes.push_back(workSpaceSizes[i]);
        }
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    tilingInfo.tilingData = std::make_unique<uint8_t[]>(rawTilingData->GetDataSize());
    tilingInfo.tilingDataSize = rawTilingData->GetDataSize();
    std::memcpy(tilingInfo.tilingData.get(), rawTilingData->GetData(), rawTilingData->GetDataSize());

    return true;
}
