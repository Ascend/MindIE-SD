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


/* !
 * \file quant_four_over_six_a5_tiling.cpp
 * \brief
 */
#include "quant_four_over_six_a5_tiling_arch35.h"
#include <cmath>

// OPBASE 移除（同仓惯例）：inline 等价实现（源仓 opbase util/math_util.h、platform_util.h）。
namespace {
template <typename T> inline bool IsFloatEqualQ(T a, T b)
{
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return std::signbit(a) == std::signbit(b);
    }
    return a == b;
}
template <typename T> inline T CeilDivQ(T a, T b) { return (a + b - 1) / b; }
constexpr uint32_t VREG_SIZE_BYTES_Q = 256;  // OPBASE GetVRegSize 恒定值（vector 寄存器字节数）
}  // namespace
#include "platform/platform_info.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE = 3;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 4;
constexpr int64_t INDEX_ATTR_DST_DTYPE_MAX = 5;
constexpr int64_t NUM_TWO = 2;
constexpr float NUM_FOUR_FLOAT = 4.0; // dst_type_max=4: per-block 4/6 自适应触发值
constexpr int64_t MODE_THREE = 3;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t N_ALIGN64 = 64;
constexpr int64_t N_ALIGN128 = 128;
constexpr int64_t ATTR_BLOCK_SIZE = 32;
constexpr size_t MAX_DIM_NUM = 7;

// T9b: 仅保留 46 自适应（dst_type_max=4, bf16 -> fp4x2_e2m1, scale_alg=2, 尾轴, blocksize=32）。
// R1: 46 路径仅支持 bf16 输入（fp16 + dst_type_max=4 在 kernel 中无实现，强制拦截防静默）。
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};

template <typename T>
static inline uint64_t GetRemainder(uint64_t num, T div)
{
    return div == 0 ? div : num % div;
}

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static RoundModeList GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

static ge::graphStatus GetAttr(const gert::TilingContext* context, QuantFourOverSixA5TilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrAxis);
    tilingParam.axis = static_cast<int64_t>(*attrAxis);
    OP_LOGD(context->GetNodeName(), "The attr axis is %ld", tilingParam.axis);

    // B5: 硬性条件——量化轴（尾轴）32 对齐（infershape 同款，tiling 独立防绕过）
    auto* inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    const auto& inShapeRef = inShape->GetShape();  // StorageShape 无 GetDimNum/GetDim，经 GetShape() 取 Shape
    int64_t dimNorm = (tilingParam.axis >= 0) ? tilingParam.axis
                                              : tilingParam.axis + inShapeRef.GetDimNum();
    OP_CHECK_IF(
        (dimNorm != inShapeRef.GetDimNum() - 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
            std::to_string(tilingParam.axis),
            "dst_type_max=4 (4/6 adaptive) is only supported on the tail axis"),
        return ge::GRAPH_FAILED);
    int64_t dimSize = inShapeRef.GetDim(dimNorm);
    OP_CHECK_IF(
        (dimSize > 0 && dimSize % ATTR_BLOCK_SIZE != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
            std::to_string(tilingParam.axis),
            "The axis dim must be 32-aligned (4/6 adaptive is block-aligned only)"),
        return ge::GRAPH_FAILED);

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "round_mode", roundModeStr, "The value of round_mode must be [rint, round, floor]"),
        return ge::GRAPH_FAILED);
    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);
    // T9b/R1: 46 自适应仅支持 FLOAT4_E2M1（dst_type=40）
    OP_CHECK_IF(
        tilingParam.dstType != ge::DT_FLOAT4_E2M1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type",
            std::to_string(tilingParam.dstType),
            "dst_type_max=4 (4/6 adaptive) only supports dst_type=40 (FLOAT4_E2M1)"),
        return ge::GRAPH_FAILED);

    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.blockSize = static_cast<int64_t>(*attrBlockSize);
    // T9c/B1: 46 自适应仅 blocksize=32（尾轴模板仅支持 blockSize==32，T9a 已证；
    // 原 (0,1024] 且 %32==0 的宽容校验已收紧）
    OP_CHECK_IF(
        tilingParam.blockSize != ATTR_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "block_size",
            std::to_string(tilingParam.blockSize),
            "dst_type_max=4 (4/6 adaptive) is only supported with blocksize=32"),
        return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrScaleAlg);
    tilingParam.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    // T9b: 46 自适应仅支持 scale_alg=2
    OP_CHECK_IF(
        tilingParam.scaleAlg != NUM_TWO,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale_alg",
            std::to_string(tilingParam.scaleAlg),
            "dst_type_max=4 (4/6 adaptive) only supports scale_alg=2"),
        return ge::GRAPH_FAILED);

    auto* attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_DTYPE_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstTypeMax);
    tilingParam.dstTypeMax = static_cast<float>(*attrDstTypeMax);
    // T9b: 仅保留 4.0（46 自适应触发值）；0/6/7 等非自适应路径已随清理删除
    OP_CHECK_IF(
        !IsFloatEqualQ(tilingParam.dstTypeMax, NUM_FOUR_FLOAT),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type_max",
            std::to_string(tilingParam.dstTypeMax),
            "dst_type_max must be 4.0 (4/6 adaptive is the only remaining path)"),
        return ge::GRAPH_FAILED);
    tilingParam.invDstTypeMax = 1.0 / tilingParam.dstTypeMax;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(),
            "x", ge::TypeUtils::DataTypeToSerialString(xDtype), "[DT_BF16]"),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(),
            "y", ge::TypeUtils::DataTypeToSerialString(yDtype), "[DT_FLOAT4_E2M1]"),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "mxscale", ge::TypeUtils::DataTypeToSerialString(scaleDtype), "[DT_FLOAT8_E8M0]"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext* context, const QuantFourOverSixA5TilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape.GetDimNum() < 1 || xShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()), "The shape dim of x must be within the range [1, 7]"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParam.axis >= static_cast<int64_t>(xShape.GetDimNum()) ||
            tilingParam.axis < static_cast<int64_t>(-1 * xShape.GetDimNum()),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "axis", std::to_string(tilingParam.axis),
            "The value of axis must be within the range [-" + std::to_string(xShape.GetDimNum()) + ", " + std::to_string(xShape.GetDimNum() - 1) + "]"),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    if (Y_SUPPORT_DTYPE_SET.count(yDtype) != 0) {
        OP_CHECK_IF(
            GetRemainder(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_TWO) != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "x", Ops::Base::ToString(xShape),
                "When the yDtype is FLOAT4_E2M1, the tail axis of x must be an even number"),
            return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y", Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape), "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);

    auto axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    xShape.SetDim(axis, CeilDivQ(xShape.GetDim(axis), tilingParam.blockSize));

    auto mxScaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShapePtr);
    auto mxScaleShape = mxScaleShapePtr->GetStorageShape();

    OP_CHECK_IF(
        mxScaleShape.GetDimNum() != xShape.GetDimNum() + 1,
OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "mxscale", std::to_string(mxScaleShape.GetDimNum()),
            "The shape dim of mxscale must be the shape dim of x plus 1"),
        return ge::GRAPH_FAILED);

    auto newScaleShape = xShape;
    newScaleShape.SetDim(axis, (xShape.GetDim(axis) + DIGIT_TWO - 1) / DIGIT_TWO);
    newScaleShape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(
        newScaleShape != mxScaleShape,
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "mxscale", Shape2String(mxScaleShape), Shape2String(newScaleShape)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCalcMode(const gert::TilingContext* context, QuantFourOverSixA5TilingParam& tilingParam)
{
    // T9b: 仅 46 自适应（scale_alg=2 且 dst_type_max=4.0）→ MODE_THREE。
    // 非自适应 MODE_TWO（dst_type_max 0/6/7）与 MODE_ZERO/ONE 已随清理删除。
    if (tilingParam.scaleAlg == NUM_TWO) {
        tilingParam.calcMode = MODE_THREE;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale_alg",
            std::to_string(tilingParam.scaleAlg), "The value of scale_alg must be 2 (4/6 adaptive)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BaseCalc(const gert::TilingContext* context, QuantFourOverSixA5TilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    // 将量化轴索引转换成非负数
    tilingParam.axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    // 量化轴是否是尾轴
    tilingParam.isTailAxis = tilingParam.axis == static_cast<int64_t>(xShape.GetDimNum() - 1);
    int64_t dimSize = xShape.GetDim(tilingParam.axis);
    if (tilingParam.blockSize == ATTR_BLOCK_SIZE) {
        tilingParam.isBlockSize32 = true;
    }
    if (dimSize < tilingParam.blockSize) {
        tilingParam.blockSize = dimSize;
    }
    if (tilingParam.blockSize ==
        ATTR_BLOCK_SIZE) { // 需要再次判断，存在量化轴为尾轴，尾轴大小为32，但是blockSize>32时，也应该走尾轴、blockSize=32的模板
        tilingParam.isBlockSize32 = true;
    }
    // 合轴
    for (int64_t i = 0; i < tilingParam.axis; i++) {
        tilingParam.preAxisSize *= xShape.GetDim(i);
    }
    tilingParam.quantAxisSize = dimSize;
    for (size_t i = tilingParam.axis + 1; i < xShape.GetDimNum(); i++) {
        tilingParam.postAxisSize *= xShape.GetDim(i);
    }
    tilingParam.blockSizeNumInAxis = CeilDivQ(dimSize, tilingParam.blockSize);
    tilingParam.tailBlockSize = GetRemainder(dimSize, tilingParam.blockSize);
    tilingParam.isPad = tilingParam.tailBlockSize != 0;
    if (tilingParam.tailBlockSize == 0) {
        tilingParam.tailBlockSize = tilingParam.blockSize;
    }
    tilingParam.nAlignNum = tilingParam.postAxisSize == N_ALIGN64 ? N_ALIGN64 : N_ALIGN128;
    return ge::GRAPH_SUCCESS;
}

static bool IsTailAxisAndBlockSize32(QuantFourOverSixA5TilingParam& tilingParam)
{
    if (tilingParam.isTailAxis && tilingParam.isBlockSize32) {
        tilingParam.blockSize = ATTR_BLOCK_SIZE;
        return true;
    }
    return false;
}

ge::graphStatus Tiling4QuantFourOverSixA5(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4QuantFourOverSixA5 running begin.");

    QuantFourOverSixA5TilingParam tilingParam;

    OP_CHECK_IF(
        CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "The data type check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "The attr get failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (tilingParam.totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);

    OP_CHECK_IF(
        (tilingParam.ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    tilingParam.vfLen = VREG_SIZE_BYTES_Q;
    tilingParam.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_CHECK_IF(
        SetCalcMode(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Set calculation mode failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        BaseCalc(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The base calculation failed."), return ge::GRAPH_FAILED);

    // dst_type_max=4 触发 per-block 4/6 自适应：仅尾轴 + blocksize=32 支持
    // （自适应 kernel 仅在尾轴模板实现，非尾轴路径不应收到 dst_type_max=4）
    OP_CHECK_IF(
        IsFloatEqualQ(tilingParam.dstTypeMax, NUM_FOUR_FLOAT) &&
            (!tilingParam.isTailAxis || tilingParam.blockSize != ATTR_BLOCK_SIZE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type_max",
            std::to_string(tilingParam.dstTypeMax),
            "dst_type_max=4 (4/6 adaptive) is only supported on the tail axis with blocksize=32"),
        return ge::GRAPH_FAILED);

    bool IsTailAnd32 = IsTailAxisAndBlockSize32(tilingParam);
    // 当量化轴是尾轴，且BlockSize为32时，进入尾轴负载均衡模板
    // （T9b: 这是唯一剩余路径；非尾轴 main/optimize 模板已随清理删除）
    OP_CHECK_IF(
        !IsTailAnd32,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
            std::to_string(tilingParam.axis),
            "4/6 adaptive (dst_type_max=4) is only supported on the tail axis with blocksize=32"),
        return ge::GRAPH_FAILED);

    QuantFourOverSixA5TailAxisTiling tailAxisTiling(context, tilingParam);
    return tailAxisTiling.DoTiling();
}

ge::graphStatus TilingPrepare4QuantFourOverSixA5(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4QuantFourOverSixA5 entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the QuantFourOverSixA5 op.
IMPL_OP_OPTILING(QuantFourOverSixA5)
    .Tiling(Tiling4QuantFourOverSixA5)
    .TilingParse<QuantFourOverSixA5CompileInfo>(TilingPrepare4QuantFourOverSixA5);
} // namespace optiling

// T4 全量构建修复：源仓 opbase 公共库（third_party/opbase/src/op_common/log/log.cpp）
// 提供 Ops::Base::ToString(const gert::Shape&) 的实现，T2 迁移未携带该公共库；
// 本算子 tiling 错误路径引用该符号，单算子构建时 TBE 可从 CANN 全局符号表解析，
// 但 MindIE-SD 全量构建合并 liboptiling.so 后加载失败（undefined symbol），
// 导致 4 个 compute unit 的 bin 生成全部中断。此处按源仓实现补齐等价定义。
#include <sstream>
namespace Ops {
namespace Base {
static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static std::string ToString(const std::vector<int64_t> &v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

std::string ToString(const std::vector<const gert::Shape*> &v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << ToString(ToVector(*v[i])) << ", ";
        }
        oss << ToString(ToVector(*v[v.size() - 1]));
    }
    oss << "]";
    return oss.str();
}

std::string ToString(const gert::Shape &shape)
{
    return ToString(ToVector(shape));
}
} // namespace Base
} // namespace Ops
