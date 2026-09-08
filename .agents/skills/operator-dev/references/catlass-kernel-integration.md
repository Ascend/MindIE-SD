# catlass 类外部 AscendC kernel 接入 MindIE-SD（集成方案与踩坑）

> 场景：手头有 catlass（或类 catlass 的独立 ASC-language kernel 源，自带 host launch、
> 编译期模板、`<<<blockDim, nullptr, stream>>>` 形态）需要在 mindiesd 内**以标准算子形态**
> 使用（与其他算子同 `libPTAExtensionOPS.so`、同 `torch.ops.mindiesd::*` 注册、编译可选）。
> 本文覆盖 kernel 接入/构建/C++/设备侧；**接入 compile 图的约束与 compile 前后收益核验见
> `../compilation-dev/references/pattern-dev-notes.md` §5**（本文件不再重复）。
> MiniMax-H3 FFN 融合案例收益数字仅作指针：`tmp/mmx_w8a8/mmx_h3_w8a8_ffn_fusion_analysis.md`
> （§5i/§5j）、`tmp/mmx_w8a8/h3_w8a8_optimization_headroom.md`；同族第二案例
> mm_gelu_mxquant（FLUX/Wan/Qwen，含 GraphPatternEntry 真图使能）编排见
> `catlass-ffn-fusion-guide.md`，细节见 `mmgelu-flux-wan-qwen-case.md`。

## 1. 形态决策（先选，避免返工）

| 形态 | 说明 | 结论 |
|---|---|---|
| 外部 .so + python ctypes | kernel 编成独立 .so，python 侧 ctypes 加载/调 extern C | 弃：运行期外部路径、加载/分发脆、torch.compile 不可见（需再包 custom op） |
| 独立 kernel .so + plugin 链接 | mindiesd 构建先编 kernel .so，plugin 链接（DT_NEEDED + 部署同目录/`LD_LIBRARY_PATH`） | 过渡可用；仓库全局 `CMAKE_SKIP_RPATH` 禁 rpath，分发需额外 loader 处理 |
| **ASC`<<<>>>` 混编单 .so（推荐）** | kernel 源以 ASC 语言**直接编进 libPTAExtensionOPS.so**（CXX 链接器链接 ASC 对象，设备段保留，已实测） | 最终形态：单产物、无运行期依赖解析、注册与其他算子一致 |

## 2. 落地结构（推荐形态）

- kernel 源入仓（自包含），目录归 `csrc/ops/mm_swiglu_mxquant/mm_swiglu_mxquant.cpp`（与其他
  算子同 `csrc/ops/` 目录族）。
  - 自研的**派生 kernel/epilogue/tile 头 vendored** 到 `csrc/ops/{op}/include/catlass/...`
    （csrc/CMakeLists 将该 include 置于 `MINDIESD_CATLASS_HOME/include` 前），catlass 树保持
    0 改动；命名唯一，勿与 catlass 原生头同名（防 include 遮蔽）。完整流程见
    `catlass-ffn-fusion-guide.md`。
  - 注意：该目录**不提供** `op_host/CMakeLists.txt`——`csrc/ops` 的 CANN op 构建
    （`op_add_subdirectory`）只 glob 含 `op_host/CMakeLists.txt` 的目录，故本目录不会被当作
    aclnn op 收集；它仅为源码归属，由 plugin 构建以 ASC 语言混编进 `libPTAExtensionOPS.so`。
  - 剥离对 catlass examples/shared_lib 头与 platform-manager 的依赖（见 §5）；
    host 侧只保留：`ToUnderlyingArguments` + `GetWorkspaceSize`（静态缓存 workspace/group-list）+ `<<<blockNum, nullptr, stream>>>(params)`。
- 编译接线：`csrc/CMakeLists.txt`
  - 可选依赖：构建机设 `MINDIESD_CATLASS_HOME=<catlass 树>` 才启用；catlass 仅作 include 只读依赖（0 改动）；未设置时整 op 禁用（`register_ops.cpp` 用宏包注册、python fake 条件跳过，import 不炸）。
  - 启用分支：`find_package(ASC REQUIRED)` + `enable_language(ASC)` + catlass include +
    `target_compile_options`（ASC 专属 `--npu-arch=dav-3510` 等）+ `set_source_files_properties(... LANGUAGE ASC)` + `target_sources` 加 kernel 源。
- torch 入口：`csrc/plugin/mm_swiglu_mxquant.{h,cpp}` + `register_ops.cpp` 条件
  `m.def`/`m.impl`（`TORCH_LIBRARY_IMPL(..., PrivateUse1)`）；编译宏 `MINDIESD_CATLASS_FUSION_ENABLED` 由 csrc cmake 加。
- python 薄包装：`mindiesd/layers/mm_swiglu_mxquant.py`（校验 + 设备属性 + 调
  `torch.ops.mindiesd.*`）+ fake（`register_mindie_fake_op`，条件注册）。
- `build/build_plugin.sh`：kernel 编译已并入 plugin cmake，无预构建步骤（仅 env 提示）。

## 3. CMake/链接要点与坑

- 多语言 target（CXX+ASC）链接器 = 首个启用语言（CXX），可正常链接 ASC 对象（设备段保留，
  产物体积与 ASC-linker 相当）；无需强制 ASC 链接器。
- **`add_compile_options`/`link_directories` 晚于 target 创建无效** → 一律用
  `target_compile_options` / `target_link_directories`（本仓库 PTAExtensionOPS 在文件前部
  `add_library`，条件分支在尾部）。
- `<<<>>>` launch 桩 `AscendLaunchKernelWithHostArgs` 是**静态实现**，在 CANN
  `libascendc_runtime.a`（`${ASCEND_HOME_PATH}/x86_64-linux/lib64/` 或 `lib64/`，用 glob 定位）；
  其依赖符号（`mmGetTid`/`MsprofReportApi`/`DlogRecord`/`strcpy_s` 等）来自
  libruntime/libmmpa/libunified_dlog/libmsprofiler/libascend_dump/libprofapi/libascendalog/libc_sec
  ——plugin 全局 `-Wl,--no-undefined` 下必须全部显式链接（逐个补，直到链接通过）。
- rpath：仓库 `CMAKE_SKIP_RPATH TRUE` 会屏蔽 build/install rpath；手写
  `target_link_options(-Wl,-rpath,$ORIGIN)` 会被 makefile 二次转义成无效 `$$ORIGIN`。
  → 首选「无运行期依赖」形态（kernel 直接进同一 .so）；必须分 .so 时用 loader 前把目录
  前置 `LD_LIBRARY_PATH`（见 `register_ops._load_mindie_ops_library` 姿势）。

## 4. torch/plugin C++ 入口模式

- schema：`m.def("ns::op(Tensor ..., int aic_num) -> (Tensor, Tensor)")`；impl 注册
  `TORCH_LIBRARY_IMPL(mindiesd, PrivateUse1)`。
- **C++ kernel 多返回值必须 `std::tuple<at::Tensor, at::Tensor>`**：`std::vector` 被 torch
  判为单返回值 → schema 校验失败且在 dlopen 期直接 terminate（报
  "The number of returns is different. 2 vs 1"）。
- 当前 NPU stream：`aclrtStream s = c10_npu::getCurrentNPUStream();`
  （`torch_npu/csrc/core/npu/NPUStream.h`；python 侧取句柄用 `stream._as_parameter_.value`，
  `native_handle` 会抛 "Backend doesn't support"）。
- 权重侧一次性预处理（如列序调换）放 **C++ impl 内缓存**（进程级 static map，key=
  `data_ptr`，加锁）——避免把有状态逻辑留在 python traced 路径（compile guard 风险见
  pattern-dev-notes.md §4/§5）。
- fp8 输出：`at::empty({m, f}, x1.options())`（dtype 随输入 fp8e4m3）；scale 用
  `dtype(at::kByte)`。

## 5. 设备/运行时事实与坑

- 平台核数查询 `aclrtGetDeviceCapability(..., ACL_DEV_ATTR_AICORE_CORE_NUM=101, ...)` 在
  部分设备返回 `rc=107000`（不支持）→ 用 `torch.npu.get_device_properties(dev).cube_core_num`。
- `platform_ascendc::PlatformAscendCManager` 实现位于 **ATB `libtbe_adapter.so`**（非
  ascendcl/platform）——为免外部依赖，kernel 内**不要**查询 AIC 核数，改由调用方把
  `blockNum`（= cube_core_num）作为 op 的 `int` 参数传入。
- fp8 张量不能 `torch.randn` 直接生成（NPU normal 不支持 fp8）→ bf16 randn +
  `npu_dtype_cast`；e8m0 scale 测试数据可从 float 转 uint8。
- catlass `DeviceGemm` host API（CanImplement/GetWorkspaceSize/ToUnderlyingArguments/
  `<<<>>>`）只依赖 ACL stream + device 指针 → torch 张量 `data_ptr` 零拷贝直喂。

## 6. 集成侧验证（数值口径见第 3 条"位级特例"说明；compile 侧见 pattern-dev-notes.md §5）

1. standalone 数值对拍：同量化输入（DxQ 出 fp8+scale）喂 kernel，输出解码对拍 fp64
   model-semantics 参考（rel≈fp8 量化级 ~0.004 即通过；量化输出自比对有 2 的幂盲区，不可作唯一基准）。
2. 容器级对拍：同一 FFN 容器原实现 vs 融合实现同输入输出（应 ~0）。
3. 全流程位级一致：同种子跑两遍（融合 off/on）比较 latents（应 0）——
   **位级仅当融合前后数值路径完全等价时才成立（h3 mm_swiglu 特例：输入量化与激活路径一致）**；
   一般融合（如 mm_gelu 的 kernel fp32 激活 vs 参考 bf16 路径）为 **fp8 量化级近似**（字节一致
   98%+/rel≈1e-3），验收按量化级容差 + 模型级质量门，勿套"应 0"（判别法见
   `mmgelu-flux-wan-qwen-case.md` §2）。
4. eager/compile 双跑、compile 回归（重编译）排查与 kernel 级收益 diff → 属 compile 侧，
   见 `../compilation-dev/references/pattern-dev-notes.md` §5（含 MiniMax w8a8 实测数字指针）。

## 维护与更新

以下情况按 dev-workflow 复盘更新本文：新的外部 kernel 接入形态/工具链变化、CANN ASC
链接与 `libascendc_runtime.a` 位置变化、mindiesd 构建流程（setup.py/build_plugin.sh）改动。
compile 侧结论变化时更新 `../compilation-dev/references/pattern-dev-notes.md` §5。
