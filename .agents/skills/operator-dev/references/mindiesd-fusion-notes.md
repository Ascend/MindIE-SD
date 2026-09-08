# MindIE-SD 算子融合经验（MindIE-SD/CANN 特有；与 cannbot 经验并行使用，不互斥）

> 来源：MiniMax-H3 qk_norm+RoPE 大融合全流程（图识别→triton→Ascend C→多轮远端
> 全清重建→AB 否定）复盘。只收 **MindIE-SD/CANN 集成侧** 经验；triton 算法级优化
> （avoid_scalar_lowering / vector_core_partition / 双缓冲等）属 cannbot，不在此重复。

## 1. kernel 改动"没生效"排障（最高频坑，先看这条）

现象：改了 kernel 源码、重编后行为不变（如 Ascend C 输出仍旧语义）。

- **根因链**：① 本仓 custom op 的 kernel `.o/.json` 按 **tiling-key 哈希**缓存，改源码不改
  key（模板参数）→ 编译系统复用旧 `.o`；② `setup.py build_py` 对已存在 vendors 多为
  **拷贝预编译产物**，增量不可信；③ 推送后**远端源副本可能没更新**（csrc 与编译用
  vendors/`build/.../src` 可能不一致）。
- **标准动作（改 kernel 后必做）**：
  1. sftp 推送 csrc 源 → **读回并 grep 校验**关键标记（防推送未生效）；
  2. `rm -rf build/build build/vendors mindiesd/ops/vendors build/lib.linux-x86_64-cpython-312/mindiesd/ops`；
  3. `source set_env.sh && python setup.py build_py && pip install -e . --no-deps`（全清重建约 25-45min）；
  4. md5 校验 `csrc/ops/{op}` 与 `mindiesd/ops/vendors/.../aie_ascendc_impl/.../{op}` 一致；
  5. **数值冒烟**（不能只看编译过/只查 shape）。
- **sentinel 法**：无法确认"跑的是不是我改的 kernel"时，在 kernel 里加一处**可观测的
  语义改动**（如 partial 时输出尾段 ×-1，注意 bf16 不支持 Muls，放 fp32 侧 cast 前），
  重建后看输出是否带该标记 → 一锤定音是否生效/数据是否完整。
- 案例：partial 96/128 "尾列全零"误判为代码 bug 多轮，实为旧二进制未加载；sentinel 证实
  后转置（含尾列）一次正确。

## 2. 同名算子与 CANN 内建冲突

- CANN 会**内建同名算子**（案例：CANN 9.1 内建 `NormRopeConcat`，repo `csrc/ops/` 是其
  backport，tiling-key 哈希一致）。运行时实际执行哪个（builtin vs 自定义 OPP）不显然 →
  用 §1 sentinel 实证，不要假设。
- **不要用"整算子改名"规避同名冲突**：本仓自定义 op 的 op 名/目录名/文件基名与 CANN
  autogen（`aclnn_{op}.cpp`、proto autogen、kernel config）**强绑定**；改名会触发
  "No rule to make target .../autogen/..." 类基建失败（实测两次 clean 重建失败）。
- 更稳的解法候选（未走通，留给后续）：plugin 层走**直调**（绕过 aclnn 名字解析），或
  自定义 OPP 优先级/安装路径控制——动手前先查加载器（`register_ops.py`/`find_op_path`）。

## 3. Ascend C API 陷阱（非算法，cannbot 文档可能不覆盖）

- `Muls/Adds` 等标量二元 API **不支持 bf16(`__bf16`) 张量**（static assert
  `SupportType<__bf16,...>`）；同类操作移到 fp32 侧做，再 cast。
- `Gather(dst, src, idx, base, count)` 的**索引单位/语义需先用文档或最小样例确认**：
  本会话用 partner 索引做 HALF 旋转，D=8 确定性对拍发现第二半输出恒取 `x[0]`（索引未按
  预期映射）——库内无参照用法，盲改 + 30-45min 重建/轮 成本极高。
- 尽量用**对齐分段/重载**代替跨 lane 置换：需要 "行内伙伴元素" 时，直接对齐加载
  48/48/32 子块（寄存器已有），避免 gather。

## 4. triton-on-Ascend 短行地板与瓶颈判定

- 短行（D=128）+ 行内归约的 vector kernel 存在 **~440us/site 量级地板**（实测 1×3967×56×128）。
- **瓶颈判定法（写 kernel 前/后快速做）**：把"读流量减半"当实验——本案例 48/48/32 单趟
  读（256→128 元素/行）只降 6% → 判定瓶颈是**指令/延迟（行内归约）而非带宽**，别继续在
  流量上优化。
- 跨 lane gather/置换在 NPU triton 上会**标量化（慢 ~100× 量级）**，用对齐分段重载。
- **有损叠加（w8a8）冲突**：QuantMatmul 输出 fp32 中间岛时，为 bf16 调的 triton 融合核
  严重退化（norm_rope 案例 0.44→1.6ms/site，且连带 InplaceCopy/Cast 膨胀 0.37→17.8ms，
  模型 16.62→88.84ms；该融合已整体移除）→ **fusion pattern 需按图 dtype 门控**，默认对
  量化路径关闭。

## 5. 融合收益前置评估（写 kernel 之前先判"值不值"）

- 先给目标区域**带宽下限**：`(读+写字节) / 实测有效带宽(~600GB/s)` 得理论最小耗时；
  再算**区域占比**（案例：GEMM 74% / FA 12.5% / 目标区 4.5%）。
- **统一口径再比**：per-site vs per-block vs 模型级要写清；旧链用 profile 多 kernel
  device 时间求和，新核用 standalone event 计时（含 launch）——两者严格不可比，最终以
  **模型级 AB** 为准（案例：triton 融合 per-site 443us > 旧链 ~270us → AB +0.80ms 定论）。
- 收益天花板法：若"目标区域 100% 免费"也只有 ~1% 级收益（案例 ≤24.0ms 目标需
  block ≤440us 而带宽下限已 >500us）→ 直接判定不可达，省掉实现轮次。
- 负面结论同样要归档（本会话已入 `dummy-run/references/minimax-h3-notes.md §12`），
  供后续不做重复实验。

## 6. catlass 复用边界与语义核验（2026-09 MiniMax-H3 FFN 融合研究；仅收录已验证/非特例项）

> 会话级性能与"融合收益"结论（单设备观察，可能为特例）**不收录**；完整记录见会话工作区
> `tmp/mmx_w8a8/mmx_h3_w8a8_ffn_fusion_analysis.md`。

- **复用边界（本项目协作约定）**：catlass 作只读依赖——不向其 include/、dispatch policy、
  examples CMake 列表写入；派生 kernel/epilogue/tag 外置（experiment 期可放 example 内自包含
  include + 自有命名空间 tag；`BlockEpilogue` 偏特化放自有头即可，无需 patch 库）。**正式
  集成落点 = vendored 进 mindiesd `csrc/ops/{op}/include/catlass/...`**（csrc/CMakeLists 将该
  include 置于 `MINDIESD_CATLASS_HOME/include` 之前；命名唯一不与 catlass 原生同名），
  完整流水线见 `catlass-ffn-fusion-guide.md`。实验期临时挂载后**必须整体移除**并核验 0 残留
  （目录/CMake 列表/旧二进制）。
- **构建缓存坑（已验证）**：catlass `scripts/build.sh` 复用 build 目录时**不重新 configure**，
  改动 `-D{宏}` 需 `rm -rf build` 全清重建，否则旧宏残留导致"改源码行为不变"。
- **语义核验方法（已验证可行）**：需要确认 catlass 例程/自研融合 kernel 的数值语义时，
  以 **fp32 真实输出例程（如 53）** 或**量化输出解码后与 fp64/真实值对拍**为准，不要只依赖
  量化输出的自比对（其敏感性未证实）。
- **torch↔catlass MXFP8 数据格式一致（实测于 m=128/k=512）**：`npu_dynamic_mx_quant` 的
  fp8e4m3+e8m0 scale（A: `[m,ceil(k/32)/2,2]`；W 按 k 量化: `[n,ceil(k/32)/2,2]`）与
  catlass 例程输入布局字节一致，可直接喂入；解码对拍 rel ~2%（测试范围外未验证）。
  注：`ceil(k/32)/2` 与本文档其他处 `ceil(k/64)` 为同一布局的两种写法（k%64==0 时相等），
  以实际 torch `npu_dynamic_mx_quant` 输出 shape 为准。
