# 自研 CANN 算子的运行期部署校验（可见性 / 走的是哪一个 / 数值对不对）

> **加载时机**：自研算子（mindiesd `csrc/ops/*` 产物）在运行环境里报
> `aclnnXxx … inferShape function does not exist`、行为"像没生效"、
> 或需要给出**部署侧**（不是算子实现侧）的通过证据时。
> **不重复的内容**：部署**顺序**的机制说明（产物位置、`ASCEND_CUSTOM_OPP_PATH`、必须先 `import mindiesd`）
> 单点在 `framework-integration/SKILL.md` §1.5 与
> `framework-integration/references/cache-dit-enablement.md` §2.2——本文件只承接**校验动作与判据**。
> 改 kernel 之后的**重建 / 缓存 / md5 / sentinel** 冒烟口径见 `mindiesd-fusion-notes.md` §1。
> **数字纪律**：不写绝对耗时 / 精度读数；逐 op 阈值以仓内该 op 的 golden 文件为准。

## 1. 三步部署校验（按顺序做，别跳）

1. **可见性**：运行进程里先 `import mindiesd`、**再**初始化 NPU / 建任何张量
   （GE 初始化后自定义算子注册不生效）。判据：不再报 `inferShape function does not exist`；
   仍报 ⇒ 是**部署顺序或算子包没进运行 CANN**，不是参数 / 几何问题（见 `../SKILL.md` §1.5 的根因三类）。
2. **走的是哪一个**：同名算子可能与 CANN 内建重名（见 `mindiesd-fusion-notes.md` §2）⇒
   用 sentinel 法（临时加可观测的语义改动）或计数证据确认实际执行的是自研产物，
   **不要假设**"改了源码跑的就是我的"。
3. **数值对不对**：跑该 op 的 golden
   （`tests/ops/{op}/…_golden.py`，如 `eagle_quant_block_sparse_attention_golden.py`）。
   **通过标准 = 该 golden 文件内声明的阈值**（EB / 容差逐 op 不同，以文件为准，不在此复制）；
   golden 过 + 数值冒烟过才算部署通过——**只看"编译过"或只查 shape 不算**
   （口径见 `mindiesd-fusion-notes.md` §1「数值冒烟」）。

## 2. 症状 → 落点判定表

| 症状 | 落点 |
|---|---|
| `aclnnXxx … inferShape function does not exist` | 部署顺序 / 算子包未进运行 CANN（本节 §1.1） |
| 改了 kernel 行为不变（输出仍旧语义） | 重建与缓存问题 → `mindiesd-fusion-notes.md` §1（tiling-key 缓存、全清重建、md5 核对） |
| 跑的是 CANN 内建而非自研产物 | 同名冲突 → `mindiesd-fusion-notes.md` §2（sentinel 实证；勿整算子改名） |
| golden 不过 / 数值不符 | 算子实现或数值契约 → 本技能开发链（实现侧）；量化契约级差异 → `quantization-dev` |
| 运行期算子 fallback / 没命中图的期望位点 | 不是部署问题 → `pattern-dev` / `framework-integration` §1.5 |

## 维护与更新

当新增自研算子、golden 位置或阈值口径变化、或出现新的"部署可见性"故障模式时，
按 dev-workflow 的复盘流程更新本文件；顺序机制仍以 `framework-integration` 对应节为真源。
