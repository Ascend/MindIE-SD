# 阶段耗时与显存追踪

## _PhaseTimer 类

在每个 dummy run 脚本中内嵌统一的 `_PhaseTimer` 实例，追踪 BUILD 和 INFERENCE 两阶段的各模块耗时与显存：

- **BUILD 阶段**：各组件构造时调用 `timer.record_build(name, elapsed)`，自动记录耗时 + 显存增量
- **INFERENCE 阶段**：通过 `register_forward_pre_hook` / `register_forward_hook` 挂载到 pipe 的 `text_encoder`、`transformer`、`transformer_2`、`vae` 子模块，自动记录每次 forward 耗时和内存快照
- **Warmup/Timed 分离**：先 `capture_warmup()` 消除 NPU 算子冷启动，再 `capture_timed()` 精确计时

典型输出为 BUILD + INFERENCE 两段式汇总（**以下为格式示意，数值以 `<值>` 占位**）：

```text
======================================================================
  BUILD                                     Time(s)    Mem(GB)
  ----------------------------------------------------------
  Transformer                                   <值>      <值>
  Transformer_2                                 <值>      <值>
  VAE                                           <值>      <值>
  Text encoder + scheduler + tokenizer          <值>      <值>
  Move to device                                <值>      <值>
  ----------------------------------------------------------
  BUILD TOTAL                                   <值>

  INFERENCE                                 Time(s)    Mem(GB)   Peak(GB)
  --------------------------------------------------------------------
  -- Warmup --
  text_encoder                                  <值>      <值>      <值>
  transformer                                   <值>      <值>      <值>

  -- Timed --
  text_encoder                                  <值>      <值>      <值>
  transformer                                   <值>      <值>      <值>
  --------------------------------------------------------------------
  OVERALL TOTAL                                 <值>                <值>
======================================================================
```

## 注意事项

- forward hook 仅触发 `forward()` 调用，**不触发** `decode()` / `encode()` 等子方法。VAE decode 耗时需通过总耗时减去 hook 记录耗时反推。
- `torch_npu.npu.synchronize()` 在 hook 前后各调用一次，确保 NPU 异步操作已完成再计时。
- 显存通过 `torch_npu.npu.memory_allocated(device_id)` 获取已分配量，峰值取各阶段最大值。

## 维护与更新

当_PhaseTimer 实现或追踪需求变化时，按 dev-workflow 的复盘流程更新本文件。
