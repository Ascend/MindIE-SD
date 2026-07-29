# 错误码参考

MindIE-SD 的错误码统一采用 `MIE<XX>E<NNNNNN>` 的格式，其中产品段 `06` 代表 MindIE-SD。错误码会在 Error 级别的日志中输出，日志格式与字段含义请参见[日志参考](log.md)。

MindIE-SD 常见错误码及其含义、可能原因与排查建议如[表1](#table1)所示。

**表 1** MindIE-SD 错误码 <a id="table1"></a>

|错误码|错误描述|可能原因|排查建议|
|--|--|--|--|
|MIE06E000001|参数非法（Parameters invalid）。输入参数不符合要求。|输入参数的类型、取值范围、shape 或支持的取值列表与要求不一致。|对照日志消息中的实际参数与期望值，修正调用方传入的参数。|
|MIE06E000002|配置参数错误（Config parameter err）。配置项校验失败。|配置项缺失、非法，或与运行时要求不一致。|检查配置文件、环境变量，以及日志消息中给出的期望值。|
|MIE06E000003|Torch 执行错误（Torch exec err）。torch/TorchNPU 算子执行失败。|torch 或 TorchNPU 算子在执行过程中失败。|检查算子的输入 shape、dtype、设备放置，以及 CANN/TorchNPU 的错误栈。|
|MIE06E000004|模型初始化错误（Model init err）。模型初始化失败。|模型权重、配置或运行时资源未就绪。|检查模型路径、权重文件、配置取值、NPU 显存，以及初始化错误栈。|
|MIE06E000005|模型执行错误（Model exec err）。模型执行失败。|模型 forward、调度或自定义算子路径在执行过程中失败。|检查请求参数、tensor 的 shape 与 dtype、调度器状态，以及 CANN 算子的错误栈。|

> [!NOTE]说明
> 以上为 MindIE-SD 常见错误码。
