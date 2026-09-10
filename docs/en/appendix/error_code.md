# Error Code Reference

MindIE-SD error codes uniformly adopt the `MIE<XX>E<NNNNNN>` format, where the product segment `06` represents MindIE-SD. Error codes are output in Error-level logs. For the log format and field meanings, see [Log Reference](log.md).

Common MindIE-SD error codes and their meanings, possible causes, and troubleshooting suggestions are shown in [Table 1](#table1).

**Table 1** MindIE-SD error codes <a id="table1"></a>

|Error Code|Error Description|Possible Cause|Troubleshooting Suggestion|
|--|--|--|--|
|MIE06E000001|Parameters invalid. The input parameters do not meet the requirements.|The type, value range, shape, or supported value list of the input parameters is inconsistent with the requirements.|Compare the actual parameters in the log message with the expected values, and correct the parameters passed by the caller.|
|MIE06E000002|Configuration parameter error. Configuration item validation failed.|A configuration item is missing, invalid, or inconsistent with runtime requirements.|Check the configuration file, environment variables, and the expected values given in the log message.|
|MIE06E000003|Torch execution error. torch/TorchNPU operator execution failed.|A torch or TorchNPU operator failed during execution.|Check the input shape, dtype, and device placement of the operator, as well as the CANN/TorchNPU error stack.|
|MIE06E000004|Model initialization error. Model initialization failed.|Model weights, configuration, or runtime resources are not ready.|Check the model path, weight files, configuration values, NPU memory, and the initialization error stack.|
|MIE06E000005|Model execution error. Model execution failed.|The model forward, scheduling, or custom operator path failed during execution.|Check the request parameters, tensor shape and dtype, scheduler status, and the CANN operator error stack.|

> [!NOTE]NOTE
> The preceding lists the common MindIE-SD error codes.
