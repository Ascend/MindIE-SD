# Environment Variables

MindIE SD related environment variables are as follows:

**Table 1** Environment Variables

| Environment Variable | Description | Configuration | Default |
| -- | -- | -- | -- |
| MINDIE_LOG_LEVEL | MindIE SD log level. | Supports levels: "critical", "error", "warn", "info", "debug", "null". When set to "null", logging is disabled. Supports component-specific configuration: `export MINDIE_LOG_LEVEL="sd:debug"` where "sd:" specifies the SD component (the prefix can be omitted). | info |
| MINDIE_LOG_TO_STDOUT | MindIE SD log stdout control switch. | Supports "true", "1", "false", "0". "true" or "1" enables, "false" or "0" disables. Supports component-specific configuration: `export MINDIE_LOG_TO_STDOUT="sd:true"`. | true |
| MINDIE_LOG_TO_FILE | MindIE SD log file persistence control switch. | Supports "true", "1", "false", "0". "true" or "1" enables, "false" or "0" disables. Supports component-specific configuration: `export MINDIE_LOG_TO_FILE="sd:true"`. | true |
| MINDIE_LOG_PATH | MindIE SD log file save path configuration. | Supports user-specified log storage path. Default is "~/mindie/log/". Runtime logs are saved in the "debug" subdirectory of the specified path. Relative paths like "./custom_log" will be resolved under "~/mindie/log/custom_log". Absolute paths like "/home/usr/custom_log" write directly. Supports component-specific configuration: `export MINDIE_LOG_PATH="sd:./custom_log"`. | ~/mindie/log/ |
| MINDIE_LOG_ROTATE | MindIE SD rotation configuration. | Supports rotation period "-s" with values: "daily", "weekly", "monthly", "yearly", and positive integers (interpreted as days). Supports max file size "-fs" in MB (e.g., "-fs 20"). Supports max file count "-r" (e.g., "-r 10"). Can combine all: `export MINDIE_LOG_ROTATE="-s 10 -fs 20 -r 10"`. | -s 30 -fs 20 -r 10 |
| MINDIE_LOG_VERBOSE | MindIE SD verbose logging switch. | Supports "true", "1", "false", "0". "true" or "1" enables, "false" or "0" disables. Supports component-specific configuration: `export MINDIE_LOG_VERBOSE="sd:true"`. | true |
| SKIP_ALL_OPS_PLUGIN_BUILD | Operator and PyTorch plugin cache reuse switch for CI builds only. | When set to "1", complete cached operator artifacts allow the incremental build to skip kernel compilation. Plugin compilation is skipped only when the cached `libPTAExtensionOPS.so` can be loaded by the current PyTorch. Missing, incomplete, or unloadable artifacts automatically fall back to a full build. | 0 |
