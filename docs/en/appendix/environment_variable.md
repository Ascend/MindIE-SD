# Environment Variables

The following environment variables are commonly used by MindIE SD.

**Table 1. Environment variables**

| Environment variable | Description | Configuration method | Default |
| -- | -- | -- | -- |
| `MINDIE_LOG_LEVEL` | Log level used by MindIE SD. | Supported levels are `"critical"`, `"error"`, `"warn"`, `"info"`, `"debug"`, and `"null"`. When set to `"null"`, logging is disabled. Direct configuration is supported, for example `export MINDIE_LOG_LEVEL="sd:debug"`. The `sd:` prefix scopes the setting to the SD component and may be omitted. | `info` |
| `MINDIE_LOG_TO_STDOUT` | Controls whether logs are printed to standard output. | Supported values are `"true"`, `"1"`, `"false"`, and `"0"`. Direct configuration is supported, for example `export MINDIE_LOG_TO_STDOUT="sd:true"`. The `sd:` prefix may be omitted. | `true` |
| `MINDIE_LOG_TO_FILE` | Controls whether logs are written to files. | Supported values are `"true"`, `"1"`, `"false"`, and `"0"`. Direct configuration is supported, for example `export MINDIE_LOG_TO_FILE="sd:true"`. The `sd:` prefix may be omitted. | `true` |
| `MINDIE_LOG_PATH` | Output path for MindIE SD log files. | Users may specify a custom log path. The default is `~/mindie/log/`, with runtime logs written under the `debug` subdirectory. If a relative path such as `./custom_log` is provided, the logs are written under `~/mindie/log/custom_log`. If an absolute path such as `/home/usr/custom_log` is provided, the logs are written directly there. Example: `export MINDIE_LOG_PATH="sd:./custom_log"`. The `sd:` prefix may be omitted. | `~/mindie/log/` |
| `MINDIE_LOG_ROTATE` | Log rotation configuration. | Supports the rotation period option `-s`, with values in `["daily", "weekly", "monthly", "yearly"]` or any positive integer. When a positive integer is used, the unit is days, for example `-s 100`. Supports the maximum log file size option `-fs` in MB, for example `-fs 20`. Supports the maximum file count option `-r`, for example `-r 10`. These options can be combined, for example `export MINDIE_LOG_ROTATE="-s 10 -fs 20 -r 10"`. | `-s 30 -fs 20 -r 10` |
| `MINDIE_LOG_VERBOSE` | Controls whether optional verbose log information is printed. | Supported values are `"true"`, `"1"`, `"false"`, and `"0"`. Direct configuration is supported, for example `export MINDIE_LOG_VERBOSE="sd:true"`. The `sd:` prefix may be omitted. | `true` |
