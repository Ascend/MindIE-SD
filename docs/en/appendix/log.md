# Log Reference

Log operations are normalized. You can set logging for all MindIE components through six environment variables. See [Table 1](#table1). For details about the log configuration of the environment variables, see [MindIE Log Reference](https://www.hiascend.com/document/detail/en/mindie/23RC1/ref/logreference/mindie_log_0213.html).

## Usage Instructions

    The MindIE component name values are (with the MindIE prefix omitted): [motor, server, llm, llmmodels, sd].
    The MindIE SD component name is *sd*

**Table 1** Log settings  <a id="table1"></a>

|Environment Variable|Default Configuration|Value Range and Description|
|--|--|--|
|MINDIE\_LOG\_LEVEL|INFO|Unified configuration of the log level of each MindIE component.<br>The log level values are [CRITICAL, ERROR, WARN, INFO, DEBUG]; when the value is null, logging is disabled.|
|MINDIE\_LOG\_TO\_FILE|true|Unified configuration of whether to write the logs of each MindIE component to files.<br>The value range is [false, true], and [0, 1] is also supported.|
|MINDIE\_LOG\_TO\_STDOUT|true|Unified configuration of whether to print the logs of each MindIE component.<br>The value range is [false, true], and [0, 1] is also supported.|
|MINDIE\_LOG\_VERBOSE|true|Unified configuration of whether to add optional log content to the logs of each MindIE component. Currently, logs are divided into fixed log content and optional log content. The complete debug log format is: **[date time]** [pid] [tid] [component name]**[uppercase log level]** [file:line] : **[error code] [*] log message**; the non-bold content is optional content, which is added when the environment variable is set to **true**.<br>The value range is [false, true], and [0, 1] is also supported.<br>**[*] in the log format indicates the name of a subcomponent or a smaller unit module, which can be optionally presented in the log to facilitate problem locating.**|
|MINDIE\_LOG\_PATH|~/mindie/log|Unified configuration of the directory where the log files of each MindIE component are saved.|
|MINDIE\_LOG\_ROTATE|<ul><li>s: default value is daily / 30</li><li>fs: default value is 20 (MB)</li><li>r: default value is 10</li></ul>|Unified configuration of log rotation for each MindIE component.<br>The log rotation format for a component is: *component name* : -s *cycle* -fs *filesize* -r *rotate*<ul><li>If there is no component name before ":", the setting applies to all components by default;</li><li>*cycle* indicates the time rotation period, which can be daily/weekly/monthly/yearly, or a number of days from 1 to 180, with a default of daily and a period of 30.</li><li>*filesize* indicates the size of each log file (in MB), with a value range of [1, 500].</li><li>*rotate* indicates the maximum number of log files that can be retained per process, with a value range of [1, 64].</li></ul>|

## Log Overview

Currently, MindIE SD logs contain only runtime debug logs.

### Log Record Format

When `MINDIE_LOG_VERBOSE` is enabled, the complete log format of all MindIE components is as follows:

```text
[date time] [pid] [tid] [component name] [uppercase log level] [file:line] : [error code] [*] log message
```

When `MINDIE_LOG_VERBOSE` is disabled, the log retains only the mandatory content, and the format is as follows:

```text
[date time] [component name] [uppercase log level] log message
```

> [!NOTE]NOTE
> \*: indicates that if a component contains subcomponents or smaller functional modules, they are presented before the log information. Both the error code and the "\*" subcomponent identifier are output as part of the log message.

**Table 2** Log field description

|Field|Description|
|--|--|
|**date time**|Date and time.|
|pid|Process ID.|
|tid|Thread ID.|
|Component name|The component name of MindIE. MindIE SD is always presented as [MindIE-SD] in logs.|
|**Uppercase log level**|The uppercase form of the log level. For details about log levels, see [Table 4 Log levels](#table4).|
|file:line|File name: code line number.|
|error code|The error code of Error-level logs. For details about error codes, see *[MindIE SD Error Code Reference](error_code.md)*.|
|**log message**|Specific error information.|

**The bold content is mandatory in logs**, while the remaining fields are optional log information and can be configured through the environment variable "MINDIE\_LOG\_VERBOSE". For details, see [Configure Log Content](#configure-log-content).

## Viewing Logs

MindIE collects logs at the INFO level and above by default. The default flush path of log files is shown in [Table 3](#table3). For details about how to set the flush path, see [Configuring the Log Flush Path](#configuring-the-log-flush-path).

**Table 3** Log path <a id="table3"></a>

|Path|Description|
|--|--|
|~/mindie/log|Default log flush path.|
|~/mindie/log/debug|Path of the runtime debug logs automatically generated under the default log flush path.|

Log files are named in the unified format `mindie-sd_process ID_timestamp.log` (the timestamp format is YYYYMMDDHHMMSS, accurate to the second). You can locate the relevant log file based on the process ID and timestamp.

[Example 1] Log file of MindIE SD.

```text
mindie-sd_123_20241008020600.log
```

Use the following command to view logs. Replace the log file name in the command with the actual file name.

[Example 2] View the log file of MindIE SD.

```bash
cat mindie-sd_123_20241008020600.log
```

## Others

### Setting the Log Level

Runtime debug logs are divided into five levels as shown in [Table 4](#table4).

**Table 4** Log levels <a id="table4"></a>

|Log Level|Abbreviation|Log Content|
|--|--|--|
|CRITICAL|critical|Critical. An emergency in which the system service is severely impaired or completely unavailable, affecting users at scale and requiring immediate handling by O&M personnel. For example, the system fails to start or a process hangs.|
|ERROR|error|Error. The system operating environment or functions are affected, or unexpected data/events cause function execution errors. For example, data fails to be written to the database or task creation fails.|
|WARNING|warn|Warning. A potential risk or hidden danger in the system that does not affect the normal execution of system functions. For example, data verification contains errors, but the system can recover through the error correction function without affecting function execution.|
|INFO|info|Information. Used to record normal system operation information, outputting some status or status change information, such as the current system status and database connection status.|
|DEBUG|debug|Debug. Used to trace the running path, such as tracing function entry and exit, and record debugging information. The recorded information is comprehensive and is intended for developers to locate complex problems. It adds code-level information output, such as the name and parameters of the currently called function, internal variable values, and function call return values. This information must be recorded before an exception or error is returned.|

The log levels are ordered from low to high as follows: DEBUG < INFO < WARNING < ERROR < CRITICAL. The lower the level, the more detailed the output logs.

Set the log level of each component through the environment variable "MINDIE\_LOG\_LEVEL". The default log level is "info".

The specific format for setting the log level of a component is: _component name_:  _log level_.

- The log level has the following options: [critical, error, warn, info, debug]. null indicates that the log of the corresponding component is disabled.

- The component name has the following options: [motor, server, llm, llmmodels, sd]

- If there is no component name before ":", the setting is applied to all components by default for unified configuration.

- When setting log levels for multiple components at the same time, separate them with ";". A later setting has a higher priority than an earlier one, and the later setting overrides the earlier one.

> [!NOTE]NOTE
> The values of the preceding components and log levels are case-insensitive.

[Example 1] Set the log level of all MindIE components to "debug" in a unified manner.

```bash
export MINDIE_LOG_LEVEL="debug"
```

[Example 2] Set the log level of MindIE SD to "debug", and set the levels of the other components to "info".

```bash
export MINDIE_LOG_LEVEL="info ; sd:debug"
```

### Setting the Log Display Mode

Use the environment variable "MINDIE\_LOG\_TO\_FILE" to set whether logs of each MindIE component are written to a file. The default value is "true", indicating that logs are written.

Use the environment variable "MINDIE\_LOG\_TO\_STDOUT" to set whether logs of each MindIE component are printed. The default value is "true", indicating that logs are printed.

The format for setting whether logs of a component are written or printed is: _component name_: \{0, 1, true, false\}.

- "0" and "false" indicate no, and "1" and "true" indicate yes.

- If no component name precedes ":", the setting applies to all components by default.

- When configuring multiple components at the same time, separate them with ";". Settings that appear later take precedence over earlier ones, and later settings override earlier settings.

[Example 1] Do not write MindIE SD logs to a file.

```bash
export MINDIE_LOG_TO_FILE="sd: false"
```

[Example 2] Print the log streams of all MindIE components.

```bash
export MINDIE_LOG_TO_STDOUT="true"
```

### Configuring the Log Flush Path

Use the environment variable "MINDIE\_LOG\_PATH" to set the flush path for the logs of each MindIE component. The default flush root directory is "~/mindie/log", and the actual log files are uniformly written to the "debug" subdirectory under it (that is, "~/mindie/log/debug" by default).

The format for setting the log flush path is: *component name*: *path*.

- If the path starts with "/", it indicates that the path is an absolute path, and logs are written to the "debug" subdirectory under that path;

- If the path does not start with "/", it indicates that the path is a relative path, relative to the default root directory "~/mindie/log", and logs are written to the "debug" subdirectory under the concatenated path;

- If no component name precedes ":", the setting applies to all components by default.

- When configuring multiple components at the same time, separate them with ";". A later setting takes precedence over an earlier one, and the later setting overrides the earlier one.

> [!NOTE]NOTE
>
> - The path must not contain special characters such as control characters.
> - The log path must not be a symbolic link. The program validates it. Ensure that the log path is valid.

[Example 1] Flush the MindIE SD logs to "/home/working/debug".

```bash
export MINDIE_LOG_PATH="sd: /home/working/"
```

[Example 2] Flush the MindIE SD logs to "~/mindie/log/sd/debug".

```bash
export MINDIE_LOG_PATH="sd: sd"
```

### Configure Log Content

Use the environment variable "MINDIE\_LOG\_VERBOSE" to set whether to print optional information in the log content of a component. The default value is "true", which means optional information is printed.

The setting format is: _component name_: \{0, 1, true, false\}.

- "0" and "false" mean no, and "1" and "true" mean yes.

- If there is no component name before ":", the setting applies to all components by default.

- When setting multiple components at the same time, separate them with ";". The later setting has a higher priority than the earlier one, and the later setting overrides the earlier one.

[Example 1] Do not print or save the optional log content of all MindIE components in a unified manner.

```bash
export MINDIE_LOG_VERBOSE="false"
```

[Example 2] Print or save the optional log content of MindIE SD.

```bash
export MINDIE_LOG_VERBOSE="sd: true"
```

### Configuring Log Rotation

Logs are rotated when either of the following conditions is met: the size of a single log file reaches the configured file size upper limit, or the configured time period is reached (by default, daily at 00:00). Log rotation is configured through the environment variable "MINDIE\_LOG\_ROTATE".

The rotation-related parameters, their default values, and value ranges are as follows:

- Time period: rotation by day (daily), week (weekly), month (monthly), or year (yearly) is supported. A number of days from 1 to 180 can also be set. The default is daily with a period of 30.

- File size: the value range of each log file size is 1 MB to 500 MB, and the default value is 20 MB.

- Number of files: The value range of the maximum number of log files that can be retained per process is 1 to 64, and the default value is 10. When the number of historical files exceeds this value, the oldest file will be deleted.

Set the log rotation format of a component as follows: _component name_: -s _cycle_ -fs _filesize_ -r _rotate_

- If there is no component name before ":", unified configuration is performed for all components by default.

- When multiple components are set at the same time, separate them with ";". The later setting has a higher priority than the earlier one, and the later setting overrides the earlier one.

- -s _cycle_ indicates the time period, which can be daily/weekly/monthly/yearly, or a number of days from 1 to 180; -fs _filesize_ indicates the size of each log file (in MB); -r _rotate_ indicates the maximum number of log files that can be retained per process.

    > [!NOTE]NOTE
    > After rotation, the newly generated log file is named in the format `mindie-sd_process ID_timestamp.log` (the timestamp is accurate to the second). When the number of historical files exceeds the maximum number of log files that can be retained per process, the oldest log file is automatically deleted.

[Example 1] Set the log file size of all MindIE components to 500 MB in a unified manner.

```bash
export MINDIE_LOG_ROTATE="-fs 500"
```

[Example 2] Set the MindIE SD log file size to no more than 40 MB, and retain one file per process.

```bash
export MINDIE_LOG_ROTATE="sd: -fs 40 -r 1"
```
