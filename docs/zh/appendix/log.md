# 日志参考

对日志操作进行了归一化，通过六个环境变量能对MindIE所有组件进行日志设置，参见[表1](#table1)。环境变量的日志配置详情请参见[MindIE日志参考](https://www.hiascend.com/document/detail/zh/mindie/23RC1/ref/logreference/mindie_log_0213.html)。

## 使用说明

    mindie组件名称取值为（省略mindie前缀）：[motor, server, llm, llmmodels, sd]。
    MindIE SD组件名称即为 *sd*

**表 1** 日志设置  <a id="table1"></a>

|环境变量|默认配置|取值范围及解释|
|--|--|--|
|MINDIE_LOG_LEVEL|INFO|统一设置MindIE各组件日志级别。<br>日志级别取值 [CRITICAL, ERROR, WARN, INFO, DEBUG]；取值为 null 时关闭日志。|
|MINDIE_LOG_TO_FILE|true|统一设置MindIE各组件日志是否写入文件。<br>取值范围为：[false, true]，且支持[0, 1]。|
|MINDIE_LOG_TO_STDOUT|true|统一设置MindIE各组件是否打印日志。<br>取值范围为：[false, true]，且支持[0, 1]。|
|MINDIE_LOG_VERBOSE|true|统一设置MindIE各组件日志中是否加入可选日志内容，当前日志分为固定日志内容和可选日志内容。完整调试日志格式：**[date time]** [pid] [tid] [组件名称]**[大写日志级别]** [file:line] : **[error code] [*] log message**；非加粗内容为可选内容，当环境变量设置为true时会加入可选内容。<br>取值范围为：[false, true]，且支持[0, 1]。<br>**日志格式中的[*]表示子组件或更小单位模块的名称，可以选择将其呈现在日志中，方便更好的定位问题。**|
|MINDIE_LOG_PATH|~/mindie/log|统一设置MindIE各组件日志写入文件的保存目录。|
|MINDIE_LOG_ROTATE|<ul><li>s：默认值为 daily / 30</li><li>fs：默认值为20 (MB)</li><li>r：默认值为10</li></ul>|统一设置MindIE各组件日志轮转。<br>设置某个组件的日志轮转格式为：*组件名称* : -s *cycle* -fs *filesize* -r *rotate*<ul><li>如果":"前无组件名称，则默认为对所有组件统一进行设置；</li><li>*cycle*表示时间轮转周期，可取 daily/weekly/monthly/yearly，或1～180的天数，默认 daily、周期为30。</li><li>*filesize*表示每个日志文件大小（单位MB），取值范围 [1, 500]。</li><li>*rotate*表示每个进程可保留的最多日志文件个数，取值范围 [1, 64]。</li></ul>|

## 日志简介

目前MindIE SD的日志仅包含运行调试日志。

### 日志记录格式

当MINDIE_LOG_VERBOSE开启时，MindIE所有组件的完整日志格式如下：

```text
[date time] [pid] [tid] [组件名称] [大写日志级别] [file:line] : [error code] [*] log message
```

当MINDIE_LOG_VERBOSE关闭时，日志仅保留必选内容，格式如下：

```text
[date time] [组件名称] [大写日志级别] log message
```

> [!NOTE]说明
> \*：表示如果组件内有子组件或者更小的功能模块，会在日志信息前进行呈现。错误码与"\*"子组件标识均作为日志消息（log message）的一部分输出。

**表 2** 日志字段说明

|字段|说明|
|--|--|
|**date time**|日期时间。|
|pid|进程号。|
|tid|线程号。|
|组件名称|MindIE的组件名称，MindIE-SD在日志中固定呈现为[MindIE-SD]。|
|**大写日志级别**|日志级别的大写形式，日志级别请参见[表4 日志级别](#table4)。|
|file:line|文件名:代码行号。|
|error code|Error级别日志的错误码，错误码请参见《[MindIE SD错误码参考](error_code.md)》。|
|**log message**|具体错误信息。|

**加粗内容为日志的必选内容**，其余字段为日志的可选信息，可以通过环境变量"MINDIE\_LOG\_VERBOSE"进行配置。具体操作请参见[配置日志内容](#配置日志内容)。

## 查看日志

MindIE默认收集INFO级别及以上的日志，日志文件的默认落盘路径如[表3](#table3)所示。落盘路径的设置可参见[配置日志落盘路径](#配置日志落盘路径)。

**表 3** 日志路径 <a id="table3"></a>

|路径|说明|
|--|--|
|~/mindie/log|默认的日志落盘路径。|
|~/mindie/log/debug|默认日志落盘路径下，自动生成的运行调试日志路径。|

日志文件命名格式统一为：`mindie-sd_进程号_时间戳.log`（时间戳格式为YYYYMMDDHHMMSS，精确到秒）。可以根据进程号和时间戳来定位到相关的日志文件。

【示例1】MindIE SD的日志文件。

```text
mindie-sd_123_20241008020600.log
```

使用如下命令查看日志，需将命令中的日志文件名替换为实际文件名。

【示例2】查看MindIE SD的日志文件。

```bash
cat mindie-sd_123_20241008020600.log
```

## 其他

### 设置日志级别

运行调试日志被分为如[表4](#table4)所示的5个等级。

**表 4** 日志级别 <a id="table4"></a>

|日志级别|简写|日志内容|
|--|--|--|
|CRITICAL|critical|紧急。系统业务严重受损或者完全不可用的紧急情况，规模性的用户受影响，需要运维人员紧急处理。例如系统无法启动或进程挂死等。|
|ERROR|error|错误。系统运行环境/功能受影响，或非预期的数据/事件造成功能执行出错。例如数据入库失败、任务创建失败等。|
|WARNING|warn|警告。系统出现的潜在风险或隐患，但不影响系统功能的正常执行。例如数据校验存在错误，但系统可通过纠错功能恢复，不影响功能的执行。|
|INFO|info|信息。用于系统运行正常的信息记录，输出一些状态或状态变化的信息，例如当前系统的状态、数据库的连接状态等信息。|
|DEBUG|debug|调试。用于跟踪运行路径，如跟踪函数的进入和退出等，记录调试信息。记载的信息全面，是给开发人员用于定位复杂的问题。增加了代码级的信息输出，如当前调用的函数名和参数、内部变量值、函数调用返回值等。抛出异常或者错误返回之前需要记录。|

日志级别等级由低到高顺序：DEBUG < INFO < WARNING < ERROR < CRITICAL，级别越低，输出日志越详细。

通过环境变量"MINDIE\_LOG\_LEVEL"设置各组件日志级别，日志级别默认为"info"。

设置某个组件日志级别的具体格式为：_组件名称_:  _日志级别_。

- 日志级别有以下选项：[critical, error, warn, info, debug]，null 表示关闭对应组件日志。
- 组件名称有以下选项：[motor, server, llm, llmmodels, sd]
- 如果":"前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件日志级别时用";"隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

> [!NOTE]说明
> 以上组件和日志级别的取值不区分大小写。

【示例1】统一将MindIE所有组件的日志级别设成"debug"。

```bash
export MINDIE_LOG_LEVEL="debug"
```

【示例2】除了MindIE SD的日志级别设成"debug"，其余组件的级别都设置为"info"。

```bash
export MINDIE_LOG_LEVEL="info ; sd:debug"
```

### 设置日志展示方式

通过环境变量"MINDIE\_LOG\_TO\_FILE"设置MindIE各组件日志是否写入文件，默认为"true"写入。

通过环境变量"MINDIE\_LOG\_TO\_STDOUT"设置MindIE各组件日志是否打印，默认为"true"打印。

设置某个组件日志是否写入或打印的格式为：_组件名称_: \{0, 1, true, false\}。

- "0"和"false"代表否，"1"和"true"代表是。
- 如果":"前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用";"隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

【示例1】不将MindIE SD的日志写入文件。

```bash
export MINDIE_LOG_TO_FILE="sd: false"
```

【示例2】将MindIE所有组件的日志流打印。

```bash
export MINDIE_LOG_TO_STDOUT="true"
```

### 配置日志落盘路径

通过环境变量"MINDIE\_LOG\_PATH"设置MindIE各组件日志的落盘路径，默认的落盘根目录为"~/mindie/log"，实际日志文件统一写入其下的"debug"子目录（即默认为"~/mindie/log/debug"）。

设置日志落盘路径的格式为：_组件名称_:  _路径_。

- 若路径开头为"/"，则表明该路径为绝对路径，日志写入该路径下的"debug"子目录；
- 若路径开头无"/"，则表明该路径为相对路径，相对于默认根目录"~/mindie/log"，日志写入拼接路径下的"debug"子目录；
- 如果":"前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用";"隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

> [!NOTE]说明
>
> - 路径里不能包含控制字符等特殊字符。
> - 日志路径不能为软链接，程序会对其进行校验，请保证日志路径合理。

【示例1】将MindIE SD的日志落盘到"/home/working/debug"。

```bash
export MINDIE_LOG_PATH="sd: /home/working/"
```

【示例2】将MindIE SD的日志落盘到"~/mindie/log/sd/debug"。

```bash
export MINDIE_LOG_PATH="sd: sd"
```

### 配置日志内容

通过环境变量"MINDIE\_LOG\_VERBOSE"设置某个组件的日志内容中是否打印可选信息，默认为"true"打印可选信息。

设置的格式为：_组件名称_: \{0, 1, true, false\}。

- "0"和"false"代表否，"1"和"true"代表是。
- 如果":"前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用";"隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

【示例1】统一不打印或保存MindIE所有组件的可选日志内容。

```bash
export MINDIE_LOG_VERBOSE="false"
```

【示例2】打印或保存MindIE SD的可选日志内容。

```bash
export MINDIE_LOG_VERBOSE="sd: true"
```

### 配置日志轮转

日志在以下任一条件满足时进行轮转：单个日志文件大小达到设定的文件大小上限；或达到设定的时间周期（默认按天，每天0点）。通过环境变量"MINDIE\_LOG\_ROTATE"设置日志的轮转。

轮转相关参数及默认值、取值范围如下：

- 时间周期：支持按天（daily）、按周（weekly）、按月（monthly）、按年（yearly）轮转，也可设置1～180的天数，默认按天、周期为30。
- 文件大小：每个日志文件大小的取值范围为1MB～500MB，默认为20MB。
- 文件个数：每个进程可保留的最多日志文件个数的取值范围为1个～64个，默认为10个。当历史文件个数超过该值时，最旧的文件将被删除。

设置某个组件的日志轮转格式为：_组件名称_: -s _cycle_ -fs _filesize_ -r _rotate_

- 如果":"前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用";"隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。
- -s _cycle_表示时间周期，可取 daily/weekly/monthly/yearly，或1～180的天数；-fs _filesize_表示每个日志文件大小（单位MB）；-r _rotate_表示每个进程可保留的最多日志文件个数。

    > [!NOTE]说明
    > 轮转后新生成的日志文件命名格式为：`mindie-sd_进程号_时间戳.log`（时间戳精确到秒）。当历史文件个数超出每个进程可保留的最多日志文件个数时，最旧的日志文件会被自动删除。

【示例1】统一将MindIE所有组件日志文件大小设置为500MB。

```bash
export MINDIE_LOG_ROTATE="-fs 500"
```

【示例2】设置MindIE SD日志文件大小不得超过40MB，每个进程保留一个文件。

```bash
export MINDIE_LOG_ROTATE="sd: -fs 40 -r 1"
```
