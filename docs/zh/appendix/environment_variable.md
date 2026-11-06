# 环境变量

本文档介绍环境变量，列举MindIE SD日志级别、日志输出与落盘、日志轮转等配置项的说明、配置方式与缺省值，适用于需要配置MindIE SD环境变量的用户。

**表 1** 环境变量

|环境变量|说明|配置方式|缺省值|
|--|--|--|--|
|MINDIE_LOG_LEVEL|MindIE SD日志级别。|支持配置级别"critical"、"error"、"warn"、"info"、"debug"、"null"，当配置为null的时候，表示日志功能关闭。<br>支持直接配置方式：export MINDIE_LOG_LEVEL="sd:debug"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_LEVEL="debug"。|info|
|MINDIE_LOG_TO_STDOUT|MindIE SD日志打印控制开关。|支持配置"true"、"1"、"false"、"0"，配置为"true"或"1"时表示开启，配置"false"或"0"时表示关闭。<br>支持直接配置方式：export MINDIE_LOG_TO_STDOUT="sd:true"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_TO_STDOUT="true"。|true|
|MINDIE_LOG_TO_FILE|MindIE SD日志文件落盘控制开关。|支持配置"true"，"1"，"false"，"0"，配置为"true"或"1"时表示开启，配置"false"或"0"时表示关闭。<br>支持直接配置方式：export MINDIE_LOG_TO_FILE="sd:true"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_TO_FILE="true"。|true|
|MINDIE_LOG_PATH|MindIE SD日志文件保存路径配置。|支持用户指定日志落盘路径，默认为"~/mindie/log/"，运行日志会保存到指定路径下的"debug"文件夹中。<br>若用户配置输入为相对路径如"./custom_log"，则日志根目录为"~/mindie/log/custom_log"，运行日志文件位于"~/mindie/log/custom_log/debug"下；若用户配置输入为绝对路径如"/home/usr/custom_log"，则日志根目录为"/home/usr/custom_log"，运行日志文件位于"/home/usr/custom_log/debug"下。<br>支持直接配置方式：export MINDIE_LOG_PATH="sd:./custom_log"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_PATH="./custom_log"。|~/mindie/log/|
|MINDIE_LOG_ROTATE|MindIE SD轮转配置。|支持配置轮转周期"-s"，配置范围为["daily", "weekly", "monthly", "yearly"]以及所有正整数，当为正整数时，轮转周期单位为daily，如"-s 100"则代表轮转周期为100天。<br>支持配置日志最大文件大小"-fs"，配置范围为整数，单位为MB，如"-fs 20"代表单个日志文件最大记录大小为20MB。<br>支持配置日志最大文件个数"-r"，配置范围为整数，单位为个，如"-r 10"代表最大文件记录个数为10。<br>可同时配置上述参数，例：export MINDIE_LOG_ROTATE="-s 10 -fs 20 -r 10"|-s 30 -fs 20 -r 10|
|MINDIE_LOG_VERBOSE|MindIE SD日志是否打印可选信息开关。|支持配置"true"，"1"，"false"，"0"，配置为"true"或"1"时表示开启，配置"false"或"0"时表示关闭。<br>支持直接配置方式：export MINDIE_LOG_VERBOSE="sd:true"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_VERBOSE="true"。|true|
|SKIP_ALL_OPS_PLUGIN_BUILD|算子与PyTorch插件缓存复用开关，仅用于CI构建。|配置为"1"时，若算子缓存产物完整，则通过增量构建跳过Kernel编译；若缓存的`libPTAExtensionOPS.so`可被当前PyTorch加载，则跳过插件编译。产物缺失、不完整或插件加载失败时自动回退到全量编译。|0|
