# 开发者指南

本章节详细展示开发者如何搭建完整的开发环境、验证算子接口/算法的有效性。

## 构建指导——方式一：镜像安装方式

本章节指导开发者进行容器镜像安装。

1. 安装驱动固件

宿主机需要确保安装过NPU驱动和固件。如未安装，请参见[《CANN 软件安装指南》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)中的“选择安装场景”章节或“选择安装场景”章节（社区版），根据安装方式、操作系统、业务场景选择安装场景，选择完成后单击“开始阅读”，按“安装NPU驱动和固件”章节进行安装。

-   安装方式：选择“在物理机上安装”。
-   操作系统：选择使用的操作系统，MindIE支持的操作系统请参考硬件配套和支持的操作系统。
-   业务场景：选择“训练&推理&开发调试”。

用户在宿主机自行安装Docker（版本要求大于或等于24.x.x）。
配置源之前，请确保安装环境能够连接网络。

2. 获取MindIE容器镜像

-   单击[昇腾镜像仓库链接](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)，进入MindIE镜像下载页面。
-   单击页面右上角登录按钮，使用华为账号登录（如果没有请先注册）。
-   在MindIE镜像下载页面的“镜像版本”页签，根据您的设备形态，单击对应镜像后方“操作”栏中的“立即下载”按钮。
-   根据弹出的镜像下载操作指导页面下载镜像，示例如图1所示。

![镜像下载](figures/镜像下载.png)

3. 使用镜像

执行以下命令启动容器，容器启动命令仅供参考，可根据实际情况自行修改容器名称、镜像名称、挂载路径等。
```
docker run -it -d --net=host --shm-size=1g \
    --name <container-name> \
    --device=/dev/davinci_manager:rwm \
    --device=/dev/hisi_hdc:rwm \
    --device=/dev/devmm_svm:rwm \
    --device=/dev/davinci0:rwm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path-to-weights:/path-to-weights:ro \
    mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts bash
```
> **说明：** 
>“mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts”为镜像名称和标签，可根据实际情况修改。可在宿主机执行以下命令查看当前机器上已有的镜像：
>```
>docker images
>```
>对于--device参数，挂载权限设置为rwm，而非权限较小的rw或r，原因如下：
>-   对于Atlas 800I A2 推理服务器，若设置挂载权限为rw，可以正常进入容器，同时也可以使用npu-smi命令查看npu占用信息，并正常运行MindIE业务；但如果挂载的npu（即对应挂载选项中的davincixxx，如npu0对应davinci0）上有其它任务占用，则使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set_device()会失败）。
>-   对于Atlas 800I A3 超节点服务器，若设置挂载权限为rw，进入容器后，使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set_device()会失败）。

执行以下命令进入容器。
```
docker exec -it <container-name> bash
```

4. 安装其他环境依赖

-   使用模型进行推理前需要安装对应的依赖，根据Modelers/Modelzoo仓上模型README，进行相关依赖的安装。
```
pip install -r requirements.txt
```

-   安装gcc、g++

若镜像环境中没有gcc、g++，请用户自行安装，并导入头文件路径:
```
yum install gcc g++ -y
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```

## 构建指导——方式二：物理机安装方式

本章节介绍如何在物理机上搭建完整的开发环境，包含驱动固件、CANN、PyTorch、Torch NPU、MindIE SD（安装包安装&源码编译）安装方式，以及MindIE SD卸载&更新方式。

1. 安装驱动固件

宿主机需要确保安装过NPU驱动和固件。如未安装，请参见[《CANN 软件安装指南》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)中的“选择安装场景”章节或“选择安装场景”章节（社区版），根据安装方式、操作系统、业务场景选择安装场景，选择完成后单击“开始阅读”，按“安装NPU驱动和固件”章节进行安装。

-   安装方式：选择“在物理机上安装”。
-   操作系统：选择使用的操作系统，MindIE支持的操作系统请参考硬件配套和支持的操作系统。
-   业务场景：选择“训练&推理&开发调试”。

用户在宿主机自行安装Docker（版本要求大于或等于24.x.x）。
配置源之前，请确保安装环境能够连接网络。

2. 安装CANN

需要安装的CANN软件包包括：
-   CANN Toolkit开发套件包
-   CANN Kernels算子包

请参见[《CANN 软件安装指南》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)中的“选择安装场景”章节或“选择安装场景”章节（社区版），根据安装方式、操作系统、业务场景选择安装场景，选择完成后单击“开始阅读”，按“安装CANN（物理机场景） \> 安装CANN软件包”章节进行安装。

-   安装方式：选择“在物理机上安装”。
-   操作系统：选择使用的操作系统。
-   业务场景：选择“训练&推理&开发调试”。

3. 安装PyTorch和Torch NPU

需要安装的软件包包括：
-   PyTorch框架whl包（支持版本为：2.1.0）
-   torch_npu插件whl包

请参见《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)”章节安装PyTorch框架。
请参见《Ascend Extension for PyTorch 软件安装指南》中的“[（可选）安装扩展模块](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0008.html)”章节安装torch_npu插件。

> **说明：** 
>若镜像环境中没有gcc、g++，请用户自行安装，并导入头文件路径
>```
>yum install gcc g++ -y
>export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
>```

4. 安装其他环境依赖

-   使用模型进行推理前需要安装对应的依赖，根据Modelers/Modelzoo仓上模型README，进行相关依赖的安装。
```
pip install -r requirements.txt
```

5. 安装MindIE SD

方式一：安装包安装

MindIE SD无需单独安装，安装MindIE时，MindIE SD将自动安装。MindIE软件包安装步骤如下：

1）将获取到的MindIE软件包上传到安装环境任意路径（如/home/package）进入软件包所在路径，增加对软件包的可执行权限
```
cd /home/package
chmod +x Ascend-mindie_<version>_linux-<arch>_<abi>.run
```

2）执行以下命令添加ascend-toolkit包的环境变量（以root用户为例，以下为root用户的默认安装路径）
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3）执行以下命令安装软件（以下命令支持--install-path=<path>等参数，具体参数说明请参见软件包参数说明）
```
./软件包名.run --install --quiet
```

4）输入命令验证是否安装成功
```
python3 -c “import torch, torch_npu, mindiesd; print(torch_npu.npu.is_available())"
```

方式二：源码编译安装

拉取代码：
```
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
python setup.py bdist_wheel
```

> **说明：** 
>若环境中没有wheel等依赖，请用户自行安装
>```
>pip install wheel
>```

-   方式一：常规安装（使用默认版本号）
```
cd dist
pip install mindiesd-*.whl
```

-   方式二：开发者可编辑模式安装（可通过环境变量MINDIE_SD_VERSION_OVERRIDE修改版本号）
```
pip install -e .
```

## 自测方法<a name="section6665639165712"></a>

本章节介绍MindIE SD仓算子自测方式，看护算子精度。

1. 卸载MindIE SD

可通过以下命令卸载MindIE SD：
```
pip uninstall mindiesd
```

2. 运行全量UT测试用例：
```
pip install -r MindIE-SD/requirements.txt
pip install coverage
cd MindIE-SD/tests
bash run_test.sh
```

3. 运行LA单算子精度测试用例

修改MindIE-SD/tests/plugin/la_acc_prof.py文件，选择Option 1或Option 2，通过加载test_la.csv或enumerated_cases.csv文件，测试LA算子在所设置shape下的精度。

-   "./tests/plugin/test_la.csv"：设置了常用SD模型的输入shape
-   "enumerated_cases.csv"：枚举的各种shape

完成修改后运行以下命令：
```
cd MindIE-SD/
python tests/plugin/la_acc_prof.py
```

运行成功后会在MindIE-SD目录下保存两个结果文件acc_output_results_1.csv、acc_output_results_1.csv，记录了la和fascore的相似度，可查看算子在所需shape下的精度。

## 问题定位

1. 环境变量

安装完成后，安装路径下提供进程级环境变量设置脚本“set_env.sh“，以自动完成环境变量设置，该脚本包含如[表1](#table1198934616167)所示中的LD_LIBRARY_PATH和ASCEND_CUSTOM_OPP_PATH，用户进程结束后自动失效。

**表 1**  环境变量1

<a name="table1198934616167"></a>
<table><thead align="left"><tr id="row798920462161"><th class="cellrowborder" valign="top" width="30%" id="mcps1.2.3.1.1"><p id="p1098954601614"><a name="p1098954601614"></a><a name="p1098954601614"></a>环境变量</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.2.3.1.2"><p id="p39891546131618"><a name="p39891546131618"></a><a name="p39891546131618"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row134754515181"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1347510512186"><a name="p1347510512186"></a><a name="p1347510512186"></a>LD_LIBRARY_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p1347545161816"><a name="p1347545161816"></a><a name="p1347545161816"></a>动态库的查找路径。</p>
</td>
</tr>
<tr id="row8304021181812"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p730422151813"><a name="p730422151813"></a><a name="p730422151813"></a>ASCEND_CUSTOM_OPP_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p5304132113187"><a name="p5304132113187"></a><a name="p5304132113187"></a>推理引擎自定义算子包安装路径。</p>
</td>
</tr>
<tr id="row188531382187"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p18531838101815"><a name="p18531838101815"></a><a name="p18531838101815"></a>ASCEND_RT_VISIBLE_DEVICES</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p45231032112017"><a name="p45231032112017"></a><a name="p45231032112017"></a>指定当前进程所用的昇腾AI处理器的逻辑ID，如有需要请自行配置。</p>
<p id="p128531638141814"><a name="p128531638141814"></a><a name="p128531638141814"></a>配置示例："0,1,2"或"0-2"；昇腾AI处理器的逻辑ID间使用“,”表示分割，使用“-”表示连续。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  环境变量2

<a name="table1980612381539"></a>
<table><thead align="left"><tr id="row10807938155315"><th class="cellrowborder" valign="top" width="19.980000000000004%" id="mcps1.2.5.1.1"><p id="zh-cn_topic_0000001678206378_p82134174820"><a name="zh-cn_topic_0000001678206378_p82134174820"></a><a name="zh-cn_topic_0000001678206378_p82134174820"></a>环境变量名</p>
</th>
<th class="cellrowborder" valign="top" width="20.540000000000003%" id="mcps1.2.5.1.2"><p id="zh-cn_topic_0000001678206378_p11213161134818"><a name="zh-cn_topic_0000001678206378_p11213161134818"></a><a name="zh-cn_topic_0000001678206378_p11213161134818"></a>说明</p>
</th>
<th class="cellrowborder" valign="top" width="47.040000000000006%" id="mcps1.2.5.1.3"><p id="p149937434118"><a name="p149937434118"></a><a name="p149937434118"></a>配置方式</p>
</th>
<th class="cellrowborder" valign="top" width="12.440000000000001%" id="mcps1.2.5.1.4"><p id="p1671518914117"><a name="p1671518914117"></a><a name="p1671518914117"></a>缺省值</p>
</th>
</tr>
</thead>
<tbody><tr id="row13807113810533"><td class="cellrowborder" valign="top" width="19.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p28071038125314"><a name="p28071038125314"></a><a name="p28071038125314"></a>MINDIE_LOG_LEVEL</p>
</td>
<td class="cellrowborder" valign="top" width="20.540000000000003%" headers="mcps1.2.5.1.2 "><p id="p1180763825317"><a name="p1180763825317"></a><a name="p1180763825317"></a><span id="ph850915471163"><a name="ph850915471163"></a><a name="ph850915471163"></a>MindIE SD</span>日志级别。</p>
</td>
<td class="cellrowborder" valign="top" width="47.040000000000006%" headers="mcps1.2.5.1.3 "><p id="p116433113113"><a name="p116433113113"></a><a name="p116433113113"></a>支持配置级别"critical，error，warn，info，debug，null"，当配置为null的时候，表示日志功能关闭。</p>
<p id="p198054683413"><a name="p198054683413"></a><a name="p198054683413"></a>支持直接配置方式：export MINDIE_LOG_LEVEL="sd:debug"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_LEVEL="debug"。</p>
</td>
<td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.5.1.4 "><p id="p6807103825317"><a name="p6807103825317"></a><a name="p6807103825317"></a>info</p>
</td>
</tr>
<tr id="row98078382532"><td class="cellrowborder" valign="top" width="19.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p2080793895320"><a name="p2080793895320"></a><a name="p2080793895320"></a>MINDIE_LOG_TO_STDOUT</p>
</td>
<td class="cellrowborder" valign="top" width="20.540000000000003%" headers="mcps1.2.5.1.2 "><p id="p88082385533"><a name="p88082385533"></a><a name="p88082385533"></a><span id="ph760625212016"><a name="ph760625212016"></a><a name="ph760625212016"></a>MindIE SD</span>日志打印控制开关。</p>
</td>
<td class="cellrowborder" valign="top" width="47.040000000000006%" headers="mcps1.2.5.1.3 "><p id="p18191122943816"><a name="p18191122943816"></a><a name="p18191122943816"></a>支持配置"true"，"1"，"false"，"0"，配置为"true"或"1"时表示开启，配置"false"或"0"时表示关闭。</p>
<p id="p4548114734216"><a name="p4548114734216"></a><a name="p4548114734216"></a>支持直接配置方式：export MINDIE_LOG_TO_STDOUT="sd:true"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_TO_STDOUT="true"。</p>
</td>
<td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.5.1.4 "><p id="p1808138125312"><a name="p1808138125312"></a><a name="p1808138125312"></a>true</p>
</td>
</tr>
<tr id="row680816388535"><td class="cellrowborder" valign="top" width="19.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p1780873835317"><a name="p1780873835317"></a><a name="p1780873835317"></a>MINDIE_LOG_TO_FILE</p>
</td>
<td class="cellrowborder" valign="top" width="20.540000000000003%" headers="mcps1.2.5.1.2 "><p id="p280810380531"><a name="p280810380531"></a><a name="p280810380531"></a><span id="ph32461853172013"><a name="ph32461853172013"></a><a name="ph32461853172013"></a>MindIE SD</span>日志文件落盘控制开关。</p>
</td>
<td class="cellrowborder" valign="top" width="47.040000000000006%" headers="mcps1.2.5.1.3 "><p id="p219315394616"><a name="p219315394616"></a><a name="p219315394616"></a>支持配置"true"，"1"，"false"，"0"，配置为"true"或"1"时表示开启，配置"false"或"0"时表示关闭。</p>
<p id="p96911410204611"><a name="p96911410204611"></a><a name="p96911410204611"></a>支持直接配置方式：export MINDIE_LOG_TO_FILE="sd:true"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_TO_FILE="true"。</p>
</td>
<td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.5.1.4 "><p id="p031554115465"><a name="p031554115465"></a><a name="p031554115465"></a>true</p>
</td>
</tr>
<tr id="row10808038105310"><td class="cellrowborder" valign="top" width="19.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p1980873818535"><a name="p1980873818535"></a><a name="p1980873818535"></a>MINDIE_LOG_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="20.540000000000003%" headers="mcps1.2.5.1.2 "><p id="p18808838145317"><a name="p18808838145317"></a><a name="p18808838145317"></a><span id="ph11941554192012"><a name="ph11941554192012"></a><a name="ph11941554192012"></a>MindIE SD</span>日志文件保存路径配置。</p>
</td>
<td class="cellrowborder" valign="top" width="47.040000000000006%" headers="mcps1.2.5.1.3 "><p id="p1475116313499"><a name="p1475116313499"></a><a name="p1475116313499"></a>支持用户指定日志落盘路径，默认为"~/mindie/log/"，其中运行日志会保存到指定路径的"debug"文件夹下。</p>
<p id="p16808153815316"><a name="p16808153815316"></a><a name="p16808153815316"></a>若用户配置输入为相对路径如"./custom_log"，则日志文件会写入"~/mindie/log/custom_log"下；若用户配置输入为绝对路径如"/home/usr/custom_log"，则日志文件会写入"/home/usr/custom_log"下。</p>
<p id="p185163035212"><a name="p185163035212"></a><a name="p185163035212"></a>支持直接配置方式：export MINDIE_LOG_PATH="sd:./custom_log"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_PATH="./custom_log"。</p>
</td>
<td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.5.1.4 "><p id="p3808738105315"><a name="p3808738105315"></a><a name="p3808738105315"></a>~/mindie/log/</p>
</td>
</tr>
<tr id="row323112819554"><td class="cellrowborder" valign="top" width="19.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p1223111819558"><a name="p1223111819558"></a><a name="p1223111819558"></a>MINDIE_LOG_ROTATE</p>
</td>
<td class="cellrowborder" valign="top" width="20.540000000000003%" headers="mcps1.2.5.1.2 "><p id="p192311487556"><a name="p192311487556"></a><a name="p192311487556"></a><span id="ph1895895412206"><a name="ph1895895412206"></a><a name="ph1895895412206"></a>MindIE SD</span>轮转配置。</p>
</td>
<td class="cellrowborder" valign="top" width="47.040000000000006%" headers="mcps1.2.5.1.3 "><p id="p1263616339533"><a name="p1263616339533"></a><a name="p1263616339533"></a>支持配置轮转周期"-s"，配置范围为["daily", "weekly", "monthly","yearly"]以及所有正整数，当为正整数时，轮转周期单位为daily，如"-s 100"则代表轮转周期为100天。</p>
<p id="p1328573525513"><a name="p1328573525513"></a><a name="p1328573525513"></a>支持配置日志最大文件大小"-fs"，配置范围为整数，单位为MB，如"-fs 20"代表单个日志文件最大记录大小为20MB。</p>
<p id="p126561339133419"><a name="p126561339133419"></a><a name="p126561339133419"></a>支持配置日志最大文件个数"-r"，配置范围为整数，单位为个，如"-r 10"代表最大文件记录个数为10。</p>
<p id="p131101612123613"><a name="p131101612123613"></a><a name="p131101612123613"></a>可同时配置上述参数，例：export MINDIE_LOG_ROTATE="-s 10 -fs 20 -r 10"</p>
</td>
<td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.5.1.4 "><p id="p1323113885511"><a name="p1323113885511"></a><a name="p1323113885511"></a>-s 30 -fs 20 -r 10</p>
</td>
</tr>
<tr id="row936202113201"><td class="cellrowborder" valign="top" width="19.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p1036182152014"><a name="p1036182152014"></a><a name="p1036182152014"></a>MINDIE_LOG_VERBOSE</p>
</td>
<td class="cellrowborder" valign="top" width="20.540000000000003%" headers="mcps1.2.5.1.2 "><p id="p53612218203"><a name="p53612218203"></a><a name="p53612218203"></a><span id="ph20662621122810"><a name="ph20662621122810"></a><a name="ph20662621122810"></a>MindIE SD</span>日志是否打印可选信息开关。</p>
</td>
<td class="cellrowborder" valign="top" width="47.040000000000006%" headers="mcps1.2.5.1.3 "><p id="p12361021182012"><a name="p12361021182012"></a><a name="p12361021182012"></a>支持配置"true"，"1"，"false"，"0"，配置为"true"或"1"时表示开启，配置"false"或"0"时表示关闭。</p>
<p id="p3780191812134"><a name="p3780191812134"></a><a name="p3780191812134"></a>支持直接配置方式：export MINDIE_LOG_VERBOSE="sd:true"，其中'sd:'代表指定SD组件，也可省略直接配置export MINDIE_LOG_VERBOSE="true"。</p>
</td>
<td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.5.1.4 "><p id="p1136152152012"><a name="p1136152152012"></a><a name="p1136152152012"></a>true</p>
</td>
</tr>
</tbody>
</table>


2. 常见异常说明

在使用MindIE SD进行推理业务时，模型相关文件（权重、配置、模型代码等）的安全性需由用户保证，常见的异常如下：
- 如用户在模型初始化的时候，用户修改模型配置的默认参数可能会影响模型接口，若权重文件过大或配置文件中参数过大，可能会导致由out of memory导致的报错信息，例如："RuntimeError: NPU out of memory. Tried to allocate xxx GiB."。
- 使用模型推理时，模型内部会进行推理计算，若用户输入的tensor shape过大，可能会出现由out of memory导致的报错信息，例如："RuntimeError: NPU out of memory. Tried to allocate xxx GiB."。
- 在使用MindIE SD进行生成业务时，如果出现错误输入、环境不适配等问题时，代码中会抛出异常，需要用户在上层进行异常捕获处理，常见的异常类型见表格：

<a name="table1198934616167"></a>
<table><thead align="left"><tr id="row798920462161"><th class="cellrowborder" valign="top" width="30%" id="mcps1.2.3.1.1"><p id="p1098954601614"><a name="p1098954601614"></a><a name="p1098954601614"></a>异常类型</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.2.3.1.2"><p id="p39891546131618"><a name="p39891546131618"></a><a name="p39891546131618"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19898461167"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p898964618167"><a name="p898964618167"></a><a name="p898964618167"></a>ZeroDivisionError</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p169891946101610"><a name="p169891946101610"></a><a name="p169891946101610"></a>除0异常。
</td>
</tr>
<tr id="row134754515181"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1347510512186"><a name="p1347510512186"></a><a name="p1347510512186"></a>ValueError
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p1347545161816"><a name="p1347545161816"></a><a name="p1347545161816"></a>参数值异常。
</td>
</tr>
</tbody>
</table>


3. 文件目录权限说明

MindIE SD API会对传入的文件或文件夹做权限安全校验，常见文件与文件夹类型及权限要求说明如下：

<a name="table1198934616167"></a>
<table><thead align="left"><tr id="row798920462161"><th class="cellrowborder" valign="top" width="30%" id="mcps1.2.3.1.1"><p id="p1098954601614"><a name="p1098954601614"></a><a name="p1098954601614"></a>文件</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.2.3.1.2"><p id="p39891546131618"><a name="p39891546131618"></a><a name="p39891546131618"></a>权限要求
</th>
</tr>
</thead>
<tbody><tr id="row19898461167"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p898964618167"><a name="p898964618167"></a><a name="p898964618167"></a>Config文件
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p169891946101610"><a name="p169891946101610"></a><a name="p169891946101610"></a>对应三组权限不得超过640且需要和执行用户的所需组和权限保持一致。
</td>
</tr>
<tr id="row134754515181"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1347510512186"><a name="p1347510512186"></a><a name="p1347510512186"></a>模型权重文件
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p1347545161816"><a name="p1347545161816"></a><a name="p1347545161816"></a>对应三组权限不得超过640且需要和执行用户的所需组和权限保持一致。
</td>
</tr>
<tr id="row134754515181"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1347510512186"><a name="p1347510512186"></a><a name="p1347510512186"></a>模型权重文件夹
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p1347545161816"><a name="p1347545161816"></a><a name="p1347545161816"></a>对应三组权限不得超过750和执行用户的所需组和权限保持一致
</td>
</tr>
</tbody>
</table>
