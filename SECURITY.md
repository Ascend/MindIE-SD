# 安全声明

## 安全须知

使用MindIE时，为保证安全，用户应根据自身业务，审视整个系统的网络安全加固措施，按照所在组织的安全策略进行相关配置，包括但不局限于软件版本、口令复杂度要求、安全配置（协议、加密套件、秘钥长度等），权限配置、防火墙设置等。关于更多安全声明与建议可参考[昇腾社区MindIE安全管理与加固](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0041.html)，以社区最新版本为准。

## 运行环境建议

- 为减少潜在的安全风险，建议使用非root、非管理员类型账户执行系统操作，确保只有root才是系统的最高权限用户，确保系统中各系统账号的UID不同，遵循权限最小化原则。
- 定期开展对集群的防病毒扫描，防病毒例行检查会帮助集群免受病毒、恶意代码、间谍软件以及程序侵害，降低系统瘫痪、信息泄露等风险。可以使用业界主流防病毒软件进行防病毒检查。
- 为保证生产环境的安全，降低被攻击的风险，请定期查看[昇腾社区MindIE安全管理与加固](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0041.html)修复漏洞/功能问题。

## 文件权限控制

- 建议用户将主机（包括宿主机）和容器中的umask设置为0027及以上，提高安全性。
- 建议用户对个人隐私数据、商业资产、业务开发相关的各类包含敏感内容的文件做好访问权限控制。例如本项目中安装目录权限管控、数据文件权限管控，设定的权限可参考[A-文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)。
- 禁止使用SetUID或SetGID等特殊权限的shell脚本。
- 禁止使用高危capability的可执行文件。
- 系统中不允许存在无属主的文件。

## 构建安全声明

- 本项目需要自行编译构建出包，编译过程会产生一些中间文件和编译目录，建议用户对这些文件做好权限控制，在构建过程中可根据需要修改构建脚本以避免相关安全风险，并注意构建结果安全。
- 本项目涉及Python whl包安装，为避免其他用户直接访问和修改Python代码引起代码篡改、伪造等风险，建议用户设置Python为仅安装用户可修改和使用。
- 使用Linux自带的ASLR（Address Space Layout Randomization）和KASLR（Kernel Address Space Layout Randomization）机制进行安全编译。
    - ASLR，开启后可以增强漏洞攻击防护能力，开启方式为：
        ```shell
        echo 2 > /proc/sys/kernel/randomize_va_space
        ```
    - KASLR，开启后可以增加针对内核漏洞的攻击难度，开启方式如下所示：
    1. 使用以下示例命令查看内核配置文件。
        ```shell
        vi /boot/config-$(uname -r)
        ```
        如果存在以下行则表示支持KASLR。
        ```shell
        CONFIG_RANDOMIZE_BASE=y
        ```       
    2. 打开配置文件/etc/default/grub，在GRUB_CMDLINE_LINUX_DEFAULT所在行添加kaslr参数，示例如下所示。
        ```shell
        GRUB_CMDLINE_LINUX_DEFAULT="kaslr"
        ```  
    3. 使用以下命令更新grub配置。
        ```shell
        sudo update-grub
        ```  
    4. 使用以下命令重启系统后即开启KASLR功能。
        ```shell
        sudo reboot
        ```  
- 为阻止缓冲区溢出攻击，建议使用ASLR技术，通过对堆、栈、共享库映射等线性区布局的随机化，增加攻击者预测目的地址的难度，防止攻击者直接定位攻击代码位置。该技术可作用于堆、栈、内存映射区（mmap基址、shared libraries、vdso页）。
    1. 确保当前用户拥有“/proc/sys/kernel/randomize_va_space”文件的写权限。
    2. 开启缓冲区溢出安全保护。
        ```shell
        echo 2 >/proc/sys/kernel/randomize_va_space
        ```  


## 数据安全声明

- 本项目会涉及到接收输入、加载模型权重和保存结果数据，部分接口直接或间接使用风险模块pickle，可能存在数据风险，请确保输入数据来源、保存路径地址可信，加载模型权重时，建议使用本地权重。

## 运行安全声明

- 为避免服务和客户端通信过程信息泄露，建议用户启用HTTPS通信并启用双向认证，如果启用，建议对通信认证涉及的证书、私钥、口令等做好安全访问控制。
- MindIE仅提供部分流控能力，且不直接对接公网，建议用户对MindIE流控和公网、局域网隔离做好控制。如可以使用开源软件Nginx进行保障，用户可参照[Nginx官方文档](https://nginx.org/en/docs/)和[昇腾社区Server安全加固](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0068.html)进行Nginx的部署。
- 对于全网侦听的端口和其他端口，如非必要建议关闭。
- 建议用户关闭不安全的服务，如Telnet、FTP等。
- 用户可以根据自身业务，按IP地址限制与服务器的连接速率对系统进行防DoS攻击，方法包括但不限于利用Linux系统自带iptables防火墙进行预防、优化sysctl参数等。
- 本项目默认的Gloo、DataDist和HCCL通信暂不支持TLS认证功能，如有需要，可参考[B-集合通信加固](#B-集合通信加固)。

## 公开接口声明

本项目提供的对外接口均已在资料中公开，建议直接使用资料说明的公开接口，不建议直接调用未明确公开的接口源码。

## 通信矩阵

本项目的通信矩阵，包括产品开放的端口、该端口使用的传输层协议、通过该端口与对端通信的通信网元名称、认证方式、用途等信息说明均已在资料中公开，可参考[昇腾社区MindIE通信矩阵](https://www.hiascend.com/document/detail/zh/mindie/22RC1/ref/commumatrix/Communication0000.html)，以社区最新版本为准。

## 公网地址声明
本项目代码中包含的公网地址声明均已在资料中公开，可参考[昇腾社区MindIE公网URL](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0089.html)，以社区最新版本为准。

## 漏洞机制说明
[漏洞管理](https://gitcode.com/Ascend/community/blob/master/docs/security.md)

## 免责声明

- 本项目仅供调试和开发之用，使用者需自行承担使用风险，并理解以下内容：

  - [X] 数据处理及删除：用户在使用本项目过程中产生的数据（包括但不限于推理结果、日志等）属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。
  - [X] 数据保密与传播：使用者了解并同意不得将通过本项目产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本项目及其开发者概不负责。
  - [X] 用户输入安全性：用户需自行保证输入的命令行、参数和配置文件的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入不当所导致的问题，本项目及其开发者概不负责。

- 免责声明范围：本免责声明适用于所有使用本项目的个人或实体。使用本项目即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本项目。
- 在使用本项目之前，请**谨慎阅读并理解以上免责声明的内容**。对于使用本项目所产生的任何问题或疑问，请及时联系开发者。

## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型           | Linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件（含脚本文件、库文件等）       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）            |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）            |
| 日志文件（记录完毕或者已经归档）        |  440（r--r-----）             |
| 日志文件（正在记录）                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）            |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx-----）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |

### B-集合通信加固

编译和支持安装TLS的PyTorch的操作步骤如下。
- 步骤1 编译PyTorch
    1. 编译PyTorch源码。
        ```shell
        git clone https://github.com/pytorch/pytorch.git --depth=1 -b v2.1.0
        git submodule sync && git submodule update --init --depth=1 --recursive
        ```  

    2. 安装openssl-1.1
        ```shell
        wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz
        tar -xzf openssl-1.1.1w.tar.gz
        cd openssl-1.1.1w
        ./config --prefix=/usr/local/openssl-1.1
        make -j$(nproc)
        sudo make install
        ```  

    3. 导出环境变量
        ```shell
        export OPENSSL_ROOT_DIR=/usr/local/openssl-1.1
        export LD_LIBRARY_PATH=$OPENSSL_ROOT_DIR/lib:$LD_LIBRARY_PATH
        export USE_GLOO=1
        export USE_GLOO_WITH_OPENSSL=1
        ```  

    4. 构建Python包
        ```shell
        python3 setup.py bdist_wheel
        ```  

- 步骤2 安装Pytorch。支持TLS需要安装torch 2.1.0a0+git7bcf7da版本。
    ```shell
    cd dist
    pip install --ignore-installed torch-2.1.0a0+git7bcf7da-cp311-cp311-linux_aarch.whl
    ```  

- 步骤3 编译安装Gloo
    ```shell
    git config --global http.sslVerify false
    git clone https://github.com/pytorch/gloo.git
    mkdir build && cd build
    cmake .. -USE_TCP_OPENSSL_LOAD=ON
    make -j&(nproc)
    sudo make install
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    ```  

- 步骤4 开启GLOO TLS
    ```shell
    export GLOO_DEVICE_TRANSPORT=TCP_TLS
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/path/to/tls_ca/server.key.pem
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/path/to/tls_ca/server.pem
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/path/to/tls_ca/ca.pem
    ```  