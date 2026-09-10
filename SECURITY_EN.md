# Security Declaration

## Security Precautions

When using MindIE, to ensure security, users should review the network security hardening measures of the entire system based on their own business requirements and perform relevant configurations according to the security policies of their organization, including but not limited to software versions, password complexity requirements, security configurations (protocols, cipher suites, key lengths, etc.), permission configurations, and firewall settings. For more security declarations and recommendations, refer to [MindIE Security Management and Hardening on Ascend Community](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0041.html), subject to the latest community version.

## Operating Environment Recommendations

- To reduce potential security risks, it is recommended to use non-root, non-administrator accounts to perform system operations, ensure that only root is the highest-privilege user of the system, ensure that the UIDs of all system accounts are different, and follow the principle of least privilege.

- Perform regular antivirus scans on the cluster. Routine antivirus checks help protect the cluster from viruses, malicious code, spyware, and programs, reducing risks such as system crashes and information leakage. You can use mainstream antivirus software in the industry for antivirus checks.

- To ensure the security of the production environment and reduce the risk of attacks, regularly check [Ascend Community MindIE Security Management and Hardening](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0041.html) to fix vulnerabilities/functional issues.

## File Permission Control

- It is recommended that users set umask to 0027 or higher on hosts (including the host machine) and in containers to improve security.

- It is recommended that users properly control access permissions for files containing sensitive content, such as personal privacy data, commercial assets, and business development-related materials. For example, for the installation directory permission control and data file permission control in this project, refer to [A-Recommended Maximum Permission Control Values for Files (Folders) in Each Scenario](#a-recommended-maximum-permission-control-values-for-files-folders-in-each-scenario) for the recommended permissions.

- The use of shell scripts with special permissions such as SetUID or SetGID is prohibited.

- The use of executable files with high-risk capabilities is prohibited.

- Files without an owner are not allowed in the system.

## Build Security Declaration

- This project requires self-compilation and building to produce packages. The compilation process generates some intermediate files and build directories. It is recommended that users properly control permissions on these files, modify build scripts as needed during the build process to avoid related security risks, and pay attention to the security of build results.

- This project involves Python whl package installation. To prevent risks such as code tampering and forgery caused by other users directly accessing and modifying Python code, it is recommended that users configure Python so that only the installing user can modify and use it.

- Use the ASLR (Address Space Layout Randomization) and KASLR (Kernel Address Space Layout Randomization) mechanisms built into Linux for secure compilation.

    - ASLR, when enabled, enhances vulnerability attack protection. It is enabled as follows:

        ```shell
        echo 2 > /proc/sys/kernel/randomize_va_space
        ```

    - KASLR, when enabled, increases the difficulty of attacks targeting kernel vulnerabilities. It is enabled as follows:

    1. Use the following example command to view the kernel configuration file.

        ```shell
        vi /boot/config-$(uname -r)
        ```

        If the following line exists, KASLR is supported.

        ```shell
        CONFIG_RANDOMIZE_BASE=y
        ```

    2. Open the configuration file /etc/default/grub and add the kaslr parameter to the line containing GRUB_CMDLINE_LINUX_DEFAULT, as shown in the following example.

        ```shell
        GRUB_CMDLINE_LINUX_DEFAULT="kaslr"
        ```

    3. Use the following command to update the grub configuration.

        ```shell
        sudo update-grub
        ```

    4. After the system is restarted using the following command, the KASLR feature is enabled.

        ```shell
        sudo reboot
        ```

- To prevent buffer overflow attacks, it is recommended to use the ASLR technique, which randomizes the layout of linear regions such as the heap, stack, and shared library mappings, thereby increasing the difficulty for attackers to predict target addresses and preventing attackers from directly locating attack code positions. This technique can be applied to the heap, stack, and memory mapping regions (mmap base address, shared libraries, and vdso pages).

    1. Ensure that the current user has write permission on the "/proc/sys/kernel/randomize_va_space" file.

    2. Enable buffer overflow security protection.

        ```shell
        echo 2 >/proc/sys/kernel/randomize_va_space
        ```

## Data Security Declaration

- This project involves receiving input, loading model weights, and saving result data. Some interfaces directly or indirectly use the risky module pickle, which may pose data risks. Ensure that the input data source and the save path address are trusted. When loading model weights, it is recommended to use local weights.

## Runtime Security Declaration

- To prevent information leakage during communication between the service and clients, users are advised to enable HTTPS communication and mutual authentication. If enabled, it is recommended to implement proper security access control for the certificates, private keys, and passwords involved in communication authentication.

- MindIE provides only partial flow control capabilities and does not directly connect to the public network. Users are advised to properly control MindIE flow control and isolate it from the public network and LAN. For example, the open-source software Nginx can be used for protection. Users can refer to the [Nginx Official Documentation](https://nginx.org/en/docs/) and [Ascend Community Server Security Hardening](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0068.html) for Nginx deployment.

- For ports listening on all network interfaces and other ports, it is recommended to close them if not necessary.

- Users are advised to disable insecure services, such as Telnet and FTP.

- Users can protect the system against DoS attacks by limiting the connection rate to the server based on IP addresses according to their own business needs. Methods include, but are not limited to, using the built-in iptables firewall of the Linux system for prevention and optimizing sysctl parameters.

- The default Gloo, DataDist, and HCCL communication in this project does not currently support TLS authentication. If required, refer to [B-Collective Communication Hardening](#b-collective-communication-hardening).

## Public Interface Declaration

All external interfaces provided by this project have been disclosed in the documentation. It is recommended to directly use the public interfaces described in the documentation, and it is not recommended to directly call the source code of interfaces that are not explicitly disclosed.

## Communication Matrix

The communication matrix of this project, including the ports opened by the product, the transport layer protocols used by these ports, the names of the communication network elements that communicate with the peer through these ports, the authentication methods, and the purposes, has been disclosed in the documentation. For details, see [Ascend Community MindIE Communication Matrix](https://www.hiascend.com/document/detail/zh/mindie/22RC1/ref/commumatrix/Communication0000.html). The latest version on the community prevails.

## Public Network Address Declaration

The public network address declarations contained in the source code of this project have been disclosed in the documentation. For details, see [MindIE Public Network URLs on Ascend Community](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0089.html). The latest version on the community prevails.

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/Ascend/community/blob/master/docs/security.md)

## Disclaimer

- This project is intended solely for debugging and development purposes. Users shall bear the risks of use at their own discretion and understand the following:

  - [X] Data processing and deletion: Data generated by users during the use of this project (including but not limited to inference results, logs, etc.) falls within the scope of user responsibility. Users are advised to promptly delete relevant data after use to prevent leakage or unnecessary information disclosure.

  - [X] Data confidentiality and dissemination: Users understand and agree that data generated through this project shall not be arbitrarily distributed or disseminated. This project and its developers shall not be held responsible for any information leakage, data leakage, or other adverse consequences arising there from.

  - [X] User input security: Users shall ensure the security of the command lines, parameters, and configuration files they input, and shall bear any security risks or losses arising from improper input. This project and its developers shall not be held responsible for any issues caused by improper input.

- Scope of the disclaimer: This disclaimer applies to all individuals or entities using this project. By using this project, you agree to and accept the contents of this statement and are willing to bear the risks and responsibilities arising from the use of this feature. If you have any objections, please discontinue use of this project.

- Before using this project, please **carefully read and understand the contents of the above disclaimer**. For any issues or questions arising from the use of this project, please contact the developer in a timely manner.

## Appendix

### A-Recommended Maximum Permission Control Values for Files (Folders) in Each Scenario

| Type           | Linux Permission Reference Maximum Value |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, etc.)       |   550 (r-xr-x---)             |
| Program file directory                      |   550 (r-xr-x---)            |
| Configuration file                          |  640 (rw-r-----)             |
| Configuration file directory                      |   750 (rwxr-x---)            |
| Log file (recording completed or archived)        |  440 (r--r-----)             |
| Log file (being recorded)                |    640 (rw-r-----)           |
| Log file directory                      |   750 (rwxr-x---)            |
| Debug file                         |  640 (rw-r-----)         |
| Debug file directory                     |   750 (rwxr-x---)  |
| Temporary file directory                      |   750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  |   770 (rwxrwx---)    |
| Service data file                      |   640 (rw-r-----)    |
| Service data file directory                  |   750 (rwxr-x---)      |
| Key component, private key, certificate, and ciphertext file directory    |  700 (rwx-----)      |
| Key component, private key, certificate, and encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interface and encryption/decryption script            |   500 (r-x------)        |

### B-Collective Communication Hardening

The steps for compiling and installing PyTorch with TLS support are as follows.

- Step 1 Compile PyTorch

    1. Install openssl-1.1

        ```shell
        wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz
        tar -xzf openssl-1.1.1w.tar.gz
        cd openssl-1.1.1w
        ./config --prefix=/usr/local/openssl-1.1
        make -j$(nproc)
        sudo make install
        cd ..
        ```

    2. Obtain the PyTorch source code (clone the repository and initialize submodules).

        ```shell
        git clone https://github.com/pytorch/pytorch.git --depth=1 -b v2.9.0
        cd pytorch
        git submodule sync && git submodule update --init --depth=1 --recursive
        ```

    3. Export environment variables

        ```shell
        export OPENSSL_ROOT_DIR=/usr/local/openssl-1.1
        export LD_LIBRARY_PATH=$OPENSSL_ROOT_DIR/lib:$LD_LIBRARY_PATH
        export USE_GLOO=1
        export USE_GLOO_WITH_OPENSSL=1
        ```

    4. Build the Python package

        ```shell
        python3 setup.py bdist_wheel
        ```

- Step 2 Install PyTorch. TLS support requires installing torch version 2.1.0a0+git7bcf7da.

    ```shell
    cd dist
    pip install --ignore-installed torch-2.1.0a0+git7bcf7da-cp311-cp311-linux_aarch.whl
    ```

- Step 3 Compile and install Gloo

    ```shell
    git config --global http.sslVerify false
    git clone https://github.com/pytorch/gloo.git
    mkdir build && cd build
    cmake .. -DUSE_TCP_OPENSSL_LOAD=ON
    make -j$(nproc)
    sudo make install
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    ```

- Step 4 Enable GLOO TLS

    ```shell
    export GLOO_DEVICE_TRANSPORT=TCP_TLS
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/path/to/tls_ca/server.key.pem
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/path/to/tls_ca/server.pem
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/path/to/tls_ca/ca.pem
    ```
