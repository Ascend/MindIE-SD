# Security Statement

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:12:21.650Z pushedAt=2026-06-09T03:08:24.960Z -->

## Precautions

When using MindIE, you should review and reinforce network security measures based on their specific service requirements and organizational security policies. Key configurations include but are not limited to software versions, password complexity requirements, security settings (e.g., protocols, cipher suites, key lengths), access permissions, and firewall rules. For more security declarations and recommendations, refer to [MindIE Security Management and Hardening](https://www.hiascend.com/document/detail/en/mindie/22RC1/envdeployment/instg/mindie_instg_0041.html), and always follow the latest community version.

## Operating Environment Recommendations

- To minimize potential security risks, use non-root, non-admin accounts for system operations. Ensure that only root has the highest system privileges, all system accounts have unique UIDs, and the principle of least privilege is followed.

- Conduct regular antivirus scans on the cluster to protect against viruses, malicious code, spyware, and other threats. Routine scanning reduces risks such as system failure and information leakage. Use mainstream antivirus software for cluster scans.

- To ensure the security of the production environment and reduce the risk of attacks, regularly review [Ascend Community MindIE Security Management and Hardening](https://www.hiascend.com/document/detail/en/mindie/22RC1/envdeployment/instg/mindie_instg_0041.html) for vulnerability and functionality fixes.

## File Permission Control

- It is recommended setting the `umask` value to `0027` or higher for both hosts (including the host machine) and containers to enhance security.

- It is recommended that you properly control access permissions for personal privacy data, commercial assets, and various files containing sensitive content related to business development. For example, for the installation directory permission control and data file permission control in this project, refer to [A-Recommended Maximum Permission Control Values for Files (Folders) in Various Scenarios](#a-recommended-maximum-permission-control-values-for-files-folders-in-various-scenarios) for the permissions to be set.

- Shell scripts with special permissions such as SetUID or SetGID are prohibited.

- Executable files with high-risk capabilities are prohibited.

- Files without an owner are not allowed in the system.

## Build Security Declaration

- This project requires manual compilation and packaging. The build process generates intermediate files and directories. It is recommended that you properly control access to these files, modify build scripts as needed to mitigate security risks, and ensure the security of build artifacts.

- This project requires Python `.whl` packages. To mitigate risks such as code tampering or forgery from unauthorized access or modification of Python code, it is recommended to restrict Python installation to be modifiable and usable only by the installing user.

- Enable security compilation with Linux's built-in Address Space Layout Randomization (ASLR) and Kernel Address Space Layout Randomization (KASLR) mechanisms.

    - ASLR, when enabled, can enhance protection against vulnerability attacks. The enabling method is as follows:

        ```shell
        echo 2 > /proc/sys/kernel/randomize_va_space
        ```

    - KASLR, when enabled, can increase the difficulty of attacks targeting kernel vulnerabilities. The enabling method is as follows:

    1. Use the following example command to view the kernel configuration file.

        ```shell
        vi /boot/config-$(uname -r)
        ```

        If the following line exists, it indicates that KASLR is supported.

        ```shell
        CONFIG_RANDOMIZE_BASE=y
        ```

    2. Open the configuration file `/etc/default/grub`, and add the `kaslr` parameter to the line where `GRUB_CMDLINE_LINUX_DEFAULT` is located, as shown in the following example.

        ```shell
        GRUB_CMDLINE_LINUX_DEFAULT="kaslr"
        ```

    3. Use the following command to update the `grub` configuration.

        ```shell
        sudo update-grub
        ```

    4. After restarting the system using the following command, the KASLR function will be enabled.

        ```shell
        sudo reboot
        ```

- To prevent buffer overflow attacks, it is recommended to use ASLR technology. By randomizing the layout of linear regions such as the heap, stack, and shared library mappings, it increases the difficulty for attackers to predict the target address, preventing them from directly locating the attack code position. This technology can be applied to the heap, stack, and memory mapping areas (mmap base address, shared libraries, vdso pages).

    1. Ensure that the current user has write permission for the `/proc/sys/kernel/randomize_va_space` file.

    2. Enable buffer overflow security protection.

        ```shell
        echo 2 >/proc/sys/kernel/randomize_va_space
        ```

## Data Security Declaration

- This project involves input reception, model weight loading, and result data saving. Some interfaces directly or indirectly use the risky `pickle` module, which may pose data security risks. Ensure that input data sources and save paths are trustworthy. When loading model weights, it is recommended to use local weights.

## Operational Security Declaration

- To prevent information leakage during service-client communication, it is recommended to enable HTTPS with mutual authentication. If enabled, ensure secure access control over the certificates, private keys, and passwords involved in communication authentication.

- MindIE provides only limited flow control and does not directly interface with the public network. It is recommended that you implement additional control measures for MindIE flow control and isolate both public and local networks accordingly. For enhanced protection, you may use open-source software such as Nginx. Refer to the [Nginx Official Documentation](https://nginx.org/en/docs/) and [Ascend Community Server Security Hardening](https://www.hiascend.com/document/detail/en/mindie/22RC1/envdeployment/instg/mindie_instg_0068.html) for Nginx deployment.

- For ports that listen on all networks and other ports, you are advised to close them if they are not necessary.

- You are advised to disable insecure services, such as Telnet and FTP.

- You can limit connection rates by IP address based on  business needs to protect the system from DoS attacks. Methods include but are not limited to using Linux's built-in iptables firewall and optimizing sysctl parameters.

- TBy default, this project's built-in communication backends—Gloo, DataDist, and HCCL—do not support TLS authentication. If needed, refer to [B-Collective Communication Hardening](#b-collective-communication-hardening).

## Public API Declaration

The public interfaces provided by this project are documented. It is recommended to use only the officially documented APIs. Direct invocation of undocumented internal source code is discouraged.

## Communication Matrix

The communication matrix of this project, including information such as the open ports of the product, the transport layer protocols used by these ports, the names of the communication network elements communicating with the peer through these ports, authentication methods, and purposes, has been disclosed in the documentation. Refer to the [Ascend Community MindIE Communication Matrix](https://www.hiascend.com/document/detail/en/mindie/22RC1/ref/commumatrix/Communication0000.html), and always follow the latest version on the community.

## Public Network Address Declaration

All public network address referenced in the project code have been disclosed in the documentation. For the latest version, refer to the official public URLs in  [Ascend Community MindIE Public Network Addresses](https://www.hiascend.com/document/detail/en/mindie/22RC1/envdeployment/instg/mindie_instg_0089.html).

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/Ascend/community/blob/master/docs/security.md)

## Disclaimer

- This project is intended for debugging and development purposes only. Users shall bear the risks of use and understand the following:

  - [X] Data processing and deletion: Data generated by users during the use of this project (including but not limited to inference results, logs, etc.) falls under the user's responsibility. It is recommended that users delete relevant data promptly after use to prevent leakage or unnecessary information disclosure.

  - [X] Data confidentiality and distribution: Users acknowledge and agree not to arbitrarily distribute or disclose data generated by this project. The project and its contributors bear no responsibility for any resulting information or data breaches, or other adverse consequences.

  - [X] User input security: Users shall ensure the security of the command lines, parameters, and configuration files they input, and bear any security risks or losses caused by improper input. This project and its developers shall not be held liable for any issues arising from improper input.

- Scope of Disclaimer: This disclaimer applies to all individuals or entities using this project. By using this project, you acknowledge and accept this statement and assume all risks and responsibilities arising from its use. If you do not agree, please stop using this project immediately.

- Before using this project, please **carefully read and understand the content of the above disclaimer**. For any issues or questions arising from the use of this project, please contact the developer promptly.

## Appendixes

### A-Recommended Maximum Permission Control Values for Files (Folders) in Various Scenarios

| Type           | Linux Permission Reference Maximum Value |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, etc.)       |   550 (r-xr-x---)             |
| Program file directory                      |   550 (r-xr-x---)            |
| Configuration file                          |  640 (rw-r-----)             |
| Configuration file directory                      |   750 (rwxr-x---)            |
| Log file (recording completed or archived)        |  440 (r--r-----)             |
| Log file (currently recording)                |    640 (rw-r-----)           |
| Log file directory                      |   750 (rwxr-x---)            |
| Debug file                         |  640 (rw-r-----)         |
| Debug file directory                     |   750 (rwxr-x---)  |
| Temporary file directory                      |   750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  |   770 (rwxrwx---)    |
| Business data file                      |   640 (rw-r-----)    |
| Business data file directory                  |   750 (rwxr-x---)      |
| Key component, private key, certificate, and ciphertext file directory    |  700 (rwx-----)      |
| Key component, private key, certificate, and encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interface and script            |   500 (r-x------)        |

### B-Collective communication hardening

The procedure to compile and install PyTorch with TLS support is as follows.

- Step 1 Compile PyTorch.

    1. Install openssl-1.1.

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

    3. Export environment variables.

        ```shell
        export OPENSSL_ROOT_DIR=/usr/local/openssl-1.1
        export LD_LIBRARY_PATH=$OPENSSL_ROOT_DIR/lib:$LD_LIBRARY_PATH
        export USE_GLOO=1
        export USE_GLOO_WITH_OPENSSL=1
        ```

    4. Build the Python package.

        ```shell
        python3 setup.py bdist_wheel
        ```

- Step 2 Install PyTorch. TLS support requires torch version 2.1.0a0+git7bcf7da.

    ```shell
    cd dist
    pip install --ignore-installed torch-2.1.0a0+git7bcf7da-cp311-cp311-linux_aarch.whl
    ```

- Step 3 Compile and install Gloo.

    ```shell
    git config --global http.sslVerify false
    git clone https://github.com/pytorch/gloo.git
    mkdir build && cd build
    cmake .. -DUSE_TCP_OPENSSL_LOAD=ON
    make -j$(nproc)
    sudo make install
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    ```

- Step 4 Enable GLOO TLS.

    ```shell
    export GLOO_DEVICE_TRANSPORT=TCP_TLS
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/path/to/tls_ca/server.key.pem
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/path/to/tls_ca/server.pem
    export GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/path/to/tls_ca/ca.pem
    ```
