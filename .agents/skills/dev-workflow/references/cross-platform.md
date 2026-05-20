# 跨平台开发经验

## shell 脚本隔离

由于 Windows PowerShell 不支持 `&&`，通过 SSH 执行复杂命令时，推荐将命令写入 `.sh` 文件后远端执行：

```python
script = "#!/bin/bash\nsource set_env.sh\npython3 script.py\necho EXIT=$?\n"
with sftp.file(remote_path, "w") as f:
    f.write(script)
ssh.exec_command(f"docker exec {container} bash {remote_path}")
```

## 编码问题

Windows 本地 GBK 编码可能与远端 UTF-8 输出冲突。处理方式：

- `errors="replace"` 解码
- 或输出到文件后 `cat` 读取
- 关键行提取替代全文输出

## pip install -e . 新增文件未被索引

`pip install -e .` 在首次安装时扫描包目录并建立索引，后续新增的 `.py` 文件不会自动加入。新增 Python 文件后必须重新执行 `pip install -e .`。

此规则仅适用于 `mindiesd/` 包目录和 `csrc/` 编译源码目录下的文件变更。`examples/`、`tests/`、`docs/` 等非包/非编译目录下的文件变更不需要重新执行 `pip install -e .`。
