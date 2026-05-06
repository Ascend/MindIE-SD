#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Build both Chinese and English documentation in one command."""

import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(DOCS_DIR, "_build")


def build_lang(lang):
    src_dir = os.path.join(DOCS_DIR, lang)
    out_dir = os.path.join(BUILD_DIR, lang, "html")
    env = os.environ.copy()
    env["SPHINX_LANGUAGE"] = lang
    cmd = [
        sys.executable, "-m", "sphinx",
        "-b", "html",
        "-c", DOCS_DIR,
        src_dir,
        out_dir,
    ]
    logging.info(f"\n{'='*60}")
    logging.info(f"Building {lang.upper()} documentation...")
    logging.info(f"{'='*60}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logging.error(f"ERROR: {lang.upper()} build failed with code {result.returncode}")
        sys.exit(result.returncode)
    logging.info(f"{lang.upper()} build succeeded -> {out_dir}")


def create_default_index():
    index_path = os.path.join(BUILD_DIR, "index.html")
    content = """<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url=zh/html/index.html">
  <title>MindIE SD Documentation</title>
</head>
<body>
  <p>Redirecting to <a href="zh/html/index.html">Chinese documentation</a>...</p>
  <p>English version: <a href="en/html/index.html">English documentation</a></p>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"\nDefault index created -> {index_path}")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    build_lang("zh")
    build_lang("en")
    create_default_index()
    logging.info(f"\n{'='*60}")
    logging.info("All builds completed successfully!")
    logging.info(f"  Chinese: {BUILD_DIR}/zh/html/index.html")
    logging.info(f"  English: {BUILD_DIR}/en/html/index.html")
    logging.info(f"  Default: {BUILD_DIR}/index.html (redirects to Chinese)")
    logging.info(f"\nPreview with:")
    logging.info(f"  python -m http.server 8080 --directory docs/_build")
    logging.info(f"  Then open http://localhost:8080")


if __name__ == "__main__":
    main()
