#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Sphinx configuration for MindIE SD documentation."""

import os
import shutil

PROJECT = "MindIE SD"
COPYRIGHT_TEXT = "2024-2026, Huawei Technologies Co., Ltd."
AUTHOR = "Huawei Technologies Co., Ltd."
EXTENSIONS = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

SOURCE_SUFFIX = {
    ".md": "markdown",
}

EXCLUDE_PATTERNS = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

TEMPLATES_PATH = ["_templates"]
HTML_STATIC_PATH = ["_static"]

MYST_ENABLE_EXTENSIONS = [
    "colon_fence",
    "deflist",
]
MYST_HEADING_ANCHORS = 3
SUPPRESS_WARNINGS = [
    "image.not_readable",
    "myst.header",
    "myst.xref_missing",
    "toc.not_included",
    "toc.not_readable",
]

HTML_THEME = "sphinx_book_theme"
HTML_TITLE = "MindIE SD Documentation"
HTML_THEME_OPTIONS = {
    "repository_url": "https://gitcode.com/Ascend/MindIE-SD",
    "repository_provider": "gitlab",
    "use_repository_button": True,
    "path_to_docs": "docs",
    "article_header_end": ["language-switcher.html", "article-header-buttons.html"],
}

_current_lang = os.environ.get("SPHINX_LANGUAGE", "en")

if _current_lang == "zh":
    language = "zh_CN"
    HTML_TITLE = "MindIE SD \u6587\u6863"
else:
    language = "en"

HTML_CONTEXT = {
    "current_language": _current_lang,
    "current_language_name": "\u4e2d\u6587" if _current_lang == "zh" else "English",
    "language_switcher_items": [
        {
            "name": "English",
            "url": "../en/html/index.html" if _current_lang == "zh" else "index.html",
            "lang_path": "en",
        },
        {
            "name": "\u4e2d\u6587",
            "url": "../zh/html/index.html" if _current_lang == "en" else "index.html",
            "lang_path": "zh",
        },
    ],
}


def _inject_language_switcher(app, pagename, templatename, context, doctree):
    context["current_language_name"] = HTML_CONTEXT["current_language_name"]
    context["language_switcher_items"] = HTML_CONTEXT["language_switcher_items"]


def _copy_shared_assets(app, exc):
    srcdir = app.srcdir
    if os.path.basename(srcdir) in ("zh", "en"):
        docs_dir = os.path.dirname(srcdir)
    else:
        docs_dir = srcdir
    outdir = app.outdir
    for asset in ["figures", "tech_report"]:
        src = os.path.join(docs_dir, asset)
        dst = os.path.join(outdir, asset)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)


def setup(app):
    app.connect("html-page-context", _inject_language_switcher)
    app.connect("build-finished", _copy_shared_assets)


globals().update(
    {
        "project": PROJECT,
        "copyright": COPYRIGHT_TEXT,
        "author": AUTHOR,
        "language": language,
        "extensions": EXTENSIONS,
        "source_suffix": SOURCE_SUFFIX,
        "exclude_patterns": EXCLUDE_PATTERNS,
        "templates_path": TEMPLATES_PATH,
        "html_static_path": HTML_STATIC_PATH,
        "html_context": HTML_CONTEXT,
        "myst_enable_extensions": MYST_ENABLE_EXTENSIONS,
        "myst_heading_anchors": MYST_HEADING_ANCHORS,
        "suppress_warnings": SUPPRESS_WARNINGS,
        "html_theme": HTML_THEME,
        "html_title": HTML_TITLE,
        "html_theme_options": HTML_THEME_OPTIONS,
    }
)
