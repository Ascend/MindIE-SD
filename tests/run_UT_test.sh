#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
set -euo pipefail
current_directory=$(dirname "$(readlink -f "$0")")
if command -v python3 > /dev/null; then
    python_command=python3
else
    python_command=python
fi

# CI removes the final line of this script before running it. Keep the "OK"
# marker on the line above the final line so the CI grep still matches.
"$python_command" "$current_directory/scripts/run_suites.py" --cpu_only "$current_directory/UT" "$@"
echo "UT OK"
# CI deletes this last line (sed -i '$d'); do not move it.
