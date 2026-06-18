#!/bin/bash
# ============================================================================
# banner.sh - Container startup information display
# ============================================================================
# This script is called via ENTRYPOINT to display software version and
# copyright information when the container starts.
# ============================================================================

cat <<EOF
Version: ${ASCEND_SOFTWARE_VERSION:-v3.0.0}
Container image Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

This container image and its contents are governed by the Huawei Container License Agreement ("License"). By pulling and using the container, you accept the terms and conditions of this License.
A copy of this License is made available in this container at: https://www.hiascend.com/en/legal/ascendhub-download

Note: You agree and undertake that when using Huawei or third-party software in this image, you will comply with the license agreement of the corresponding Huawei or third-party software.
EOF

exec "$@"
