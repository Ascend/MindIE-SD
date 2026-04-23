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

import re
import argparse
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# Function to parse the failures summary details
def parse_fail_summary(details, failure_error_count):
    details_list = details.split(', ')
    for item in details_list:
        key, value = item.split('=')
        value = int(value)
        if key == 'failures':
            failure_error_count[0] += value
        elif key == 'errors':
            failure_error_count[1] += value


def file_summary(log_file):
    # Matches summary message: "Ran X tests in Y.YYYs"
    test_summary_re = re.compile(r'Ran (\d+) tests? in ([\d\.]+)s')
    # Matches failure summary message: "FAILED (details)"
    fail_summary_re = re.compile(r'FAILED \((.*?)\)')
    # Matches detailed fail or error message: "ERROR/FAIL: test_name (function)"
    error_fail_re = re.compile(r'(ERROR|FAIL): (\w+) \(([\w\.]+)\)')

    total_tests = 0
    total_time = 0.0
    failure_error_count = [0, 0]
    error_fail_details = []
    with open(log_file, 'r') as file:
        for line in file:
            # Parse test summary
            test_summary_match = test_summary_re.search(line)
            if test_summary_match:
                total_tests += int(test_summary_match.group(1))
                total_time += float(test_summary_match.group(2))
                continue

            # Parse fail summary
            fail_summary_match = fail_summary_re.search(line)
            if fail_summary_match:
                parse_fail_summary(fail_summary_match.group(1), failure_error_count)
                continue

            # Parse detailed error and fail information
            error_fail_match = error_fail_re.search(line)
            if error_fail_match:
                error_fail_details.append(line.strip())

    # Print aggregated results
    summary = (
        f"Total tests run: {total_tests}\n"
        f"Total time: {total_time:.3f}s\n"
        f"Failures: {failure_error_count[0]}\n"
        f"Errors: {failure_error_count[1]}\n"
        "\nDetailed error and fail information:\n"
    )
    summary += "\n".join(error_fail_details)
    logger.info(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize unitest logs.")
    parser.add_argument("file", help="unitest log file")
    args = parser.parse_args()

    file_summary(args.file)

