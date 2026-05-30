#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""
验证评估结果是否符合 performance-evaluation skill 规范
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_config_json(config_path):
    """验证config.json是否符合规范"""
    errors = []
    warnings = []

    with open(config_path, 'r', encoding="utf-8") as f:
        config = json.load(f)

    # 必需字段检查
    required_fields = ['config_name', 'device', 'model_path', 'height', 'width', 'quantization', 'seq_len', 'timestamp']

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # 用户指定参数记录检查
    if 'user_specified' not in config:
        warnings.append("Missing 'user_specified' field - should record user-specified parameters")
    else:
        user_specified = config['user_specified']
        if 'resolution' not in user_specified:
            warnings.append("user_specified should include 'resolution'")
        if 'device' not in user_specified:
            warnings.append("user_specified should include 'device'")
        if 'quantization' not in user_specified:
            warnings.append("user_specified should include 'quantization'")

    # seq_len默认值检查
    if config.get('seq_len') != 64:
        warnings.append(f"seq_len is {config.get('seq_len')}, recommended default is 64")

    return errors, warnings


def validate_iteration_log(log_path):
    """验证iteration日志是否符合规范"""
    errors = []
    warnings = []

    with open(log_path, 'r', encoding="utf-8") as f:
        content = f.read()

    # 必需内容检查
    required_sections = ['Evaluation Configuration:', 'Device:', 'Model:', 'Resolution:', 'Quantization:', 'Summary:']

    for section in required_sections:
        if section not in content:
            errors.append(f"Missing required section: {section}")

    # seq_len检查
    if 'Sequence Length: 64' not in content:
        if 'Sequence Length:' in content:
            warnings.append("seq_len is not 64 (recommended default)")
        else:
            errors.append("Missing 'Sequence Length' field")

    # 算子分析表检查
    if 'FlashAttention' not in content or 'MatMul' not in content:
        warnings.append("Missing operator breakdown table")

    return errors, warnings


def validate_results_directory(dir_path):
    """验证整个results目录结构"""
    results_path = Path(dir_path)

    if not results_path.exists():
        logger.error("Results directory not found: %s", dir_path)
        return False

    all_valid = True
    sep = "=" * 60

    logger.info("\n%s", sep)
    logger.info("Validating: %s", dir_path)
    logger.info("%s\n", sep)

    # 检查每个配置目录
    for config_dir in results_path.iterdir():
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        logger.info("\n📁 Config: %s", config_name)
        logger.info("-" * 40)

        # 检查config.json
        config_file = config_dir / 'config.json'
        if config_file.exists():
            errors, warnings = validate_config_json(config_file)
            if errors:
                logger.info("  ❌ config.json errors:")
                for e in errors:
                    logger.info("     - %s", e)
                all_valid = False
            else:
                logger.info("  ✅ config.json: valid")

            if warnings:
                logger.info("  ⚠️  config.json warnings:")
                for w in warnings:
                    logger.info("     - %s", w)
        else:
            logger.info("  ❌ Missing config.json")
            all_valid = False

        # 检查iteration日志
        log_files = list(config_dir.glob('iteration_*.log'))
        if log_files:
            for log_file in sorted(log_files):
                errors, warnings = validate_iteration_log(log_file)
                if errors:
                    logger.info("  ❌ %s errors:", log_file.name)
                    for e in errors:
                        logger.info("     - %s", e)
                    all_valid = False
                else:
                    logger.info("  ✅ %s: valid", log_file.name)

                if warnings:
                    logger.info("  ⚠️  %s warnings:", log_file.name)
                    for w in warnings:
                        logger.info("     - %s", w)
        else:
            logger.info("  ⚠️  No iteration log files found")

    # 检查最终报告
    report_file = results_path / 'evaluation_report.md'
    if report_file.exists():
        logger.info("\n✅ evaluation_report.md: present")
    else:
        logger.info("\n⚠️  evaluation_report.md: missing")

    logger.info("\n%s", sep)
    if all_valid:
        logger.info("✅ All validations passed!")
    else:
        logger.info("❌ Some validations failed")
    logger.info("%s\n", sep)

    return all_valid


if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = 'wan2.2-a2-evaluation/results'

    valid = validate_results_directory(results_dir)
    sys.exit(0 if valid else 1)
