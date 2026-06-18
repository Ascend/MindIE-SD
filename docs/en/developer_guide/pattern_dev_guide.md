# Pattern Development Guide

This document describes the complete development workflow for Patterns (fused operator patterns) in the MindIE SD compilation module.

## Development Workflow

```text
Model Analysis -> Pattern Creation -> Three-Step Registration -> Unit Verification -> Integration Verification
```

## Stage 1: Model Analysis

Locate the actual implementation code of target operators in the model source code (e.g., RMSNorm, RoPE, AdaLayerNorm) and extract complete code snippets as the basis for Pattern and testing.

Also determine parameter origin:

| Parameter Origin | Pattern Path | Applicable Scenarios |
| ---------- | ------------- | ---------- |
| functional API (e.g., `F.rms_norm`) | `register_replacement` | Parameters passed as function arguments |
| nn.Module (e.g., `self.weight`) | Custom Graph Pass | Parameters from module member variables |

## Stage 2: Create Pattern

First check whether existing patterns already cover the target operator. If not, create a new one following the non-invasive principle (always create new files; do not modify existing pattern files).

Choose the path based on parameter origin from Stage 1:

- **register_replacement path**: Create a `PatternBase` subclass (factory + closure), register in `pattern_registry`
- **Custom Graph Pass path**: Implement a custom FX graph traversal pass

## Stage 3: Three-Step Registration

Involves modifications to 3 files, all as code additions (do not modify existing code):

1. `patterns/__init__.py` --- add to `__all__` + `from .xxx_pattern import XxxPatternGroup`
2. `passes/__init__.py` --- add entry in the `pattern_registry` dictionary
3. `compiliation_config.py` --- add `enable_xxx: bool = True` in the `FusionPatterns` dataclass

Naming convention: config keys use the `enable_<model>_<op>` format.

## Stage 4: Unit Verification

Verification criterion: `cosine_similarity(compiled, original) > 2^-7`.

> Note: Passing unit tests does not guarantee the Pattern hits the full model. The test model and pattern share the same code and will necessarily match. Full-model matching requires final confirmation via Stage 5 integration verification.

## Stage 5: Integration Verification

Three layers of verification in order; no skipping allowed:

| Layer | Method | Confirmation |
| ------ | ------ | --------- |
| 1. Pattern Hit | graph dump + logs | `PatternMatchPass replace N` count increases |
| 2. Fusion Kernel | profiling -> `kernel_details.csv` | Fused kernels appear + original kernels disappear |
| 3. Full Model Regression | dummy run + compile | Inference completes normally without crashes |

Kernel name mapping:

| Operator | Kernel Name |
| ------ | ------------ |
| RMSNorm | `rms_norm` / `RmsNorm` |
| RoPE | `npu_rotary_mul` / `RotaryMul` |
| AdaLN | `adaln` / `adln` |
| GELU | `FastGelu` |
