# MindIE SD

MindIE SD is an Ascend-focused inference acceleration toolkit for Stable Diffusion and related multimodal generation workloads.

## Highlights

- Ascend-friendly custom operators and fused kernels
- Quantization, sparse quantization, cache, and parallel execution features
- `torch.compile` integration for graph-level acceleration

## Quick start

Complete the environment preparation and MindIE SD installation first, then install model-specific dependencies and run an example:

```bash
git clone https://modelers.cn/MindIE/Wan2.1.git
cd Wan2.1
pip install -r requirements.txt
```

## Documentation

- English overview: [docs/en/README.md](docs/en/README.md)
- English quick start: [docs/en/quick_start.md](docs/en/quick_start.md)
- English installation guide: [docs/en/installing_guide.md](docs/en/installing_guide.md)
- English user guide: [docs/en/menu_user_manual.md](docs/en/menu_user_manual.md)
- English architecture: [docs/en/architecture.md](docs/en/architecture.md)
- English environment variables: [docs/en/environment_variable_configuration.md](docs/en/environment_variable_configuration.md)
- English support matrix: [docs/en/features/supported_matrix.md](docs/en/features/supported_matrix.md)
- English cache features: [docs/en/features/cache.md](docs/en/features/cache.md)
- English parallelism features: [docs/en/features/parallelism.md](docs/en/features/parallelism.md)
- English sparse quantization: [docs/en/features/sparse_quantization.md](docs/en/features/sparse_quantization.md)
- Chinese overview: [docs/zh/index.md](docs/zh/index.md)
- Chinese quick start: [docs/zh/quick_start.md](docs/zh/quick_start.md)
- Chinese installation guide: [docs/zh/installing_guide.md](docs/zh/installing_guide.md)
- Chinese architecture: [docs/zh/architecture.md](docs/zh/architecture.md)

## Community and governance

- Contributing: [contributing.md](contributing.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Governance: [docs/en/community/governance.md](docs/en/community/governance.md)
