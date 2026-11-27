# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
import pytest
import torch
from ops.device_support import init_test_device, require_support
from UT.ops_reference import test_eagle_hadamard_quant_small_shape as reference


@pytest.fixture(scope='module', name='device')
def npu_device():
    selected = init_test_device()
    require_support('eagle_hadamard_quant')
    from mindiesd.layers import register_ops  # noqa: F401

    return selected


def strided_copy(tensor):
    backing = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] * 2))
    result = backing[..., ::2].copy_(tensor)
    assert not result.is_contiguous()
    return result


def check_case(device, shape, layout, kind, fmt, strided=False):
    m, n, k = shape
    shared = layout.startswith('shared')
    nz = layout.endswith('nz')
    values, matrices = reference._fixture(m, n, k, kind, shared)
    expected = reference.quantize_reference(reference._independent_rotation(values, k, kind, shared), n, fmt)
    x = torch.tensor(values, dtype=torch.bfloat16).reshape(m, n).to(device)
    weights = reference.pack_nz(matrices, k) if nz else matrices
    weight = torch.tensor(weights, dtype=torch.bfloat16).reshape(reference._rotation_shape(n, k, layout)).to(device)
    if strided:
        x, weight = strided_copy(x), strided_copy(weight)
    y, scale = torch.ops.mindiesd.eagle_hadamard_quant(x, weight, k, fmt, nz)
    dtype = torch.uint8 if fmt == 'mxfp4' else torch.float8_e4m3fn
    for tensor, dims, target_dtype, raw in (
        (y, expected.y_shape, dtype, expected.y),
        (scale, expected.scale_shape, torch.float8_e8m0fnu, expected.scale),
    ):
        assert (tuple(tensor.shape), tensor.dtype, tensor.device) == (dims, target_dtype, device)
        assert tensor.view(torch.uint8).cpu().numpy().tobytes() == raw


@pytest.mark.parametrize('shape', reference.SMALL_SHAPES)
@pytest.mark.parametrize('layout', reference.LAYOUTS)
@pytest.mark.parametrize('kind', ('hadamard', 'permutation'))
@pytest.mark.parametrize('fmt', reference.FORMATS)
def test_small_shape(device, shape, layout, kind, fmt):
    check_case(device, shape, layout, kind, fmt)


@pytest.mark.parametrize('shape', [(3, 128, 32), (5, 256, 64)])
@pytest.mark.parametrize('layout', reference.LAYOUTS)
@pytest.mark.parametrize('fmt', reference.FORMATS)
def test_strided_and_grouped(device, shape, layout, fmt):
    check_case(device, shape, layout, 'permutation', fmt, strided=True)


class Eagle(torch.nn.Module):
    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt

    def forward(self, x, weight):
        return torch.ops.mindiesd.eagle_hadamard_quant(x, weight, 32, self.fmt, False)


@pytest.mark.parametrize('fmt', reference.FORMATS)
@pytest.mark.parametrize('capture', ['compile', 'export'])
def test_dynamic_graph(device, fmt, capture):
    model = Eagle(fmt)
    weight = torch.eye(32, dtype=torch.bfloat16).to(device)
    inputs = [torch.ones(rows, 96, dtype=torch.bfloat16, device=device) for rows in (3, 5)]
    graphs = []
    if capture == 'compile':

        def backend(graph, examples):
            graphs.append(graph)
            return graph.forward

        torch._dynamo.mark_dynamic(inputs[0], 0)
        run = torch.compile(model, fullgraph=True, backend=backend)
    else:
        rows = torch.export.Dim('rows', min=2, max=8)
        program = torch.export.export(model, (inputs[0], weight), dynamic_shapes=({0: rows}, None))
        graphs.append(program.graph_module)
        run = program.module()
    for x in inputs:
        y, scale = run(x, weight)
        assert y.shape == (len(x), 48 if fmt == 'mxfp4' else 96)
        assert y.dtype == (torch.uint8 if fmt == 'mxfp4' else torch.float8_e4m3fn)
        assert scale.shape == (len(x), 2, 2)
        assert scale.dtype == torch.float8_e8m0fnu
    op = torch.ops.mindiesd.eagle_hadamard_quant
    assert len(graphs) == 1
    (call,) = [node for node in graphs[0].graph.nodes if node.target in (op, op.default)]
    graph_y, graph_scale = call.meta['example_value' if capture == 'compile' else 'val']
    assert isinstance(graph_y.shape[0], torch.SymInt)
    assert graph_y.shape[0] == graph_scale.shape[0]
    assert graph_y.shape[1:] == y.shape[1:] and graph_y.dtype == y.dtype
    assert graph_scale.shape[1:] == scale.shape[1:] and graph_scale.dtype == scale.dtype
