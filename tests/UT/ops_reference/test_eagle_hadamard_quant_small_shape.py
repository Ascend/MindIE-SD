# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""CPU small-shape pytest for the Eagle Hadamard operator (no NPU calls).

Run from the repository root:
    python -m pytest tests/UT/ops_reference/test_eagle_hadamard_quant_small_shape.py -q

Metadata tests compile the production CheckParams header once with a C++17
compiler (CXX, c++, g++, or clang++); only these tests skip if none is available.
Numerical tests compare a CPU reference against independent butterflies/gathers
and fixed encoding anchors. Passing does not validate an NPU kernel.
Large-shape NPU validation retains its separate existing entry point.
"""

import ctypes
import math
import random
import struct
from dataclasses import dataclass
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@dataclass(frozen=True)
class MxOutput:
    y: bytes
    scale: bytes
    y_shape: tuple[int, int]
    scale_shape: tuple[int, int, int]


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bfloat16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def pack_nz(matrices, block_size):
    """Serialize [B,N1,K1,K0,N0], with column tiles before row tiles."""
    k = block_size
    return [
        [
            matrix[(row_tile + row) * k + col_tile + col]
            for col_tile in range(0, k, 16)
            for row_tile in range(0, k, 16)
            for row in range(16)
            for col in range(16)
        ]
        for matrix in matrices
    ]


def rotate_reference(x, matrices, block_size, is_sign_h_nz=False):
    """Multiply each feature block on the right, then round to BF16."""
    k = block_size
    tiles = k // 16
    output = []
    for row in x:
        values = []
        for block in range(len(row) // k):
            matrix = matrices[0 if len(matrices) == 1 else block]
            for col in range(k):
                total = 0.0
                for inner in range(k):
                    offset = inner * k + col
                    if is_sign_h_nz:
                        offset = ((col // 16) * tiles + inner // 16) * 256 + (inner % 16) * 16 + col % 16
                    total = float32(total + bfloat16(row[block * k + inner]) * bfloat16(matrix[offset]))
                values.append(bfloat16(total))
        output.append(values)
    return output


def encode_mx(value, dst_type):
    """Nearest finite value, ties to an even code; return raw sign/magnitude bits."""
    if dst_type == 'mxfp4':
        levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        sign_mask = 8
    elif dst_type == 'mxfp8':
        levels = tuple(math.ldexp((code & 7) + (8 if code >> 3 else 0), max(1, code >> 3) - 10) for code in range(127))
        sign_mask = 128
    else:
        raise ValueError('dst_type must be mxfp4 or mxfp8')
    if not math.isfinite(value):
        raise ValueError('this CPU reference only models finite values')
    code = min(range(len(levels)), key=lambda index: (abs(abs(value) - levels[index]), index & 1))
    return code | (sign_mask if math.copysign(1.0, value) < 0 else 0)


def quantize_reference(rotated, n, dst_type):
    """Encode finite BF16 rows; zero groups and unused scale slots are zero-filled.

    The fixtures stay within normal E8M0 exponents. Extreme exponents and
    nonfinite values require the independent NPU suite, not this CPU model.
    """
    if dst_type not in ('mxfp4', 'mxfp8'):
        raise ValueError('dst_type must be mxfp4 or mxfp8')
    if n <= 0 or n % 32:
        raise ValueError('n must be a positive multiple of 32')
    m = len(rotated)
    fp4 = dst_type == 'mxfp4'
    scale_stride = ((n + 63) // 64) * 2
    y = bytearray(m * n // (2 if fp4 else 1))
    scale = bytearray(m * scale_stride)
    for row_index, row in enumerate(rotated):
        if len(row) != n:
            raise ValueError('each row must contain n elements')
        for start in range(0, n, 32):
            group = row[start : start + 32]
            if not all(math.isfinite(value) for value in group):
                raise ValueError('this CPU reference only models finite values')
            amax = max(abs(value) for value in group)
            if amax == 0:
                continue
            exponent = math.frexp(amax)[1] - 1 - (2 if fp4 else 8)
            if not -127 <= exponent <= 127:
                raise ValueError('scale exponent outside this CPU reference range')
            scale[row_index * scale_stride + start // 32] = exponent + 127
            for offset, value in enumerate(group):
                code = encode_mx(math.ldexp(value, -exponent), dst_type)
                index = row_index * n + start + offset
                if fp4:
                    y[index // 2] |= code << (4 * (index % 2))
                else:
                    y[index] = code
    return MxOutput(bytes(y), bytes(scale), (m, n // 2 if fp4 else n), (m, scale_stride // 2, 2))


SMALL_SHAPES = [(0, 32, 32), (1, 32, 32), (3, 96, 32), (5, 128, 64), (3, 256, 128), (1, 256, 256)]
LAYOUTS = ('nd', 'nz', 'shared_nd', 'shared_nz')
FORMATS = ('mxfp4', 'mxfp8')


@pytest.fixture(scope='module', name='check_params')
def check_params_fixture(tmp_path_factory):
    """Call the actual production validator through a minimal CPU-only C ABI."""
    compiler = os.environ.get('CXX')
    command = shlex.split(compiler, posix=sys.platform != 'win32') if compiler else []
    if command and sys.platform == 'win32':
        command[0] = command[0].strip('"')
    if not command:
        executable = next((path for name in ('c++', 'g++', 'clang++') if (path := shutil.which(name))), None)
        if executable is None:
            pytest.skip('production CheckParams tests require a C++17 compiler; set CXX')
        command = [executable]
    build = tmp_path_factory.mktemp('eagle_hadamard_metadata')
    source = build / 'metadata.cpp'
    source.write_text(
        """#include "op_host/eagle_hadamard_quant_params.h"
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
extern "C" EXPORT const char* eagle_check(
    int64_t m, int64_t n, const int64_t* shape, int64_t rank, int64_t k,
    const char* format, int nz, int64_t* output) {
    const auto p = EagleHadamardQuant::CheckParams(m, n, shape, rank, k, format, nz != 0);
    if (p.error) return p.error;
    output[0] = p.k;
    output[1] = p.blocks;
    output[2] = p.outputColumns;
    output[3] = p.scalePairs;
    output[4] = p.format;
    return nullptr;
}
""",
        encoding='utf-8',
    )
    operator = Path(__file__).resolve().parents[3] / 'csrc' / 'ops' / 'eagle_hadamard_quant'
    library_path = build / ('metadata.dll' if sys.platform == 'win32' else 'metadata.so')
    result = subprocess.run(
        [*command, '-std=c++17', '-shared', '-fPIC', '-I', str(operator), str(source), '-o', str(library_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f'C++ metadata bridge failed:\n{result.stdout}\n{result.stderr}'
    library = ctypes.CDLL(str(library_path))
    function = library.eagle_check
    pointer = ctypes.POINTER(ctypes.c_int64)
    function.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        pointer,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int,
        pointer,
    ]
    function.restype = ctypes.c_char_p

    def check(m, n, shape, k=32, fmt='mxfp4', nz=False):
        dimensions = (ctypes.c_int64 * len(shape))(*shape)
        output = (ctypes.c_int64 * 5)()
        error = function(m, n, dimensions, len(shape), k, fmt.encode('ascii'), nz, output)
        if error:
            raise ValueError(error.decode('utf-8'))
        return tuple(output)

    return check


def _rotation_shape(n, k, layout):
    count = 1 if layout.startswith('shared') else n // k
    if layout.endswith('nz'):
        return (count, k // 16, k // 16, 16, 16)
    return (k, k) if layout.startswith('shared') else (count, k, k)


@pytest.mark.parametrize('m,n,k', SMALL_SHAPES)
@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('fmt', FORMATS)
def test_production_metadata_small_shapes(check_params, m, n, k, layout, fmt):
    shape = _rotation_shape(n, k, layout)
    actual = check_params(m, n, shape, k, fmt, layout.endswith('nz'))
    count = 1 if layout.startswith('shared') else n // k
    assert actual == (k, count, n // 2 if fmt == 'mxfp4' else n, (n + 63) // 64, int(fmt == 'mxfp8'))


@pytest.mark.parametrize('m', (63, 64, 65, 127, 128, 129))
def test_production_metadata_row_boundaries(check_params, m):
    assert check_params(m, 96, (3, 32, 32)) == (32, 3, 48, 2, 0)


@pytest.mark.parametrize(
    'm,n,shape,k,fmt,nz,error',
    [
        (1, 32, (32, 32), 16, 'mxfp4', False, 'block_size'),
        (1, 32, (32, 32), 512, 'mxfp4', False, 'block_size'),
        (-1, 32, (32, 32), 32, 'mxfp4', False, 'uint32'),
        (2**32, 32, (32, 32), 32, 'mxfp4', False, 'uint32'),
        (1, 0, (32, 32), 32, 'mxfp4', False, 'positive multiple'),
        (1, 48, (32, 32), 32, 'mxfp4', False, 'positive multiple'),
        (1, 2**32, (32, 32), 32, 'mxfp4', False, 'uint32'),
        (1, 32, (32, 32), 32, 'int4', False, 'dst_type'),
        (1, 32, (32, 32), 32, 'MXFP4', False, 'dst_type'),
        (1, 32, (32, 32), 32, 'fp4', False, 'dst_type'),
        (1, 32, (32, 32), 32, 'mxfp4', True, '5D NZ'),
        (1, 32, (1, 2, 2, 16, 16), 32, 'mxfp4', False, 'ND sign_h'),
        (1, 32, (1, 1, 2, 16, 16), 32, 'mxfp4', True, 'NZ sign_h must have shape'),
        (1, 32, (1, 2, 2, 8, 16), 32, 'mxfp4', True, 'NZ sign_h must have shape'),
        (1, 32, (1, 2, 2, 16, 8), 32, 'mxfp4', True, 'NZ sign_h must have shape'),
        (1, 32, (64, 32), 32, 'mxfp4', False, 'matrix dimensions'),
        (1, 96, (1, 32, 32), 32, 'mxfp4', False, 'one shared rotation'),
        (1, 96, (2, 32, 32), 32, 'mxfp4', False, 'hd/block_size'),
        (1, 96, (2, 2, 2, 16, 16), 32, 'mxfp4', True, 'hd/block_size'),
        (1, 96, (0, 32, 32), 32, 'mxfp4', False, 'hd/block_size'),
    ],
)
def test_production_metadata_rejects_invalid_inputs(check_params, m, n, shape, k, fmt, nz, error):
    with pytest.raises(ValueError, match=error):
        check_params(m, n, shape, k, fmt, nz)


def _sign(col, block):
    return -1 if (col + block) % 3 == 0 else 1


def _fixture(m, n, k, kind, shared):
    x = [
        [0.0 if m > 1 and row == m - 1 else ((col * 7 + row * 3) % 19 - 9) / 8 for col in range(n)] for row in range(m)
    ]
    matrices = []
    for block in range(1 if shared else n // k):
        matrix = [0.0] * (k * k)
        for col in range(k):
            if kind == 'permutation':
                matrix[((5 * col + 3 + block) % k) * k + col] = _sign(col, block)
            else:
                for row in range(k):
                    parity = bin(row & col).count('1') % 2
                    matrix[row * k + col] = (-1 if parity else 1) * _sign(col, block) / 8
        matrices.append(matrix)
    return x, matrices


def _hadamard_butterflies(work):
    stride = 1
    while stride < len(work):
        for start in range(0, len(work), 2 * stride):
            for offset in range(stride):
                left, right = work[start + offset], work[start + offset + stride]
                work[start + offset] = left + right
                work[start + offset + stride] = left - right
        stride *= 2


def _independent_rotation(x, k, kind, shared):
    """No matrix multiply or NZ addressing: FWHT butterflies / signed gather."""
    output = []
    for row in x:
        result = []
        for block in range(len(row) // k):
            matrix_index = 0 if shared else block
            work = row[block * k : (block + 1) * k]
            if kind == 'hadamard':
                _hadamard_butterflies(work)
            for col in range(k):
                value = work[(5 * col + 3 + matrix_index) % k] if kind == 'permutation' else work[col] / 8
                rounded = bfloat16(value * _sign(col, matrix_index))
                result.append(0.0 if rounded == 0 else rounded)
        output.append(result)
    return output


@pytest.mark.parametrize('m,n,k', SMALL_SHAPES)
@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('kind', ('hadamard', 'permutation'))
@pytest.mark.parametrize('fmt', FORMATS)
def test_small_shape_reference_against_independent_oracle(m, n, k, layout, kind, fmt):
    shared = layout.startswith('shared')
    nz = layout.endswith('nz')
    x, matrices = _fixture(m, n, k, kind, shared)
    if nz:
        matrices = pack_nz(matrices, k)
    expected = _independent_rotation(x, k, kind, shared)
    encoded = quantize_reference(expected, n, fmt)
    for _ in range(2):
        rotated = rotate_reference(x, matrices, k, nz)
        assert rotated == expected
        actual = quantize_reference(rotated, n, fmt)
        assert actual == encoded
        assert actual.y_shape == (m, n // 2 if fmt == 'mxfp4' else n)
        assert actual.scale_shape == (m, (n + 63) // 64, 2)
        assert len(actual.y) == m * n // (2 if fmt == 'mxfp4' else 1)
        assert len(actual.scale) == m * ((n + 63) // 64) * 2


@pytest.mark.parametrize('k', (32, 64, 128, 256))
def test_asymmetric_fixture_detects_wrong_layout_and_transpose(k):
    x, matrices = _fixture(1, 2 * k, k, 'permutation', False)
    expected = _independent_rotation(x, k, 'permutation', False)
    # Deliberately serialize row tiles first; the NZ reader must expose the error.
    wrong_nz = [
        [
            matrix[(rt + r) * k + ct + c]
            for rt in range(0, k, 16)
            for ct in range(0, k, 16)
            for r in range(16)
            for c in range(16)
        ]
        for matrix in matrices
    ]
    transposed = [[matrix[col * k + row] for row in range(k) for col in range(k)] for matrix in matrices]
    assert rotate_reference(x, wrong_nz, k, True) != expected
    assert rotate_reference(x, transposed, k) != expected
    assert rotate_reference(x, matrices[:1], k) != expected


@pytest.mark.parametrize(
    'value,expected', [(1.00390625, 1.0), (1.01171875, 1.015625), (-1.00390625, -1.0), (-1.01171875, -1.015625)]
)
def test_bfloat16_round_to_nearest_even(value, expected):
    assert bfloat16(value) == expected


@pytest.mark.parametrize('code,value', enumerate((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)))
def test_fp4_all_codes_and_signs(code, value):
    assert encode_mx(value, 'mxfp4') == code
    assert encode_mx(-value, 'mxfp4') == code | 8


@pytest.mark.parametrize(
    'fmt,value,code',
    [
        ('mxfp4', 0.25, 0),
        ('mxfp4', 0.75, 2),
        ('mxfp4', 1.25, 2),
        ('mxfp4', 5.0, 6),
        ('mxfp4', 10.0, 7),
        ('mxfp8', 1.0, 0x38),
        ('mxfp8', -1.0, 0xB8),
        ('mxfp8', 1.0625, 0x38),
        ('mxfp8', 1.1875, 0x3A),
        ('mxfp8', 2**-9, 1),
        ('mxfp8', 448.0, 0x7E),
        ('mxfp8', 512.0, 0x7E),
    ],
)
def test_encoding_known_values_ties_and_saturation(fmt, value, code):
    assert encode_mx(value, fmt) == code


@pytest.mark.parametrize(
    'fmt,scale,y0,y64', [('mxfp4', [125, 0, 126, 0], 0xC6, 6), ('mxfp8', [119, 0, 120, 0], 0x78, 0x78)]
)
def test_fixed_bytes_scale_groups_zero_group_and_padding(fmt, scale, y0, y64):
    row = [0.0] * 96
    row[0], row[1], row[64] = 1.0, -0.5, 2.0
    output = quantize_reference([row], 96, fmt)
    expected_y = bytearray(48 if fmt == 'mxfp4' else 96)
    expected_y[0] = y0
    expected_y[32 if fmt == 'mxfp4' else 64] = y64
    if fmt == 'mxfp8':
        expected_y[1] = 0xF0
    assert output.y == bytes(expected_y)
    assert output.scale == bytes(scale)
    assert output.scale_shape == (1, 2, 2)


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('m,n', [(0, 32), (3, 32), (3, 96)])
def test_empty_rows_and_zero_groups(fmt, m, n):
    output = quantize_reference([[0.0] * n for _ in range(m)], n, fmt)
    assert output.y == bytes(m * n // (2 if fmt == 'mxfp4' else 1))
    assert output.scale == bytes(m * ((n + 63) // 64) * 2)


@pytest.mark.parametrize('k', (32, 64))
@pytest.mark.parametrize('special', (math.nan, math.inf, -math.inf))
def test_nonfinite_stays_in_its_rotation_block(k, special):
    x, matrices = _fixture(1, 3 * k, k, 'hadamard', False)
    expected = rotate_reference(x, matrices, k)[0]
    x[0][k] = special
    actual = rotate_reference(x, matrices, k)[0]
    assert actual[:k] == expected[:k]
    assert actual[2 * k :] == expected[2 * k :]
    assert all(not math.isfinite(value) for value in actual[k : 2 * k])


@pytest.mark.parametrize('distribution', ('uniform', 'normal'))
def test_seeded_shared_and_per_block_rotation(distribution):
    k = 32
    rng = random.Random(42)
    mu, sigma = rng.uniform(-5, 5), rng.uniform(0.1, 2)
    x = [[bfloat16(rng.uniform(-5, 5) if distribution == 'uniform' else rng.gauss(mu, sigma)) for _ in range(96)]]
    _, shared = _fixture(1, 96, k, 'permutation', True)
    expected = _independent_rotation(x, k, 'permutation', True)
    assert rotate_reference(x, shared, k) == expected
    assert rotate_reference(x, pack_nz(shared * 3, k), k, True) == expected


@pytest.mark.parametrize('k', (32, 64))
def test_nonfinite_weight_leaves_other_blocks_intact(k):
    x, matrices = _fixture(1, 3 * k, k, 'hadamard', False)
    expected = rotate_reference(x, matrices, k)[0]
    matrices[1][k] = math.inf
    matrices[1][k + 1] = math.nan
    for weights, nz in ((matrices, False), (pack_nz(matrices, k), True)):
        actual = rotate_reference(x, weights, k, nz)[0]
        assert actual[:k] == expected[:k] and actual[2 * k :] == expected[2 * k :]
        assert not math.isfinite(actual[k]) and math.isnan(actual[k + 1])


def test_e4m3_all_finite_codes_and_rounding_midpoints():
    # Independent E4M3 decoding with an implicit leading one for normals.
    levels = [mant / 512 for mant in range(8)]
    levels += [(1 + mant / 8) * 2 ** (exp - 7) for exp in range(1, 16) for mant in range(8) if (exp, mant) != (15, 7)]
    for sign, mask in ((1, 0), (-1, 128)):
        for code, value in enumerate(levels):
            assert encode_mx(sign * value, 'mxfp8') == mask | code
        for code, (lo, hi) in enumerate(zip(levels, levels[1:])):
            mid = (lo + hi) / 2
            assert encode_mx(sign * mid, 'mxfp8') == mask | (code + code % 2)
            assert encode_mx(sign * math.nextafter(mid, lo), 'mxfp8') == mask | code
            assert encode_mx(sign * math.nextafter(mid, hi), 'mxfp8') == mask | (code + 1)


@pytest.mark.parametrize('fmt,emax', [('mxfp4', 2), ('mxfp8', 8)])
def test_e8m0_independent_group_exponents(fmt, emax):
    # Last BF16 below a power of two, exact transition, first BF16 above, then an all-zero group.
    for exponent in (-18, -1, 0, 9):
        power = 2.0**exponent
        peaks = [power * (255 / 256), power, power * (129 / 128), 0.0]
        row = [sample for peak in peaks for sample in [peak, -peak] * 16]
        result = quantize_reference([row], 128, fmt)
        code = exponent + 127 - emax
        assert result.scale == bytes([code - 1, code, code, 0])
