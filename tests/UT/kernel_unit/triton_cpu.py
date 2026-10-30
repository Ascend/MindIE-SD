"""Small, deliberately strict CPU facade for numerical Triton body tests.

This executes the original functions, not a replacement implementation. Device
rounding, concurrency and synchronization require the separate NPU tests.
"""

import inspect
import itertools
from contextlib import nullcontext
from types import SimpleNamespace

import torch


class Pointer:
    """Element-addressed storage pointer, including a tensor's storage offset."""

    def __init__(self, tensor, offset=None):
        self.tensor = tensor
        self.offset = tensor.storage_offset() if offset is None else offset
        self.type = SimpleNamespace(element_ty=tensor.dtype)
        self.dtype = self.type

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    __radd__ = __add__

    def storage(self):
        size = self.tensor.untyped_storage().nbytes() // self.tensor.element_size()
        return self.tensor.as_strided((size,), (1,), storage_offset=0)


class BlockPointer:
    def __init__(self, base, shape, strides, offsets, block_shape, order):
        self.base = base
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.offsets = tuple(offsets)
        self.block_shape = tuple(block_shape)
        self.order = order
        self.dtype = base.dtype

    def addresses(self):
        address = self.base.offset
        valid = torch.ones(self.block_shape, dtype=torch.bool)
        for axis, (offset, length, stride) in enumerate(zip(self.offsets, self.shape, self.strides)):
            dims = [1] * len(self.shape)
            dims[axis] = self.block_shape[axis]
            index = (torch.arange(self.block_shape[axis]) + offset).reshape(dims)
            address = address + index * stride
            valid &= (index >= 0) & (index < length)
        return Pointer(self.base.tensor, address), valid


class Language:
    constexpr = object
    float32 = torch.float32
    float16 = torch.float16
    bfloat16 = torch.bfloat16
    int64 = torch.int64
    int32 = torch.int32
    int8 = torch.int8
    int1 = torch.bool
    PropagateNan = SimpleNamespace(ALL=True)
    math = SimpleNamespace(exp=torch.exp, exp2=torch.exp2, log=torch.log, log2=torch.log2)

    def __init__(self):
        self.pid = (0,)
        self.grid = (1,)

    def program_id(self, axis=0):
        return torch.tensor(self.pid[axis], dtype=torch.int32)

    def num_programs(self, axis=0):
        return self.grid[axis]

    @staticmethod
    def arange(start, end):
        return torch.arange(start, end)

    @staticmethod
    def range(*args, **kwargs):
        return (torch.tensor(i, dtype=torch.int32) for i in range(*(int(a) for a in args)))

    static_range = range

    @staticmethod
    def load(pointer, mask=None, other=0, boundary_check=(), **kwargs):
        if isinstance(pointer, BlockPointer):
            pointer, valid = pointer.addresses()
            mask = valid if boundary_check else mask
        index = torch.as_tensor(pointer.offset, dtype=torch.long)
        mask = torch.ones_like(index, dtype=torch.bool) if mask is None else torch.broadcast_to(mask, index.shape)
        storage = pointer.storage()
        if torch.any(mask & ((index < 0) | (index >= storage.numel()))):
            raise IndexError("unmasked CPU kernel load outside storage")
        result = torch.full(index.shape, other, dtype=storage.dtype)
        result[mask] = storage[index[mask]]
        return result

    @staticmethod
    def store(pointer, value, mask=None, boundary_check=()):
        if isinstance(pointer, BlockPointer):
            pointer, valid = pointer.addresses()
            mask = valid if boundary_check else mask
        index, value = torch.broadcast_tensors(torch.as_tensor(pointer.offset, dtype=torch.long), value)
        mask = torch.ones_like(index, dtype=torch.bool) if mask is None else torch.broadcast_to(mask, index.shape)
        storage = pointer.storage()
        if torch.any(mask & ((index < 0) | (index >= storage.numel()))):
            raise IndexError("unmasked CPU kernel store outside storage")
        storage[index[mask]] = value[mask].to(storage.dtype)

    @staticmethod
    def full(shape, value, dtype):
        return torch.full(tuple(shape), value, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(tuple(shape), dtype=dtype)

    @staticmethod
    def sum(value, axis=0):
        return value.sum(dim=axis)

    @staticmethod
    def max(value, axis=0, **kwargs):
        return value.max(dim=axis).values

    @staticmethod
    def maximum(left, right, **kwargs):
        return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def dot(left, right, **kwargs):
        return left.float() @ right.float()

    @staticmethod
    def trans(value):
        return value.T

    @staticmethod
    def make_block_ptr(**kwargs):
        return BlockPointer(**kwargs)

    @staticmethod
    def advance(pointer, offsets):
        return BlockPointer(
            pointer.base,
            pointer.shape,
            pointer.strides,
            tuple(a + b for a, b in zip(pointer.offsets, offsets)),
            pointer.block_shape,
            pointer.order,
        )

    sigmoid = staticmethod(torch.sigmoid)
    rsqrt = staticmethod(torch.rsqrt)
    sqrt = staticmethod(torch.sqrt)
    exp = staticmethod(torch.exp)
    exp2 = staticmethod(torch.exp2)
    where = staticmethod(torch.where)


class Buffers:
    @staticmethod
    def alloc(dtype, shape, address_space=None, **kwargs):
        return torch.full(tuple(shape), torch.nan if dtype.is_floating_point else -77, dtype=dtype)

    @staticmethod
    def to_tensor(buffer, target_shape=None):
        if target_shape is not None and buffer.ndim == 4 and len(target_shape) == 2:
            return buffer.permute(1, 2, 0, 3).reshape(target_shape)
        return buffer if target_shape is None else buffer.reshape(target_shape)

    @staticmethod
    def to_buffer(value=None, address_space=None, bind_buffer=None, tensor=None):
        value = tensor if value is None else value
        if bind_buffer is not None:
            bind_buffer.copy_(value)
            return bind_buffer
        return value

    @staticmethod
    def subview(value, offsets, sizes, strides):
        return value[tuple(slice(int(o), int(o + n * s), int(s)) for o, n, s in zip(offsets, sizes, strides))]


class Ascend:
    """CPU data movement; synchronization calls are recorded boundary events."""

    ascend_address_space = SimpleNamespace(UB="UB", L1="L1", L0C="L0C")
    SYNC_IN_VF = SimpleNamespace(VST_VLD="VST_VLD")
    PIPE = SimpleNamespace(**{key: key for key in ("PIPE_V", "PIPE_FIX", "PIPE_MTE1", "PIPE_MTE3")})
    FixpipeDMAMode = SimpleNamespace(NZ2ND="NZ2ND")
    FixpipeDualDstMode = SimpleNamespace(ROW_SPLIT="ROW_SPLIT")

    def __init__(self):
        self.events = []
        self.vector_id = 0

    def sub_vec_id(self):
        return torch.tensor(self.vector_id, dtype=torch.int32)

    @staticmethod
    def scope(**kwargs):
        return nullcontext()

    @staticmethod
    def parallel(*args):
        return Language.range(*args)

    @staticmethod
    def extract_slice(value, offsets, sizes, strides):
        return Buffers.subview(value, offsets, sizes, strides).clone()

    @staticmethod
    def insert_slice(value, source, offsets, sizes, strides):
        Buffers.subview(value, offsets, sizes, strides).copy_(source)
        return value

    @staticmethod
    def copy(source, destination):
        destination.copy_(source)

    def debug_barrier(self, *args):
        self.events.append(("barrier", args))

    def sync_block_set(self, *args):
        self.events.append(("set", args))

    def sync_block_wait(self, *args):
        self.events.append(("wait", args))

    def multibuffer(self, *args):
        self.events.append(("multibuffer", args[1:]))

    @staticmethod
    def copy_from_ub_to_l1(source, destination):
        destination.copy_(source)

    def fixpipe(self, source, destination, *args):
        # A single vector role sees its contiguous row half.
        start = self.vector_id * destination.shape[0]
        destination.copy_(source[start : start + destination.shape[0]])


def original(kernel):
    return getattr(kernel, "fn", kernel)


class Launch:
    """Explicit CPU launch; unpatched production launches remain forbidden."""

    def __init__(self, kernel, language):
        self.fn = original(kernel)
        self.language = language

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            params = inspect.signature(self.fn).parameters
            kwargs = {key: value for key, value in kwargs.items() if key in params}
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            for key, value in bound.arguments.items():
                value = getattr(value, "cpu_tensor", value)
                if isinstance(value, torch.Tensor):
                    bound.arguments[key] = Pointer(value)
            actual_grid = grid(kwargs) if callable(grid) else grid
            self.language.grid = tuple(int(x) for x in actual_grid)
            for pid in itertools.product(*(range(x) for x in self.language.grid)):
                self.language.pid = pid
                with torch.no_grad():
                    self.fn(*bound.args, **bound.kwargs)

        return run


def install(monkeypatch, module):
    """Patch only this test's module globals, preserving original code objects."""
    language = Language()
    monkeypatch.setattr(module, "tl", language, raising=False)
    monkeypatch.setattr(module, "range", language.range, raising=False)
    monkeypatch.setattr(module, "bl", Buffers(), raising=False)
    monkeypatch.setattr(module, "al", Ascend(), raising=False)
    for name, value in list(vars(module).items()):
        if hasattr(value, "fn") and inspect.isfunction(value.fn):
            monkeypatch.setattr(module, name, Launch(value, language))
    return language
