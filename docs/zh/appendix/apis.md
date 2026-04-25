# API 参考

本文档描述 `mindiesd` 包中除 `layers` 模块外的核心接口。所有接口均可通过 `from mindiesd import <接口名>` 直接导入使用。

## 缓存系列

缓存系列接口提供模型推理过程中的缓存管理能力，支持注意力缓存和 DiT 块缓存。

| 接口名 | 类型 | 功能描述 |
|--------|------|----------|
| `CacheConfig` | 类 | 缓存配置类，定义缓存的方法、块数、步数等参数 |
| `CacheAgent` | 类 | 缓存代理类，根据配置管理缓存的应用 |

### CacheConfig

缓存配置类，用于定义缓存的方法、块数、步数等参数。

```python
from mindiesd import CacheConfig
```

#### 类签名

```python
@dataclass
class CacheConfig:
    method: str
    blocks_count: int
    steps_count: int
    step_start: int = 0
    step_interval: int = 1
    step_end: int = 10000
    block_start: int = 0
    block_end: int = 10000
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `method` | `str` | 是 | - | 缓存方法，支持 "attention_cache" 和 "dit_block_cache" |
| `blocks_count` | `int` | 是 | - | 每步的块数 |
| `steps_count` | `int` | 是 | - | 总步数 |
| `step_start` | `int` | 否 | `0` | 开始步数 |
| `step_interval` | `int` | 否 | `1` | 步间隔 |
| `step_end` | `int` | 否 | `10000` | 结束步数 |
| `block_start` | `int` | 否 | `0` | 开始块数 |
| `block_end` | `int` | 否 | `10000` | 结束块数 |

#### 使用示例

```python
from mindiesd import CacheConfig

# 配置注意力缓存
cache_config = CacheConfig(
    method="attention_cache",
    blocks_count=4,
    steps_count=10,
    step_start=0,
    step_interval=2,
    step_end=8,
    block_start=0,
    block_end=4
)
```

---

### CacheAgent

缓存代理类，根据配置管理缓存的应用，决定何时应用缓存策略。

```python
from mindiesd import CacheAgent
```

#### 类签名

```python
class CacheAgent:
    def __init__(self, config: CacheConfig):
        pass
    
    def apply(self, function: callable, *args, **kwargs):
        pass
```

#### 构造参数

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `config` | `CacheConfig` | 是 | - | 缓存配置对象 |

#### apply 方法

```python
apply(function: callable, *args, **kwargs) -> Any
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `function` | `callable` | 是 | - | 要执行的函数 |
| `*args` | - | 否 | - | 函数位置参数 |
| `**kwargs` | - | 否 | - | 函数关键字参数 |

#### 返回值

函数执行结果，类型与原函数返回值一致。

#### 使用示例

```python
from mindiesd import CacheConfig, CacheAgent

# 创建缓存配置
cache_config = CacheConfig(
    method="attention_cache",
    blocks_count=4,
    steps_count=10,
    step_interval=2
)

# 创建缓存代理
cache_agent = CacheAgent(cache_config)

# 应用缓存
def inference_function(x):
    # 模型推理逻辑
    return x

result = cache_agent.apply(inference_function, input_tensor)
```

#### 工作原理

1. 检查配置是否满足缓存条件
2. 如果不满足条件，直接执行原函数
3. 如果满足条件，应用缓存策略执行函数
4. 内部维护计数器，控制缓存的应用时机

---

## 量化系列

量化系列接口提供模型量化能力，支持多种量化算法和时间步感知量化。

| 接口名 | 类型 | 功能描述 |
|--------|------|----------|
| `quantize` | 函数 | 模型量化函数，将浮点模型转换为量化模型 |
| `TimestepManager` | 类 | 时间步管理类，用于多模态量化过程中的时间步索引管理 |
| `TimestepPolicyConfig` | 类 | 时间步策略配置类，定义不同时间步的量化策略 |

### quantize

模型量化函数，将浮点模型转换为量化模型，支持多种量化算法。

```python
from mindiesd import quantize
```

#### 函数签名

```python
quantize(model, quant_des_path, **kwargs) -> nn.Module
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `nn.Module` | 是 | - | 需要量化的浮点模型 |
| `quant_des_path` | `str` | 是 | - | 量化权重描述文件的绝对路径 |
| `**kwargs.timestep_config` | `TimestepPolicyConfig` | 否 | `None` | 使用时间步量化时需要传入的配置 |
| `**kwargs.dtype` | `torch.dtype` | 否 | `torch.bfloat16` | 反量化的类型，支持 `torch.float16` 或 `torch.bfloat16` |
| `**kwargs.map` | `Dict[nn.Module, nn.Module]` | 否 | `None` | 自定义层映射规则 |

#### 返回值

`nn.Module`：量化后的模型。

#### 使用示例

```python
import torch
import torch.nn as nn
from mindiesd import quantize

# 加载浮点模型
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024)
)

# 量化模型
quantized_model = quantize(model, "path/to/quant_des.json")

# 使用量化模型进行推理
input_tensor = torch.randn(1, 1024, device="npu")
output = quantized_model(input_tensor)
```

#### 支持的量化算法

- W8A8 动态量化
- W8A8 MXFP8 量化
- W4A4 动态量化
- W4A4 MXFP4 双尺度量化
- W4A4 MXFP4 动态量化
- FP8 动态量化

---

### TimestepManager

时间步管理类，用于多模态量化过程中的时间步索引管理，使用上下文变量存储时间步信息。

```python
from mindiesd import TimestepManager
```

#### 类方法

##### set_timestep_idx

```python
@classmethod
def set_timestep_idx(cls, cur_timestep: int) -> None
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `cur_timestep` | `int` | 是 | - | 当前迭代时间步 |

##### get_timestep_idx

```python
@classmethod
def get_timestep_idx(cls) -> Optional[int]
```

#### 返回值

`Optional[int]`：当前时间步索引。

##### set_timestep_idx_max

```python
@classmethod
def set_timestep_idx_max(cls, t_idx: int) -> None
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `t_idx` | `int` | 是 | - | 最大时间步索引 |

##### get_timestep_idx_max

```python
@classmethod
def get_timestep_idx_max(cls) -> Optional[int]
```

#### 返回值

`Optional[int]`：最大时间步索引。

#### 使用示例

```python
from mindiesd import TimestepManager

# 设置最大时间步
TimestepManager.set_timestep_idx_max(100)

# 在每个时间步设置当前时间步
for t in range(100):
    TimestepManager.set_timestep_idx(t)
    # 执行量化推理
    output = model(input_tensor)
```

---

### TimestepPolicyConfig

时间步策略配置类，定义不同时间步的量化策略，支持静态和动态两种策略。

```python
from mindiesd import TimestepPolicyConfig
```

#### 类签名

```python
class TimestepPolicyConfig:
    def __init__(self):
        pass
    
    def register(self, step_range, strategy):
        pass
    
    def get_strategy(self, step):
        pass
```

#### 方法说明

##### register

```python
register(step_range, strategy) -> None
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `step_range` | `int`/`list`/`range` | 是 | - | 时间步范围 |
| `strategy` | `str` | 是 | - | 量化策略，支持 "static" 和 "dynamic" |

##### get_strategy

```python
get_strategy(step) -> str
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `step` | `int` | 是 | - | 指定的时间步 |

#### 返回值

`str`：对应时间步的量化策略。

#### 使用示例

```python
from mindiesd import TimestepPolicyConfig

# 创建时间步策略配置
config = TimestepPolicyConfig()

# 注册策略：前 50 步使用静态策略，后 50 步使用动态策略
config.register(range(50), "static")
config.register(range(50, 100), "dynamic")

# 获取指定时间步的策略
strategy = config.get_strategy(25)  # 返回 "static"
strategy = config.get_strategy(75)  # 返回 "dynamic"
```

#### 策略说明

- **static**：静态量化策略，使用预计算的量化参数
- **dynamic**：动态量化策略，根据当前输入动态计算量化参数
