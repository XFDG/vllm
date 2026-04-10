# vLLM Paged Attention 源码梳理

## 先说结论

`PagedAttention` 在 vLLM 里不是“单个 kernel 的名字”，而是一整套机制：

1. 把 KV cache 按固定 `block_size` 切成很多物理页/块。
2. 每个请求维护一张 `block_table`，把“逻辑块号”映射到“物理块号”。
3. 写 KV cache 时，根据 `slot_mapping` 把新 token 写到正确物理槽位。
4. decode attention 时，kernel 通过 `block_table` 去不连续的物理页中取 K/V，再完成注意力计算。

如果只看“老的自定义 CUDA paged attention kernel”，核心实现确实在：

- `csrc/attention/paged_attention_v1.cu`
- `csrc/attention/paged_attention_v2.cu`
- `csrc/attention/attention_kernels.cuh`

但如果看“今天 vLLM 主线 CUDA 路径”，主力 backend 更多是 FlashInfer / Triton。它们仍然建立在同一套 paged KV cache 和 block table 思想之上，只是 kernel 换了。

另外，仓库里的 `docs/design/paged_attention.md` 开头已经明确写了：这是历史文档，不完全对应今天的代码。但它描述的核心思想和 `attention_kernels.cuh` 的主流程仍然是对得上的。

---

## 1. 最关键的文件

建议按下面顺序读：

- `vllm/v1/core/kv_cache_manager.py`
  - 块从哪里分配出来，最终如何形成 `block_ids`。
- `vllm/v1/core/block_pool.py`
  - 物理 block/page 池，负责 free/allocate/cache。
- `vllm/v1/worker/gpu/block_table.py`
  - 维护 GPU 侧 `block_tables`，并计算 `slot_mappings`。
- `vllm/v1/worker/gpu/model_runner.py`
  - 把 scheduler 给出的 `block_ids` 接进来，真正准备 attention 元数据。
- `vllm/v1/worker/gpu/attn_utils.py`
  - 把 group 级别的 `block_table` / `slot_mapping` 映射到具体 layer。
- `vllm/v1/attention/ops/paged_attn.py`
  - 定义 paged KV cache 的 key/value 视图形状，写 cache 的 Python 入口。
- `csrc/cache_kernels.cu`
  - `reshape_and_cache_kernel`，把 token 的 K/V 写进 paged cache。
- `csrc/attention/attention_kernels.cuh`
  - 真正的 paged attention 核心逻辑。
- `csrc/attention/paged_attention_v1.cu`
  - v1 launcher，单阶段 kernel。
- `csrc/attention/paged_attention_v2.cu`
  - v2 launcher，分区 + reduce 的双阶段 kernel。
- `vllm/v1/attention/backends/flashinfer.py`
  - 当前主线 CUDA backend 之一，如何把 `block_table` 转成 FlashInfer 需要的 paged metadata。
- `vllm/v1/attention/backends/triton_attn.py`
  - 当前主线 CUDA backend 之一，如何直接把 `block_table` 和 `slot_mapping` 交给 Triton attention。

---

## 2. 先抓住三个核心数据结构

### 2.1 `block_table`

`block_table` 的本质是：

- 行维度：请求/序列
- 列维度：该请求的第几个逻辑 block
- 值：对应的物理 block id

也就是：

```text
logical block index  ->  physical block number
```

GPU 侧实现看这里：

- `vllm/v1/worker/gpu/block_table.py:13`
- `vllm/v1/worker/gpu/block_table.py:37`
- `vllm/v1/worker/gpu/block_table.py:87`

其中 `append_block_ids()` 会把某个请求新分到的 `block_ids` 追加到表里。

### 2.2 `slot_mapping`

`slot_mapping` 是更细粒度的“token 级别物理位置映射”。它不是块号，而是最终写 cache 时的“槽位号”。

核心公式在：

- `vllm/v1/worker/gpu/block_table.py:256`
- `vllm/v1/worker/gpu/block_table.py:264`

普通情况下：

```text
slot_id = physical_block_number * block_size + block_offset
```

其中：

- `block_indices = position // block_size`
- `block_offsets = position % block_size`
- `physical_block_number = block_table[req_idx, block_indices]`

### 2.3 paged KV cache 布局

在 `vllm/v1/attention/ops/paged_attn.py:15` 里，`split_kv_cache()` 明确了旧 custom kernel 视角下的 cache 形状：

- key cache:
  - `[num_blocks, num_kv_heads, head_size / x, block_size, x]`
- value cache:
  - `[num_blocks, num_kv_heads, head_size, block_size]`

对应代码：

- `vllm/v1/attention/ops/paged_attn.py:17`
- `vllm/v1/attention/ops/paged_attn.py:25`
- `vllm/v1/attention/ops/paged_attn.py:27`

这里 key/value 形状不一样是故意的：

- K 的布局更偏向 QK dot 时的向量化读取和 memory coalescing。
- V 的布局更偏向 softmax 权重乘 V 时的访问模式。

---

## 3. 从调度到 attention kernel 的完整链路

### 3.1 block 先由 KV cache manager 分配

分配入口：

- `vllm/v1/core/kv_cache_manager.py:257`

这里会从 block pool 拿到新的 `KVCacheBlocks`。物理 block 的底层池子在：

- `vllm/v1/core/block_pool.py:129`

然后 `KVCacheBlocks.get_block_ids()` 把内部对象转成纯整数 block id：

- `vllm/v1/core/kv_cache_manager.py:65`
- `vllm/v1/core/kv_cache_manager.py:80`

### 3.2 scheduler 把 `block_ids` 塞进输出

相关位置：

- `vllm/v1/core/sched/scheduler.py:879`
- `vllm/v1/core/sched/scheduler.py:882`
- `vllm/v1/core/sched/scheduler.py:1097`

也就是 scheduler 在构造 `NewRequestData` / `CachedRequestData` 时，会把新分配的 `block_ids` 一起带出去。

### 3.3 model runner 把 `block_ids` 追加到 GPU block table

相关位置：

- `vllm/v1/worker/gpu/model_runner.py:636`
- `vllm/v1/worker/gpu/model_runner.py:663`

新请求用 `overwrite=True`，已有请求继续 decode/prefill 时用 `overwrite=False` 追加。

### 3.4 执行前真正生成 attention 元数据

执行模型前：

- `vllm/v1/worker/gpu/model_runner.py:925`
- `vllm/v1/worker/gpu/model_runner.py:975`

这里会做两件关键事：

1. `gather_block_tables()`
2. `compute_slot_mappings()`

对应：

- `vllm/v1/worker/gpu/model_runner.py:806`
- `vllm/v1/worker/gpu/block_table.py:106`
- `vllm/v1/worker/gpu/block_table.py:133`

然后 `build_slot_mappings_by_layer()` 把 group 级别的 `slot_mapping` 映射给具体 layer：

- `vllm/v1/worker/gpu/attn_utils.py:173`

这点很重要，因为今天 vLLM 支持 hybrid attention，不同 layer 可能属于不同 `kv_cache_group`。

---

## 4. 新 token 是怎么写进 paged KV cache 的

Python 入口：

- `vllm/v1/attention/ops/paged_attn.py:32`

最后调用：

- `vllm/v1/attention/ops/paged_attn.py:42`
- `vllm/_custom_ops.py:2532`
- `csrc/cache_kernels.cu:208`

最关键的几行在 `reshape_and_cache_kernel()`：

- `csrc/cache_kernels.cu:219`
- `csrc/cache_kernels.cu:225`
- `csrc/cache_kernels.cu:226`

逻辑非常直接：

```cpp
slot_idx = slot_mapping[token_idx]
block_idx = slot_idx / block_size
block_offset = slot_idx % block_size
```

然后把 token 的 key/value 拷到：

- key cache 的 `block_idx, block_offset`
- value cache 的 `block_idx, block_offset`

所以从“写 cache”的角度看，`slot_mapping` 已经把“请求内位置 -> 全局物理位置”的问题彻底算完了，kernel 只负责写。

可以把这一步理解成：

```text
token position
  -> slot_mapping
  -> (physical block id, offset inside block)
  -> paged KV cache memory
```

---

## 5. 老的 custom CUDA paged attention kernel 到底怎么实现

### 5.1 launcher 只是做模板分发

Python 包装在：

- `vllm/_custom_ops.py:108`
- `vllm/_custom_ops.py:152`

C++ launcher 在：

- `csrc/attention/paged_attention_v1.cu:46`
- `csrc/attention/paged_attention_v2.cu:46`

它们主要做三件事：

1. 取 tensor 指针和 stride
2. 按 `head_size`、`block_size`、dtype 选择模板实例
3. launch 对应 kernel

比如支持的 `head_size` 和 `block_size` 明确写死在 switch 里：

- `paged_attention_v1.cu:90`
- `paged_attention_v1.cu:145`
- `paged_attention_v2.cu:96`
- `paged_attention_v2.cu:151`

真正的核心逻辑其实都在 `attention_kernels.cuh`。

### 5.2 核心 kernel 的网格含义

`paged_attention_kernel()` 定义在：

- `csrc/attention/attention_kernels.cuh:81`

它的 grid 语义是：

```text
grid = (num_heads, num_seqs, max_num_partitions)
```

也就是说，一个 thread block 处理：

- 一个 sequence
- 一个 query head
- 一个 partition（v2 才有）

相关代码：

- `attention_kernels.cuh:106`
- `attention_kernels.cuh:107`
- `attention_kernels.cuh:145`

### 5.3 先把 query 读进 shared memory

query 读取位置：

- `attention_kernels.cuh:174`
- `attention_kernels.cuh:175`
- `attention_kernels.cuh:179`

关键思想：

- query 对于同一个 `(seq, head)` 会被反复和很多 K 做 dot
- 所以先加载到 shared memory
- 不同 thread group 分别负责 query 的不同切片

### 5.4 通过 `block_table` 找到物理 K/V block

这是 paged attention 最核心的一跳：

- `attention_kernels.cuh:202`
- `attention_kernels.cuh:252`
- `attention_kernels.cuh:389`

也就是：

```cpp
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
physical_block_number = block_table[block_idx];
```

这里的 `block_idx` 是“当前正在处理的逻辑上下文块”，`physical_block_number` 才是真正去全局内存读取 K/V 时使用的物理页号。

换句话说，paged attention 的关键不是“算 attention 的数学公式变了”，而是：

```text
上下文 token 不是连续存的
但通过 block_table，kernel 能把逻辑连续上下文重新映射到离散物理页
```

### 5.5 QK 计算

K 的读取和 QK dot 在这里：

- `attention_kernels.cuh:260`
- `attention_kernels.cuh:268`
- `attention_kernels.cuh:289`

每个 warp 处理若干 block，每个 thread group 处理一个 token 的一部分向量，最后通过 `Qk_dot` 做组内归约：

- `csrc/attention/attention_utils.cuh:29`
- `csrc/attention/attention_utils.cuh:49`

计算出的 `qk` 会写进 shared memory 里的 `logits`：

- `attention_kernels.cuh:187`
- `attention_kernels.cuh:298`

然后做：

1. block 内最大值归约
2. `exp(logits - max)`
3. 求和
4. softmax 归一化

对应：

- `attention_kernels.cuh:305`
- `attention_kernels.cuh:327`
- `attention_kernels.cuh:337`

### 5.6 再读 V，做 softmax * V

V 读取和累加在：

- `attention_kernels.cuh:375`
- `attention_kernels.cuh:397`
- `attention_kernels.cuh:426`

这一段本质就是：

```text
for each attended token:
    acc += softmax(qk[token]) * V[token]
```

最后再做 warp 内、warp 间归约并写出：

- `attention_kernels.cuh:431`
- `attention_kernels.cuh:446`
- `attention_kernels.cuh:478`

### 5.7 这个 kernel 还顺手支持了几件附加能力

从参数可以看出来它还支持：

- ALiBi
  - `attention_kernels.cuh:149`
  - `attention_kernels.cuh:292`
- block sparse
  - `attention_kernels.cuh:204`
  - `attention_kernels.cuh:229`
  - `attention_kernels.cuh:382`
- FP8 KV cache
  - `attention_kernels.cuh:275`
  - `attention_kernels.cuh:406`

所以它不是一个“最小版示例 kernel”，而是已经集成了很多生产特性的版本。

---

## 6. v1 和 v2 到底差在哪

### v1

入口：

- `csrc/attention/paged_attention_v1.cu:160`

特点：

- 不做 partition
- 一个 thread block 处理一个 `(seq, head)` 的完整上下文
- shared memory 里直接放该序列 partition 的 logits / partial outputs

你可以从这里看出 v1 的 shared memory 依赖整段序列长度：

- `paged_attention_v1.cu:78`
- `paged_attention_v1.cu:80`
- `paged_attention_v1.cu:84`

### v2

入口：

- `csrc/attention/paged_attention_v2.cu:167`

特点：

- 把长序列按 `PARTITION_SIZE` 分块
- 第一个 kernel 产出每个 partition 的：
  - `tmp_out`
  - `exp_sums`
  - `max_logits`
- 第二个 kernel 再做跨 partition reduce

关键代码：

- `paged_attention_v2.cu:82`
- `paged_attention_v2.cu:87`
- `paged_attention_v2.cu:90`
- `attention_kernels.cuh:343`
- `attention_kernels.cuh:562`

reduce kernel 的核心就是重新做一次稳定 softmax 合并：

1. 所有 partition 的 `max_logits` 先取全局最大值
2. 各 partition 的 `exp_sums` 根据新的 global max 重标定
3. 再把 `tmp_out` 按权重加起来

对应：

- `attention_kernels.cuh:600`
- `attention_kernels.cuh:633`
- `attention_kernels.cuh:649`

### 一句话理解

- v1：上下文不太长时，单 kernel 一把算完。
- v2：上下文太长时，先分 partition 算局部结果，再合并，避免 shared memory 被整段序列撑爆。

---

## 7. 当前主线 CUDA 路径和“老 paged attention kernel”的关系

这是读这个仓库时最容易混淆的一点。

### 7.1 FlashInfer backend 仍然是分页 KV，只是换了执行器

FlashInfer 会把 `block_table` 转成自己需要的三元组：

- `paged_kv_indptr`
- `paged_kv_indices`
- `paged_kv_last_page_len`

对应：

- `vllm/v1/attention/backends/flashinfer.py:787`
- `vllm/v1/attention/backends/flashinfer.py:804`
- `vllm/v1/attention/backends/flashinfer.py:824`
- `vllm/v1/attention/backends/flashinfer.py:833`

decode 时要么：

- 走 FlashInfer wrapper
  - `flashinfer.py:1151`
  - `flashinfer.py:1157`
  - `flashinfer.py:1574`

要么：

- 走 TRTLLM decode，直接传 `block_tables + seq_lens`
  - `flashinfer.py:1136`
  - `flashinfer.py:1630`

所以主线虽然不一定调用 `torch.ops._C.paged_attention_v1/v2`，但“分页 KV + block table 查页”的思想完全没变。

### 7.2 Triton backend 也是同样思路

metadata 构建时直接保存：

- `block_table`
- `slot_mapping`

对应：

- `vllm/v1/attention/backends/triton_attn.py:241`
- `vllm/v1/attention/backends/triton_attn.py:247`
- `vllm/v1/attention/backends/triton_attn.py:248`

decode 时把 `block_table` 交给 `unified_attention()`：

- `triton_attn.py:602`
- `triton_attn.py:616`

写 cache 时把 `slot_mapping` 交给 Triton reshape kernel：

- `triton_attn.py:682`
- `triton_attn.py:716`
- `triton_attn.py:721`

所以从架构层面看，今天的 vLLM 更像是：

```text
Paged KV cache / block_table / slot_mapping
    + 多种 attention backend (FlashInfer / Triton / TRTLLM / ROCm ...)
```

而不是“只有老的 paged_attention_v1/v2 这一套”。

---

## 8. 我认为最值得记住的实现要点

### 要点 1

`PagedAttention` 的本质不是新公式，而是“逻辑连续上下文”和“物理离散缓存页”之间的映射。

### 要点 2

真正把这个映射落地成地址计算的是两样东西：

- 读时：`block_table`
- 写时：`slot_mapping`

### 要点 3

旧 custom kernel 的核心只有一句：

```cpp
physical_block_number = block_table[logical_block_idx];
```

后面所有 K/V 读取都围绕这个物理块号展开。

### 要点 4

v1 和 v2 的差别主要不是数学，而是工程上的“怎么把长序列拆开算，避开 shared memory 限制”。

### 要点 5

今天的主线 CUDA 路径已经不一定直接走这两个老 kernel，但底层分页缓存思想仍然是 vLLM 的核心。

---

## 9. 如果你只想精读 30 分钟

建议顺序：

1. `vllm/v1/worker/gpu/block_table.py`
   - 先理解 `block_table` 和 `slot_mapping` 怎么算。
2. `csrc/cache_kernels.cu`
   - 看 `reshape_and_cache_kernel()` 怎么把 token 写进 paged cache。
3. `csrc/attention/attention_kernels.cuh`
   - 看 `paged_attention_kernel()` 怎么通过 `block_table` 读离散 K/V。
4. `csrc/attention/paged_attention_v1.cu` / `v2.cu`
   - 看 launcher、模板分发和 v1/v2 差异。
5. `vllm/v1/attention/backends/flashinfer.py`
   - 看今天主线 backend 怎么复用同样的分页元数据。

---

## 10. 一个简化后的伪代码

```python
# 1) scheduler / kv manager
block_ids = allocate_slots(request)
block_table[req] = block_ids

# 2) 写 KV cache
slot = block_table[req][pos // block_size] * block_size + (pos % block_size)
kv_cache[slot] = new_kv

# 3) decode attention
for head in heads:
    q = query[req, head]
    for logical_block in range(num_context_blocks):
        physical_block = block_table[req][logical_block]
        K_block = key_cache[physical_block, ...]
        V_block = value_cache[physical_block, ...]
        logits += q @ K_block.T
    probs = softmax(logits)
    out = probs @ V
```

真正的 CUDA/Triton 实现当然复杂得多，但抽象层面就是这件事。

---

## 11. 把 `paged_attention_kernel()` 的几个模板参数拆开看

下面几个参数是理解 kernel 的关键：

- `HEAD_SIZE`
  - 每个 attention head 的维度，比如 128。
- `BLOCK_SIZE`
  - 一个 KV page/block 里放多少 token，比如 16。
- `NUM_THREADS`
  - 一个 CUDA thread block 的线程数，默认 128。
- `PARTITION_SIZE`
  - 仅 v2 使用，把长序列切成 partition，比如 512。

然后 kernel 自己会推导出一批二级参数：

- `THREAD_GROUP_SIZE`
  - `max(WARP_SIZE / BLOCK_SIZE, 1)`
  - 代码在 `csrc/attention/attention_kernels.cuh:133`
- `VEC_SIZE`
  - `max(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1)`
  - 代码在 `csrc/attention/attention_kernels.cuh:157`
- `x`
  - `16 / sizeof(cache_t)`
  - 代码在 `csrc/attention/attention_kernels.cuh:195`

### 11.1 这些量直觉上是什么意思

`THREAD_GROUP_SIZE` 可以理解成：

- 做一次 QK dot 时，多少线程合作处理“同一个 token 的同一个 head”

`VEC_SIZE` 可以理解成：

- 每个线程一次向量化加载多少个标量元素

`x` 可以理解成：

- key cache 在最后一维按多大粒度分块
- 对于 FP16/BF16，`sizeof(cache_t)=2`，所以 `x=8`
- 这意味着 K cache 的最后两维相当于把 head 维度切成了很多个 8 元素小块

### 11.2 支持的 block size 下，这些值分别是多少

对于这个 kernel 支持的 `BLOCK_SIZE in {8, 16, 32}`，并假设 `NUM_THREADS=128`：

| BLOCK_SIZE | THREAD_GROUP_SIZE | NUM_THREAD_GROUPS | NUM_TOKENS_PER_THREAD_GROUP |
| --- | --- | --- | --- |
| 8 | 4 | 32 | 1 |
| 16 | 2 | 64 | 1 |
| 32 | 1 | 128 | 1 |

这张表最重要的启发是：

- 对 `BLOCK_SIZE <= 32` 的常见情况，每个 thread group 只负责 1 个 token。
- 所以一个 warp 里会有：
  - `32 / THREAD_GROUP_SIZE` 个 thread group
  - 也就正好覆盖这个 block 里的所有 token

举例：

- `BLOCK_SIZE=16`
  - `THREAD_GROUP_SIZE=2`
  - 一个 warp 有 `32 / 2 = 16` 个 thread group
  - 正好 1 个 warp 覆盖 16 个 token
- `BLOCK_SIZE=32`
  - `THREAD_GROUP_SIZE=1`
  - 一个 warp 有 32 个 thread group
  - 正好 1 个 warp 覆盖 32 个 token

这就是为什么代码注释里说：

- 一个 warp 一次处理一个 block
- 一个 thread group 处理 block 里的一个 token

对应：

- `csrc/attention/attention_kernels.cuh:198`
- `csrc/attention/attention_kernels.cuh:199`
- `csrc/attention/attention_kernels.cuh:200`

---

## 12. 用一个具体配置走一遍

我们拿最典型的一组参数：

- `HEAD_SIZE = 128`
- `BLOCK_SIZE = 16`
- `NUM_THREADS = 128`
- `scalar_t = fp16`
- `cache_t = fp16`

代进去后：

- `THREAD_GROUP_SIZE = 32 / 16 = 2`
- `VEC_SIZE = 16 / (2 * 2) = 4`
- `x = 16 / 2 = 8`
- `NUM_WARPS = 128 / 32 = 4`
- `NUM_ELEMS_PER_THREAD = 128 / 2 = 64`
- `NUM_VECS_PER_THREAD = 64 / 4 = 16`

这几个数特别值得记住：

- 一个 token 的 128 维 query/key，会由 2 个线程合作处理。
- 每个线程一次拿 4 个元素。
- 两个线程合起来一次正好拿 8 个元素，也就是一个 `x` 小块。

### 12.1 一个 warp 里线程怎么分 token

因为 `THREAD_GROUP_SIZE=2`：

- lane 0,1 是第 0 个 thread group
- lane 2,3 是第 1 个 thread group
- ...
- lane 30,31 是第 15 个 thread group

而 `BLOCK_SIZE=16`，所以：

- 这 16 个 thread group，刚好对应 block 中 16 个 token

也就是说，一个 warp 会同时在做 16 个 token 的 QK。

### 12.2 为什么 query 要先放 shared memory

因为对同一个 `(seq, head)` 来说：

- query 是固定的
- 它要和这个上下文里的所有 key token 去做 dot

所以先把 query 放进 shared memory：

- 所有 warp / thread group 后续都能复用
- 避免重复从 global memory 读同一个 query

相关代码：

- `csrc/attention/attention_kernels.cuh:174`
- `csrc/attention/attention_kernels.cuh:175`
- `csrc/attention/attention_kernels.cuh:183`

---

## 13. K 的地址到底是怎么算出来的

这是最容易“看花”的地方。

### 13.1 先看 K cache 形状

K cache 形状是：

```text
[num_blocks, num_kv_heads, head_size/x, block_size, x]
```

对于我们这个例子：

- `head_size = 128`
- `x = 8`

所以实际上是：

```text
[num_blocks, num_kv_heads, 16, 16, 8]
```

可以理解成：

- 一个物理 block
- 一个 kv head
- 里面有 16 个“小列块”
- 每个 token 在每个小列块中有 8 个连续元素

### 13.2 `k_ptr` 指向的是什么

关键代码：

- `csrc/attention/attention_kernels.cuh:268`
- `csrc/attention/attention_kernels.cuh:269`
- `csrc/attention/attention_kernels.cuh:270`

```cpp
const cache_t* k_ptr =
    k_cache + physical_block_number * kv_block_stride +
    kv_head_idx * kv_head_stride + physical_block_offset * x;
```

这个 `k_ptr` 的含义不是“整个 token 的起点”，而是：

- 某个物理 block
- 某个 kv head
- 某个 token offset
- 在 K cache 压缩布局下，这个 token 对应的基础地址

### 13.3 `offset1` / `offset2` 到底在干嘛

关键代码：

- `csrc/attention/attention_kernels.cuh:271`
- `csrc/attention/attention_kernels.cuh:272`
- `csrc/attention/attention_kernels.cuh:273`

```cpp
const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
const int offset1 = (vec_idx * VEC_SIZE) / x;
const int offset2 = (vec_idx * VEC_SIZE) % x;
```

对于我们的例子：

- `THREAD_GROUP_SIZE=2`
- `VEC_SIZE=4`
- `x=8`

#### thread 0

`vec_idx` 依次是：

```text
0, 2, 4, 6, ..., 30
```

于是：

- `offset1 = 0, 1, 2, ..., 15`
- `offset2 = 0`

#### thread 1

`vec_idx` 依次是：

```text
1, 3, 5, 7, ..., 31
```

于是：

- `offset1 = 0, 1, 2, ..., 15`
- `offset2 = 4`

这意味着：

- thread 0 负责每个 8 元素小块里的前 4 个元素
- thread 1 负责每个 8 元素小块里的后 4 个元素

两条线程加起来，正好把一个 token 的整条 128 维 key 分块读完。

这个设计非常漂亮，因为：

- 每个 thread group 合起来正好凑 16 bytes 粒度读取
- 相邻线程访问相邻地址
- 很利于 global memory coalescing

---

## 14. QK 阶段到底是谁在做 reduction

看这句：

- `csrc/attention/attention_kernels.cuh:289`
- `csrc/attention/attention_utils.cuh:31`
- `csrc/attention/attention_utils.cuh:43`

`Qk_dot<>::dot()` 做了两层事：

1. 每个线程先对自己手上的向量片段做乘加
2. 同一个 thread group 内再做 `shuffle xor` 归约

所以在 `BLOCK_SIZE=16` 的例子下：

- 一个 token 的 QK dot 由 2 个线程合作
- 最终只有 `thread_group_offset == 0` 的那个线程拿到完整 `qk`

所以代码里才会看到：

- `if (thread_group_offset == 0) logits[...] = qk`

对应：

- `csrc/attention/attention_kernels.cuh:294`
- `csrc/attention/attention_kernels.cuh:298`

### 14.1 warp 内最大值归约

这一步不是在“一个 token 内”归约，而是在“一个 warp 当前负责的一批 token”里归约最大 logit：

- `csrc/attention/attention_kernels.cuh:305`
- `csrc/attention/attention_kernels.cuh:309`

### 14.2 再做 thread block 级别归约

每个 warp 先把自己的最大值写到 shared memory：

- `csrc/attention/attention_kernels.cuh:312`
- `csrc/attention/attention_kernels.cuh:313`

然后再跨 warp 归约成整个 sequence/partition 的 `qk_max`：

- `csrc/attention/attention_kernels.cuh:317`
- `csrc/attention/attention_kernels.cuh:321`

softmax 的 `exp_sum` 也是同样思路：

- 先每个线程积自己的局部和
- 再调用 `block_sum<NUM_WARPS>()`

对应：

- `csrc/attention/attention_kernels.cuh:327`
- `csrc/attention/attention_kernels.cuh:334`
- `csrc/attention/attention_kernels.cuh:44`

---

## 15. V 阶段为什么又是另一套线程组织

QK 阶段和 V 阶段的线程映射并不一样，这是很多人第一次读会困惑的点。

### 15.1 QK 阶段的目标

QK 阶段更适合：

- 一个 thread group 对一个 token
- 把 query 和 key 分块做 dot

### 15.2 V 阶段的目标

V 阶段更适合：

- 让一组 lane 覆盖一行 value
- 每个 lane 拿该行的一段 token
- 用 softmax 权重做加权和

这就是下面这些参数的含义：

- `V_VEC_SIZE`
- `NUM_V_VECS_PER_ROW`
- `NUM_ROWS_PER_ITER`
- `NUM_ROWS_PER_THREAD`

对应：

- `csrc/attention/attention_kernels.cuh:355`
- `csrc/attention/attention_kernels.cuh:361`
- `csrc/attention/attention_kernels.cuh:362`
- `csrc/attention/attention_kernels.cuh:363`

### 15.3 继续拿 `BLOCK_SIZE=16, HEAD_SIZE=128` 的例子

此时：

- `V_VEC_SIZE = min(16 / 2, 16) = 8`
- `NUM_V_VECS_PER_ROW = 16 / 8 = 2`
- `NUM_ROWS_PER_ITER = 32 / 2 = 16`
- `NUM_ROWS_PER_THREAD = ceil(128 / 16) = 8`

这说明：

- 一行 V 会被 2 个 lane 合作处理
- 一个 warp 一轮能覆盖 16 行
- 因为 head size 是 128，所以每个 lane 要处理 8 轮行

具体地：

- `lane % 2` 决定读这行里 offset 是 `0` 还是 `8`
- `lane / 2` 决定当前是第几行

对应代码：

- `csrc/attention/attention_kernels.cuh:391`
- `csrc/attention/attention_kernels.cuh:401`
- `csrc/attention/attention_kernels.cuh:403`

所以在 V 阶段：

- lane 0/1 一起负责 row 0
- lane 2/3 一起负责 row 1
- ...
- lane 30/31 一起负责 row 15

然后通过：

- `dot(logits_vec, v_vec)`

把一个 token block 上的 softmax 权重和 V 向量片段做加权求和：

- `csrc/attention/attention_kernels.cuh:393`
- `csrc/attention/attention_kernels.cuh:426`

### 15.4 warp 内先合并“同一行”的两个 lane

因为每一行由两个 lane 分别处理 offset `0..7` 和 `8..15`，所以先在 warp 内做一次 XOR reduction：

- `csrc/attention/attention_kernels.cuh:431`
- `csrc/attention/attention_kernels.cuh:436`

这样每一行的 partial sum 就先合起来了。

### 15.5 再跨 warp 合并整个 head

因为一个 thread block 有 4 个 warp，最终还要把 4 个 warp 的结果叠起来：

- `csrc/attention/attention_kernels.cuh:446`
- `csrc/attention/attention_kernels.cuh:452`
- `csrc/attention/attention_kernels.cuh:465`

最后只有 `warp_idx == 0` 负责写最终输出：

- `csrc/attention/attention_kernels.cuh:479`

---

## 16. 为什么 K 和 V 的布局不一样

如果只看数学，K/V 都是 `[token, head_dim]`。

但工程上它们的最优访问模式不同：

- K 是为了做 `q @ k`
  - 更适合把 head_dim 按 `x` 分块，方便 thread group 向量化读取
- V 是为了做 `softmax(qk) @ v`
  - 更适合按 `[row=head_dim, col=token_in_block]` 的方式读取

所以旧 custom kernel 才故意使用：

- K:
  - `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
- V:
  - `[num_blocks, num_kv_heads, head_size, block_size]`

如果你把这一点看明白，后面很多看起来“古怪”的地址计算其实就顺了。

---

## 17. 我认为读这个 kernel 时最该盯的 6 行

如果你后面想自己重新精读代码，我建议反复盯这 6 组地方：

1. block 范围和 token 范围
   - `csrc/attention/attention_kernels.cuh:121`
   - `csrc/attention/attention_kernels.cuh:128`
2. 线程组参数
   - `csrc/attention/attention_kernels.cuh:133`
   - `csrc/attention/attention_kernels.cuh:157`
3. 逻辑块到物理块
   - `csrc/attention/attention_kernels.cuh:202`
   - `csrc/attention/attention_kernels.cuh:252`
4. K 地址拆分
   - `csrc/attention/attention_kernels.cuh:268`
   - `csrc/attention/attention_kernels.cuh:272`
5. softmax 主体
   - `csrc/attention/attention_kernels.cuh:327`
   - `csrc/attention/attention_kernels.cuh:337`
6. V 聚合和最终写出
   - `csrc/attention/attention_kernels.cuh:397`
   - `csrc/attention/attention_kernels.cuh:479`

把这几组地方串起来，整个 kernel 的主干就基本拿下了。
