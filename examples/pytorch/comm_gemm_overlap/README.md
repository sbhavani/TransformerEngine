# Overlapping Communication with GEMM in Transformer Engine Modules

Communication+GEMM overlap is a critical optimization technique for distributed training that hides communication latencies by overlapping them with computation. Transformer Engine provides built-in support for this optimization through **Userbuffers**, enabling significant speedup in large-scale tensor-parallel training.

## What is Comm+GEMM Overlap?

In distributed DL training, communication operations (like AllGather and ReduceScatter) block computation, causing GPUs to idle while data is being exchanged. Comm+GEMM overlap addresses this by:

1. **Pipelining**: Breaking down large operations into smaller chunks
2. **Overlapping**: Running communication for one chunk while computing GEMM for another
3. **Memory Management**: Using dedicated communication buffers (Userbuffers) for efficient data movement

Our implementation includes multiple advanced algorithms:
- **Pipeline-based overlap** (`split_overlap_ag/rs`) for chunked operations
- **Ring-exchange P2P** communication for distributed tensor-parallel groups  
- **Atomic GEMM overlap** for fine-grained synchronization
- **Bulk overlap** patterns for large-scale communication

## Quick Start

### Basic TE Layer with Overlap

```bash
# Single node with all GPUs
torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py

# With FP8
torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py --fp8

# Without overlap (for comparison)
torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py --no-comm-overlap
```

## Requirements and Environment Setup

### Hardware Requirements

- **Single Node**: All tensor-parallel GPUs must be on the same node
- **NVLink/NVSwitch**: Required for high-bandwidth GPU-to-GPU communication
- **Compute Capability**: 9.0+ recommended (H100 or newer)
  - Devices older than compute capability 9.0 require `UB_SKIPMC=1` in the environment in order to fall back on a less performant implementation based on CUDA Inter-Process Communication (IPC).

### Environment Variables

```bash
# Essential for overlap functionality
export CUDA_DEVICE_MAX_CONNECTIONS=1

# For older GPUs (compute capability < 9.0)
export UB_SKIPMC=1

# Optional: Debugging and timeout
export TE_DEBUG=1                    # Enable verbose logging
export UB_TIMEOUT=300               # Increase timeout for debugging
```

### CUDA Requirements

- **CUDA Toolkit 12.0+** and **CUDA driver 535+** for optimal performance (CUDA Multicast)

## How to Enable Overlap in TE Modules

### 1. Automatic Overlap (Recommended)

Transformer Engine modules automatically detect and enable overlap when configured properly:

```python
import transformer_engine.pytorch as te

# Initialize userbuffers globally (required)
te.module.base.initialize_ub(
    shape=[seq_length * batch_size, hidden_size],  # Communication buffer shape
    tp_size=tp_size,                               # Tensor parallel size
    use_fp8=True,                                  # Enable FP8 if supported
    dtype=torch.bfloat16,                         # Buffer data type
    bootstrap_backend="nccl",                     # Bootstrap communication backend
)

# Create TE layers with overlap enabled
layer = te.TransformerLayer(
    hidden_size=4096,
    ffn_hidden_size=11008,
    num_attention_heads=32,
    tp_group=tp_group,                    # Tensor parallel process group
    tp_size=tp_size,                      # Tensor parallel size
    sequence_parallel=True,               # Enable sequence parallelism
    # Overlap configuration (enabled by default when userbuffers are initialized)
    ub_tp_comm_overlap=True,             # Enable tensor-parallel communication overlap
    ub_overlap_ag=True,                  # Enable AllGather overlap
    ub_overlap_rs=True,                  # Enable ReduceScatter overlap
    ub_bulk_wgrad=True,                  # Bulk weight gradient communication
    ub_bulk_dgrad=True,                  # Bulk data gradient communication
)
```

### 2. Manual Overlap Control

For fine-grained control, you can specify overlap parameters for individual layers:

```python
# Fine-grained overlap configuration
layer_config = {
    # Core overlap settings
    "ub_tp_comm_overlap": True,          # Master switch for TP overlap
    "ub_overlap_ag": True,               # AllGather overlap in forward pass
    "ub_overlap_rs": True,               # ReduceScatter overlap in backward pass
    "ub_overlap_rs_dgrad": False,        # ReduceScatter overlap for data gradients
    
    # Bulk communication settings (more efficient for large tensors)
    "ub_bulk_wgrad": True,               # Bulk weight gradient all-reduce
    "ub_bulk_dgrad": True,               # Bulk data gradient reduce-scatter
    
    # Layer-specific naming (for debugging)
    "ub_name": "attention_qkv",          # Unique name for this layer's buffers
}

# Apply to specific layers
attention_layer = te.MultiheadAttention(
    hidden_size=4096,
    num_attention_heads=32,
    **layer_config
)
```

### 3. Buffer Configuration

```python
# Calculate buffer size
seq_length = 2048
batch_size = 8
hidden_size = 4096

# For sequence parallel
if sequence_parallel:
    buffer_elements = (seq_length * batch_size) // tp_size
else:
    buffer_elements = seq_length * batch_size

te.module.base.initialize_ub(
    shape=[buffer_elements, hidden_size],
    tp_size=tp_size,
    use_fp8=True,
    dtype=torch.bfloat16,
    bootstrap_backend="nccl",
)
```

## Parameter Reference

### Core Overlap Parameters

| Parameter | Default | Description | When to Enable |
|-----------|---------|-------------|----------------|
| `ub_tp_comm_overlap` | `False` | Master switch for tensor-parallel overlap | Always for TP training |
| `ub_overlap_ag` | `True` | AllGather overlap in forward pass | Forward pass optimization |
| `ub_overlap_rs` | `True` | ReduceScatter overlap in backward pass | Backward pass optimization |
| `ub_bulk_wgrad` | `True`* | Bulk weight gradient communication | Large models |
| `ub_bulk_dgrad` | `True`* | Bulk data gradient communication | Large batches/sequences |
| `ub_overlap_rs_dgrad` | `False` | ReduceScatter for data gradients | Advanced optimization |
| `ub_name` | `None` | Unique identifier for debugging | Multi-layer debugging |

*When `ub_tp_comm_overlap=True`, these parameters default to `True`. When `ub_tp_comm_overlap=False`, they are disabled regardless of their values.

### Monitoring and Debugging

```python
# Enable detailed logging
os.environ["TE_DEBUG"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

# Enable userbuffers debugging (shows detailed buffer allocation and communication info)
os.environ["NVTE_UBDEBUG"] = "1"

# Enable NVTX markers for Nsight Systems profiling
os.environ["NVTE_NVTX_ENABLED"] = "1"

# Monitor GPU utilization
# Use: nvidia-smi dmon -s pucvmet -d 1

# Profile with Nsight Systems (shows communication+GEMM overlap in timeline)
# nsys profile -o overlap_profile python your_script.py

# Use unique names for layers to track buffer usage and identify operations
layer = te.TransformerLayer(
    # ... other args ...
    ub_name=f"layer_{layer_idx}",  # Helps identify which layer's buffers in profiler
)
```
## Examples

### Single node, tensor-parallel LayerNormMLP:

Forward and backward passes with layer weights distributed over all GPUs in a single node.

```bash
$ torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py

# Sample output on 8x H100s:
#   [rank0:node0] |-- Created tensor-parallel group: [0, 1, 2, 3, 4, 5, 6, 7]
#   !!! [UB] Create UbufP2PCommOverlap Communicator
#   UB_TIMEOUT is set to 110 sec, 217800000000 cycles, freq: 1980000khz
#   MC initialized succesfully, window size = 549755813888
#   !!! [UBP2P] Register UBuf 1
#   !!! [UBP2P] Register UBuf 2
#   !!! [UBP2P] Register UBuf 3
#   !!! [UBP2P] Register UBuf 4
#   !!! [UB] Register UBuf 5
#   !!! [UBP2P] Register UBuf 6
#   !!! [UB] Register UBuf 7
#   !!! [UB] Register UBuf 8
#   !!! [UBP2P] Register UBuf 9
#   !!! [UB] Register UBuf 10
#   [rank0:node0] Iter 1
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 2
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 3
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 4
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 5
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
```

### Single node, mixed data- and tensor-parallel LayerNormMLP:

Uses `torch.nn.parallel.DistributedDataParallel` for replicating the model across 2 tensor-parallel groups in a single node.

```bash
$ torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py --num-replicas 2

# Sample output on 8x H100s:
#   [rank0:node0] |-- Created tensor-parallel group: [0, 1, 2, 3]
#   [rank4:node1] |-- Created tensor-parallel group: [4, 5, 6, 7]
#   [rank0:node0] |-- Created data-parallel group: [0, 4]
#   [rank3:node1] |-- Created data-parallel group: [3, 7]
#   [rank1:node1] |-- Created data-parallel group: [1, 5]
#   [rank2:node0] |-- Created data-parallel group: [2, 6]
#   !!! [UB] Create UbufP2PCommOverlap Communicator
#   UB_TIMEOUT is set to 110 sec, 217800000000 cycles, freq: 1980000khz
#   MC initialized succesfully, window size = 549755813888
#   !!! [UBP2P] Register UBuf 1
#   !!! [UBP2P] Register UBuf 2
#   !!! [UBP2P] Register UBuf 3
#   !!! [UBP2P] Register UBuf 4
#   !!! [UB] Register UBuf 5
#   !!! [UBP2P] Register UBuf 6
#   !!! [UB] Register UBuf 7
#   !!! [UB] Register UBuf 8
#   !!! [UBP2P] Register UBuf 9
#   !!! [UB] Register UBuf 10
#   [rank4:node1] Iter 1
#   [rank0:node0] Iter 1
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] Iter 2
#   [rank0:node0] Iter 2
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank4:node1] |-- Forward pass
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Backward pass
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] Iter 3
#   [rank0:node0] Iter 3
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Backward pass
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] Iter 4
#   [rank4:node1] Iter 4
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Compute loss
#   [rank4:node1] |-- Backward pass
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] Iter 5
#   [rank0:node0] Iter 5
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
```

**NOTE:** To run with FP8, add the `--fp8` flag to the commands shown above.
