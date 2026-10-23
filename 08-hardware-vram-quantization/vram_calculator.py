"""vram_calculator.py - GPU Memory & KV Cache Footprint Calculator from Scratch.

This module implements the exact hardware and memory accounting formulas for LLM serving:
1. Static Weight VRAM:
   - FP32: 4 bytes / param
   - FP16 / BF16: 2 bytes / param
   - INT8: 1 byte / param
   - INT4: 0.5 bytes / param
2. Dynamic KV Cache Footprint:
   Formula:
     VRAM_KV = 2 * Batch * Context_Length * n_layers * n_kv_heads * d_head * bytes_per_elem
     (Factor of 2 accounts for storing both Key and Value matrices).
3. Attention Architecture Scaling:
   - Multi-Head Attention (MHA): n_kv_heads == n_q_heads (memory explodes linearly).
   - Grouped-Query Attention (GQA): n_kv_heads == n_q_heads / group_ratio (e.g. 4x or 8x reduction).
   - Multi-Query Attention (MQA): n_kv_heads == 1.
4. Total Serving VRAM:
   Total = Static_Weights + Dynamic_KV_Cache + Activation_Buffers + Runtime_Overhead.
   Activation and runtime overhead depend on the engine, batch size and kernels; the defaults below
   (1.5 GiB and 1.2 GiB) are assumptions to replace with a measurement on your stack.
5. Capacity is not speed. Generating one token at batch size 1 reads every weight (and the whole
   KV cache) from memory once, so memory bandwidth caps decode speed:
     tokens/s <= bandwidth / (weight_bytes + kv_bytes)
   This is an upper bound; real engines reach a fraction of it. Prefill (processing the prompt) is
   usually limited by compute instead, and larger batches share each weight read across requests.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelArchitecture:
    name: str
    total_params: int       # Total parameters (e.g. 8_030_000_000 for Llama 3 8B)
    num_layers: int         # Transformer layers
    hidden_size: int        # d_model
    num_q_heads: int        # Number of query heads
    num_kv_heads: int       # Number of key/value heads (GQA)
    head_dim: int           # Dimension per head (hidden_size / num_q_heads)


# Ground-truth reference models
# Llama 3 8B and Llama 3.1 8B share this architecture; Llama 3.1 extended the context window to 128k.
LLAMA_3_8B = ModelArchitecture(
    name="Llama-3.1-8B",
    total_params=8_030_000_000,
    num_layers=32,
    hidden_size=4096,
    num_q_heads=32,
    num_kv_heads=8,         # 4x GQA reduction (32 Q heads / 8 KV heads)
    head_dim=128,
)

LLAMA_3_70B = ModelArchitecture(
    name="Llama-3.1-70B",
    total_params=70_600_000_000,
    num_layers=80,
    hidden_size=8192,
    num_q_heads=64,
    num_kv_heads=8,         # 8x GQA reduction (64 Q heads / 8 KV heads)
    head_dim=128,
)


def calculate_weight_vram(params: int, precision_bits: int) -> float:
    """Calculates static model weight memory in Gigabytes (GB, 10^9 bytes) and GiB (2^30 bytes).

    precision_bits: 32 (FP32), 16 (FP16/BF16), 8 (INT8), 4 (INT4/GGUF).
    """
    bytes_per_param = precision_bits / 8.0
    total_bytes = params * bytes_per_param
    gib = total_bytes / (1024 ** 3)
    return gib


def calculate_kv_cache_vram(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_length: int,
    batch_size: int = 1,
    precision_bits: int = 16,
) -> float:
    """Calculates dynamic Key-Value Cache VRAM in GiB (2^30 bytes).

    Formula:
      Total_Bytes = 2 * Batch * SeqLen * Num_Layers * Num_KV_Heads * Head_Dim * Bytes_Per_Element
    """
    bytes_per_element = precision_bits / 8.0
    # 2 accounts for both Key and Value tensors:
    total_bytes = 2 * batch_size * context_length * num_layers * num_kv_heads * head_dim * bytes_per_element
    gib = total_bytes / (1024 ** 3)
    return gib


def calculate_total_serving_vram(
    model: ModelArchitecture,
    context_length: int,
    batch_size: int = 1,
    weight_bits: int = 16,
    kv_cache_bits: int = 16,
    cuda_overhead_gib: float = 1.2,
    activation_gib: float = 1.5,
) -> dict[str, float]:
    """Calculates full runtime VRAM budget including static weights, KV cache, and overhead."""
    weights_gib = calculate_weight_vram(model.total_params, weight_bits)
    kv_gib = calculate_kv_cache_vram(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        context_length=context_length,
        batch_size=batch_size,
        precision_bits=kv_cache_bits,
    )
    total_gib = weights_gib + kv_gib + cuda_overhead_gib + activation_gib
    return {
        "weights_gib": round(weights_gib, 2),
        "kv_cache_gib": round(kv_gib, 2),
        "cuda_overhead_gib": cuda_overhead_gib,
        "activation_gib": activation_gib,
        "total_serving_gib": round(total_gib, 2),
    }


def decode_tokens_per_second_upper_bound(weights_gib: float, kv_gib: float, bandwidth_gb_per_s: float) -> float:
    """Batch-1 decode ceiling: each token reads all weights plus the KV cache once from memory."""
    bytes_per_token = (weights_gib + kv_gib) * (1024 ** 3)
    return bandwidth_gb_per_s * 1e9 / bytes_per_token


def run_verification_suite() -> None:
    """Verifies weight VRAM arithmetic, GQA reduction ratio, and 128k context scaling."""
    eps = 0.05

    # 1. Verify Weight Sizing for Llama-3-8B across precisions
    fp16_weights = calculate_weight_vram(LLAMA_3_8B.total_params, 16)
    int8_weights = calculate_weight_vram(LLAMA_3_8B.total_params, 8)
    int4_weights = calculate_weight_vram(LLAMA_3_8B.total_params, 4)

    # 8.03B * 2 bytes = 16.06 GB = ~14.96 GiB
    assert abs(fp16_weights - 14.96) < eps, f"Expected ~14.96 GiB for FP16, got {fp16_weights:.2f}"
    # INT8 should be exactly half of FP16
    assert abs(int8_weights - (fp16_weights / 2.0)) < eps
    # INT4 should be exactly a quarter of FP16 (~3.74 GiB)
    assert abs(int4_weights - (fp16_weights / 4.0)) < eps
    assert abs(int4_weights - 3.74) < eps

    # 2. Verify Dynamic KV Cache Calculation for Llama-3-8B at 128k Context
    # Batch=1, SeqLen=131,072 (128k), Layers=32, KV_Heads=8, Head_Dim=128, FP16 (2 bytes)
    # Total = 2 * 1 * 131072 * 32 * 8 * 128 * 2 = 17,179,869,184 bytes = exactly 16.00 GiB!
    kv_128k = calculate_kv_cache_vram(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        context_length=131_072,
        batch_size=1,
        precision_bits=16,
    )
    assert abs(kv_128k - 16.00) < eps, f"Expected exactly 16.00 GiB for 128k KV cache, got {kv_128k:.2f}"

    # 3. Verify GQA vs. MHA Memory Reduction Invariant:
    # If Llama-3 used naive MHA (32 KV heads instead of 8), KV cache would be 4x larger (64 GiB)
    kv_128k_mha = calculate_kv_cache_vram(
        num_layers=32,
        num_kv_heads=32,  # MHA
        head_dim=128,
        context_length=131_072,
        batch_size=1,
        precision_bits=16,
    )
    assert abs(kv_128k_mha - 64.00) < eps
    ratio = kv_128k_mha / kv_128k
    assert abs(ratio - 4.0) < eps, f"GQA 32/8 must yield exactly 4x memory reduction, got {ratio}"

    # 4. Total Budget Assertion: Proves that 8B model in FP16 at 128k exceeds a 24GB GPU (RTX 4090/3090)
    # Weights (~14.96 GiB) + KV Cache (16.00 GiB) + Overhead (1.2) + Activations (1.5) = ~33.66 GiB
    budget_128k = calculate_total_serving_vram(LLAMA_3_8B, context_length=131_072, weight_bits=16)
    assert budget_128k["total_serving_gib"] > 24.0, "FP16 8B at 128k must exceed 24GB consumer GPU"

    # INT4 weights + FP8 KV cache: 3.74 + 8.00 + 1.2 + 1.5 = ~14.44 GiB -> fits in a 16 GB card, barely
    budget_quantized = calculate_total_serving_vram(
        LLAMA_3_8B, context_length=131_072, weight_bits=4, kv_cache_bits=8
    )
    assert budget_quantized["total_serving_gib"] < 16.0, "Quantized 8B at 128k must fit in 16GB GPU"

    # 5. Capacity is not speed: at ~1,000 GB/s, FP16 8B decodes at most ~62 tokens/s with an empty cache,
    #    ~30 tokens/s at 128k (the KV cache is read too), and INT4 weights raise the ceiling ~4x.
    bw = 1_000.0
    fp16_empty = decode_tokens_per_second_upper_bound(fp16_weights, 0.0, bw)
    fp16_long = decode_tokens_per_second_upper_bound(fp16_weights, kv_128k, bw)
    int4_empty = decode_tokens_per_second_upper_bound(int4_weights, 0.0, bw)
    assert 61 < fp16_empty < 63 and 29 < fp16_long < 31 and 240 < int4_empty < 260, (fp16_empty, fp16_long, int4_empty)

    print("[OK] VRAM calculator: weight math, GQA 4x reduction, 128k KV footprint, and the bandwidth ceiling verified.")


if __name__ == "__main__":
    print("Executing GPU VRAM & KV Cache Memory Verification...")
    run_verification_suite()

    # Demonstrative printout for engineers sizing infrastructure
    print(f"\n--- Serving Budget: {LLAMA_3_8B.name} (Batch Size = 1) ---")
    for ctx in [2_048, 8_192, 32_768, 131_072]:
        b = calculate_total_serving_vram(LLAMA_3_8B, context_length=ctx, weight_bits=16, kv_cache_bits=16)
        ceiling = decode_tokens_per_second_upper_bound(b["weights_gib"], b["kv_cache_gib"], 1_000.0)
        print(f"Context: {ctx:6d} tokens | Weights: {b['weights_gib']:5.1f} GiB | KV Cache: {b['kv_cache_gib']:5.1f} GiB | "
              f"Total: {b['total_serving_gib']:5.1f} GiB | Decode ceiling @1 TB/s: {ceiling:5.1f} tok/s")
    print("(Totals assume 1.5 GiB activations and 1.2 GiB runtime overhead; measure yours.)")
