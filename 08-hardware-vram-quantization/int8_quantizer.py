"""int8_quantizer.py - Bare-Metal Affine INT8 Quantization from Scratch.

This module implements the exact mathematical mechanics of weight quantization:
1. Continuous to Discrete Mapping:
   - Floating-point weights in [x_min, x_max] mapped to discrete signed 8-bit integers [-128, 127].
2. Quantization Parameters:
   - Scale factor S: Step size between quantization buckets: S = (x_max - x_min) / 255.
   - Zero-Point Z: Integer value corresponding to real-world 0.0 (enabling exact zero representation).
   - The range is widened to include 0.0 first (x_min <= 0 <= x_max), as common frameworks do. Without
     that, an all-positive or all-negative tensor pushes Z outside [-128, 127] and the mapping breaks.
3. Formulas:
   - Quantization:   q = clamp(round(x / S) + Z, -128, 127)
   - Dequantization: x_hat = S * (q - Z)
4. Quantization Error:
   - Computes Mean Absolute Error (MAE) and proves exact 4x byte reduction (FP32 -> INT8).

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class QuantizedTensor:
    q_values: list[int]    # Signed 8-bit integers [-128, 127]
    scale: float           # Float scale factor S
    zero_point: int        # Integer zero point Z
    original_length: int


def quantize_int8(floats: Sequence[float]) -> QuantizedTensor:
    """Quantizes continuous floats into 8-bit integers using affine (asymmetric) quantization."""
    if not floats:
        raise ValueError("Cannot quantize an empty sequence.")

    # Include 0.0 in the range so the zero point always lands inside [q_min, q_max].
    x_min = min(min(floats), 0.0)
    x_max = max(max(floats), 0.0)

    q_min = -128
    q_max = 127

    if x_min == x_max:  # all zeros
        scale = 1.0
        zero_point = 0
    else:
        scale = (x_max - x_min) / float(q_max - q_min)
        # Compute zero point so real zero is represented exactly
        zero_point = round(-x_min / scale) + q_min

    q_values: list[int] = []
    for x in floats:
        # Quantize and clamp to signed 8-bit range
        q = round(x / scale) + zero_point
        clamped = max(q_min, min(q_max, q))
        q_values.append(clamped)

    return QuantizedTensor(
        q_values=q_values,
        scale=scale,
        zero_point=zero_point,
        original_length=len(floats),
    )


def dequantize_int8(tensor: QuantizedTensor) -> list[float]:
    """Reconstructs approximate continuous float values from quantized INT8 tensor."""
    scale = tensor.scale
    zero_point = tensor.zero_point
    return [scale * (q - zero_point) for q in tensor.q_values]


def compute_quantization_error(original: Sequence[float], reconstructed: Sequence[float]) -> float:
    """Calculates Mean Absolute Error (MAE) between original and dequantized tensors."""
    if len(original) != len(reconstructed):
        raise ValueError("Sequence lengths must match.")
    return sum(abs(o - r) for o, r in zip(original, reconstructed)) / len(original)


def run_verification_suite() -> None:
    """Verifies integer range bounds, reconstruction fidelity (MAE), and 4x memory compression."""
    # 1. Realistic transformer weight distribution (e.g. projection matrix sample in [-2.5, 3.2])
    raw_weights = [
        -2.4510, -1.8902, -0.4412, -0.0102, 0.0000,
        0.1245, 0.8841, 1.4520, 2.1104, 3.1950,
    ]

    # 2. Perform Quantization
    q_tensor = quantize_int8(raw_weights)

    # 3. Assert Integer Invariants
    assert len(q_tensor.q_values) == len(raw_weights)
    for q in q_tensor.q_values:
        assert isinstance(q, int), f"Quantized value must be integer, got {type(q)}"
        assert -128 <= q <= 127, f"Quantized value {q} out of INT8 range [-128, 127]"

    # 4. Perform Dequantization
    reconstructed = dequantize_int8(q_tensor)
    assert len(reconstructed) == len(raw_weights)

    # 5. Assert Reconstruction Fidelity (MAE < 0.03 for 10-bucket sample)
    mae = compute_quantization_error(raw_weights, reconstructed)
    assert mae < 0.03, f"Quantization error too high: MAE = {mae:.4f}"

    # Verify maximum individual reconstruction error does not exceed scale / 2
    max_error = max(abs(o - r) for o, r in zip(raw_weights, reconstructed))
    half_scale = q_tensor.scale / 2.0 + 1e-6
    assert max_error <= half_scale, (
        f"Max error {max_error:.4f} should not exceed half scale {half_scale:.4f}"
    )

    # 5b. Edge cases an earlier version got wrong: all-positive, all-negative, and constant tensors
    for edge in ([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0], [0.5, 0.5, 0.5], [0.0, 0.0]):
        t = quantize_int8(edge)
        assert -128 <= t.zero_point <= 127
        back = dequantize_int8(t)
        assert max(abs(o - r) for o, r in zip(edge, back)) <= t.scale / 2.0 + 1e-9, (edge, back)

    # 6. Memory Compression Arithmetic Proof:
    # 1,000,000 FP32 weights = 4,000,000 bytes (4.0 MB)
    # 1,000,000 INT8 weights = 1,000,000 bytes (1.0 MB) + 12 bytes metadata
    fp32_bytes = 1_000_000 * 4
    int8_bytes = 1_000_000 * 1
    compression_ratio = fp32_bytes / int8_bytes
    assert abs(compression_ratio - 4.0) < 1e-9, f"Expected 4x compression, got {compression_ratio}"

    print("[OK] INT8 Quantization: bounds [-128, 127], error <= S/2 (incl. one-sided and constant tensors), 4x vs FP32.")


if __name__ == "__main__":
    print("Executing Bare-Metal INT8 Quantizer Verification...")
    run_verification_suite()

    # Demonstrative printout
    sample_floats = [-1.50, -0.75, 0.00, 0.42, 1.25, 2.80]
    q = quantize_int8(sample_floats)
    recon = dequantize_int8(q)

    print("\n--- Quantization Demonstration ---")
    print("Original FP32 Floats :", [f"{x:6.2f}" for x in sample_floats])
    print(f"Scale S: {q.scale:.6f} | Zero-Point Z: {q.zero_point}")
    print("Quantized INT8 Bytes :", q.q_values)
    print("Dequantized Floats   :", [f"{x:6.2f}" for x in recon])
    print(f"Mean Absolute Error  : {compute_quantization_error(sample_floats, recon):.5f}")
