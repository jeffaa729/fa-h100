#!/usr/bin/env python3

"""Profile official FlashAttention on tensors shared with the CUDA benchmark.
The optional raw dump contains the BF16 output converted to FP32."""

import argparse
import math
from pathlib import Path
import torch
from flash_attn import flash_attn_func


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=int)
    parser.add_argument("sequence", type=int)
    parser.add_argument("heads", type=int)
    parser.add_argument("head_size", type=int)
    parser.add_argument("operation", choices=("forward",))
    parser.add_argument("dump", nargs="?", type=Path)
    return parser.parse_args()


def checked_profiler_call(call, name):
    result = call()
    status = result[0] if isinstance(result, tuple) else result
    if status != 0:
        raise RuntimeError(f"{name} failed with CUDA status {status}")


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("official FlashAttention reference requires CUDA")

    elements = args.batch * args.sequence * args.heads * args.head_size
    index = torch.arange(elements, dtype=torch.int64, device="cuda")

    def bf16_values(multiplier, modulus, offset, divisor):
        return (((index * multiplier) % modulus).float() - offset).div_(
            divisor
        ).to(torch.bfloat16).reshape(
            args.batch, args.sequence, args.heads, args.head_size
        )

    query = bf16_values(17, 101, 50, 64)
    key = bf16_values(23, 97, 48, 61)
    value = bf16_values(31, 89, 44, 59)
    scale = 1.0 / math.sqrt(args.head_size)

    def forward():
        return flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
        )

    cudart = torch.cuda.cudart()
    checked_profiler_call(cudart.cudaProfilerStart, "cudaProfilerStart")
    with torch.no_grad():
        forward()
    torch.cuda.synchronize()
    checked_profiler_call(cudart.cudaProfilerStop, "cudaProfilerStop")

    if args.dump is not None:
        output = forward()
        torch.cuda.synchronize()
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("wb") as destination:
            destination.write(output.float().cpu().contiguous().numpy().tobytes())



if __name__ == "__main__":
    main()
