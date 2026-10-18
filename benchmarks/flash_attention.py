"""Causal D128 BF16 correctness; fixed B=4, H=8 MHA runtime comparisons."""

import importlib

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch


def load_reference_apis(reference):
    aliases = {
        "flash_attention": "flash_attention_2",
        "fa2": "flash_attention_2",
        "fa3": "flash_attention_3",
        "fa4": "flash_attention_4",
    }
    reference = aliases.get(reference, reference)
    valid = ("pytorch", "flash_attention_2", "flash_attention_3", "flash_attention_4", "all")
    if reference not in valid:
        raise ValueError("FlashAttention references: pytorch, fa2, fa3, fa4, or all")

    selected = (
        ("flash_attention_2", "flash_attention_3", "flash_attention_4")
        if reference == "all" else (reference,)
    )
    apis = {}
    if "flash_attention_2" in selected:
        apis["FlashAttention-2"] = importlib.import_module("flash_attn").flash_attn_func
    if "flash_attention_3" in selected:
        module = importlib.import_module("flash_attn_interface")
        apis["FlashAttention-3"] = module.flash_attn_func
    if "flash_attention_4" in selected:
        apis["FlashAttention-4"] = importlib.import_module("flash_attn.cute").flash_attn_func
    return reference, apis


def forward_reference(tensors, scale):
    """FP32 causal oracle with bounded score storage, including at T=16384."""
    q, k, v = tensors
    group_size = q.shape[2] // k.shape[2]
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=2)
        v = v.repeat_interleave(group_size, dim=2)
    q, k, v = (x.float().transpose(1, 2) for x in (q, k, v))
    length = q.shape[2]
    keys = torch.arange(length, device=q.device)
    result = torch.empty_like(q)
    row_lse = torch.empty(q.shape[:-1], device=q.device)
    for start in range(0, length, 128):
        end = min(start + 128, length)
        scores = q[:, :, start:end] @ k.transpose(-1, -2) * scale
        mask = torch.arange(start, end, device=q.device)[:, None] >= keys[None, :]
        scores.masked_fill_(~mask, -torch.inf)
        result[:, :, start:end] = scores.softmax(-1) @ v
        row_lse[:, :, start:end] = scores.logsumexp(-1)
    return result.transpose(1, 2).bfloat16(), row_lse


def cases(args, family):
    lib = library("flash_attention")
    forward = bind(lib, "gpu_kernels_flash_forward", [P] * 5 + [I] * 5 + [F, P])
    default_reference = "pytorch" if args.test else ("flash_attention_4" if args.suite == "h100" else "flash_attention_2")
    reference, reference_apis = load_reference_apis(args.reference or default_reference)

    if args.test:
        shapes = (
            (1, 64, 2, 2, 128),
            (2, 64, 4, 2, 128),
            (1, 128, 4, 1, 128),
            (1, 256, 8, 8, 128),
        )
    elif args.suite == "quick":
        shapes = ((4, 2048, 8, 8, 128),)
    else:
        shapes = tuple((4, sequence, 8, 8, 128) for sequence in (2048, 4096, 8192, 16384))

    for b, t, query_heads, key_value_heads, d in shapes:
        query_shape = (b, t, query_heads, d)
        key_value_shape = (b, t, key_value_heads, d)
        query = (torch.randn(query_shape, device="cuda") * .5).bfloat16()
        key = (torch.randn(key_value_shape, device="cuda") * .5).bfloat16()
        value = (torch.randn(key_value_shape, device="cuda") * .5).bfloat16()
        inputs = (query, key, value)
        scale = d**-.5

        def pytorch_forward(tensors=inputs):
            return forward_reference(tensors, scale)

        expected = tuple(x.detach() for x in pytorch_forward())
        output = torch.empty(query_shape, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(b, query_heads, t, device="cuda")

        def custom_forward():
            checked(lib, "flash", forward(
                *pointers((output, lse, *inputs)),
                b, t, query_heads, key_value_heads, d, scale, stream()))
            return output, lse

        functions = {"custom": custom_forward}
        if reference == "pytorch":
            functions["PyTorch"] = pytorch_forward

        for label, api in reference_apis.items():
            if label == "FlashAttention-2":
                def official_forward(tensors=inputs, function=api):
                    return function(
                        *tensors, dropout_p=0., softmax_scale=scale,
                        causal=True, return_attn_probs=True)[:2]
            elif label == "FlashAttention-3":
                def official_forward(tensors=inputs, function=api):
                    return function(
                        *tensors, softmax_scale=scale, causal=True,
                        return_attn_probs=True)[:2]
            else:
                def official_forward(tensors=inputs, function=api):
                    return function(
                        *tensors, softmax_scale=scale, causal=True,
                        return_lse=True)[:2]
            functions[label] = official_forward

        size = f"B={b},T={t},Hq={query_heads},Hkv={key_value_heads},D={d}"
        yield Operation(size, "bf16", "forward", functions, expected,
                        (1e-2, 1e-4), (1e-2, 1e-5))
