"""
Install:
    pip install pytest
Uasge:
    python -m pytest diffusers/tests/torch_to_oflow/test_temp_fix_compile_impl.py
"""
import numpy as np
import pytest
from onediff.infer_compiler.backends.oneflow.import_tools.patch_for_compiler import (
    FakeCuda,
)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len_q", [4, 8, 64, 128, 256, 512])
@pytest.mark.parametrize("seq_len_k", [4, 8, 64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [8, 16, 32, 64])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.0])
# @pytest.mark.parametrize("dropout_p", [0.0, 0.22, 0.48])
@pytest.mark.parametrize("seed", [1])
def test_flash_attention(
    batch_size, n_heads, seq_len_q, seq_len_k, head_dim, is_causal, dropout_p, seed
):
    np.random.seed(seed=seed)
    query = np.random.rand(batch_size, n_heads, seq_len_q, head_dim).astype(np.float16)
    key = np.random.rand(batch_size, n_heads, seq_len_k, head_dim).astype(np.float16)
    value = np.random.rand(batch_size, n_heads, seq_len_k, head_dim).astype(np.float16)

    def torch_flash_attention() -> np.ndarray:
        import torch

        q = torch.from_numpy(query).to("cuda")
        k = torch.from_numpy(key).to("cuda")
        v = torch.from_numpy(value).to("cuda")
        result = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return result.cpu().detach().numpy()

    def oneflow_flash_attention() -> np.ndarray:
        import oneflow as flow  # usort: skip

        q = flow.tensor(query, dtype=flow.float16).to("cuda")
        k = flow.tensor(key, dtype=flow.float16).to("cuda")
        v = flow.tensor(value, dtype=flow.float16).to("cuda")

        return FakeCuda.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        ).numpy()

    torch_result = torch_flash_attention()

    result = oneflow_flash_attention()  # BHMK

    assert np.allclose(torch_result.flatten(), result.flatten(), atol=1e-2, rtol=1e-2)
