"""
Quick test to confirm which SDPA backend PyTorch selects for:
  - SWA path  (float attn_mask with -inf padding)
  - Baseline  (is_causal=True)

Run on MetaCentrum after activating the venv:
  python test_sdp_backend.py
"""
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print(f"device: {device}, dtype: {dtype}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

B, nh, T, hd = 2, 12, 1024, 64
q = torch.randn(B, nh, T, hd, device=device, dtype=dtype)
k = torch.randn(B, nh, T, hd, device=device, dtype=dtype)
v = torch.randn(B, nh, T, hd, device=device, dtype=dtype)

window_size = 128
mask = torch.ones(T, T, dtype=torch.bool, device=device)
mask = torch.tril(mask)
mask = torch.triu(mask, diagonal=-(window_size - 1))
attn_bias = torch.zeros(T, T, device=device, dtype=dtype)
attn_bias.masked_fill_(~mask, float('-inf'))

print("\n=== SWA (float attn_mask) ===")
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    try:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        print("FlashAttention: YES (unexpected!)")
    except RuntimeError as e:
        print(f"FlashAttention: NO  =>  {e}")

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    try:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        print("Math backend:   YES")
    except RuntimeError as e:
        print(f"Math backend:   NO  =>  {e}")

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
    try:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        print("MemEfficient:   YES")
    except RuntimeError as e:
        print(f"MemEfficient:   NO  =>  {e}")

print("\n=== Baseline (is_causal=True) ===")
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    try:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("FlashAttention: YES")
    except RuntimeError as e:
        print(f"FlashAttention: NO  =>  {e}")
