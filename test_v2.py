"""SubspaceNetCC V2 完整 sanity check"""
import sys
import torch
import numpy as np
from src.system_model import SystemModelParams
from src.signal_creation import Samples
from src.data_handler import create_channel_cross_correlation_tensor
from src.models import ModelGenerator, torch_toeplitz_from_vector

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# === 测试 1: Toeplitz 重构正确性 ===
print("=" * 60)
print("Test 1: torch_toeplitz_from_vector")
print("=" * 60)
r = torch.tensor([[1.0+0j, 2.0+1j, 3.0+0.5j, 4.0-1j]])  # [BS=1, N=4]
R = torch_toeplitz_from_vector(r)
print(f"Input r:   {r[0]}")
print(f"Output R:\n{R[0]}")

# 验证 Hermitian
herm_err = (torch.norm(R - R.conj().transpose(-2, -1)) / torch.norm(R)).item()
print(f"Hermitian error: {herm_err:.2e}")
assert herm_err < 1e-6, "Must be Hermitian"

# 验证 Toeplitz（每条对角线常值）
for k in range(1, 4):
    diag = torch.diagonal(R[0], offset=k)
    assert torch.allclose(diag, diag[0], atol=1e-6), f"Off-diag {k} not constant"
print("✓ Toeplitz structure perfect")
print()

# === 测试 2: 初始化时残差为零 ===
print("=" * 60)
print("Test 2: Zero-init residual = baseline CC + Toeplitz")
print("=" * 60)
params = (SystemModelParams()
          .set_parameter("N", 16).set_parameter("M", 3).set_parameter("T", 200)
          .set_parameter("snr", 10).set_parameter("signal_type", "NarrowBand")
          .set_parameter("signal_nature", "non-coherent"))

np.random.seed(0); torch.manual_seed(0)
sm = Samples(params); sm.set_doa(None)
X_np = sm.samples_creation(0, 1, 0, 1)[0]
X = torch.tensor(X_np, dtype=torch.complex64)

cc = create_channel_cross_correlation_tensor(X, K=8, G=8, L=25, ref_channel=0)
print(f"CC tensor shape: {cc.shape}")

model_cfg = (ModelGenerator()
             .set_model_type("SubspaceNetCC").set_diff_method("root_music").set_tau(8)
             .set_cc_params(G=8, K=8, L=25, ref_channel=0).set_model(params))
model = model_cfg.model
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

batch = cc.unsqueeze(0).float()
with torch.no_grad():
    doa, doa_all, roots, Rz = model(batch)

print(f"\nRz shape: {Rz.shape}")
herm_err = (torch.norm(Rz - Rz.conj().transpose(-2, -1)) / torch.norm(Rz)).item()
print(f"Rz Hermitian error: {herm_err:.2e}")

true_deg = np.sort(sm.doa * 180 / np.pi)
pred_deg = np.sort(doa[0].cpu().numpy() * 180 / np.pi)
print(f"\nTrue DOA (deg, sorted):  {true_deg}")
print(f"Pred DOA (deg, sorted):  {pred_deg}")
err = pred_deg - true_deg
print(f"Per-angle error (deg):   {err}")
print(f"RMSE (deg):              {np.sqrt(np.mean(err**2)):.3f}")

print()
print("Expected: 初始化下 V2 应该约等于经典 CC+Toeplitz baseline (~1°-2°)")
print("         如果 RMSE < 5°，说明残差确实为零，初始化正确")
print("         如果 RMSE > 10°，说明初始化有问题")

if np.sqrt(np.mean(err**2)) < 5:
    print("\n✓ V2 architecture works correctly. Ready for training.")
else:
    print("\n✗ V2 may have a bug. Check torch_toeplitz_from_vector or fc2 zero-init.")
