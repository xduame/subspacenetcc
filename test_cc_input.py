"""SubspaceNetCC 形状与前向传播 sanity check。"""
import sys

import torch
from src.system_model import SystemModelParams
from src.signal_creation import Samples
from src.data_handler import create_channel_cross_correlation_tensor
from src.models import ModelGenerator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. 配置
params = (SystemModelParams()
          .set_parameter("N", 8).set_parameter("M", 3).set_parameter("T", 200)
          .set_parameter("snr", 10).set_parameter("signal_type", "NarrowBand")
          .set_parameter("signal_nature", "non-coherent")
          .set_parameter("eta", 0).set_parameter("bias", 0).set_parameter("sv_noise_var", 0))

G, K, L = 8, 8, 25

# 2. 生成一段观测
sm = Samples(params)
sm.set_doa(None)
X_np = sm.samples_creation(0, 1, 0, 1)[0]
X = torch.tensor(X_np, dtype=torch.complex64)

# 3. 测试 create_channel_cross_correlation_tensor
cc = create_channel_cross_correlation_tensor(X, K=K, G=G, L=L, ref_channel=0)
print("[CC tensor] shape:", cc.shape, "dtype:", cc.dtype, "has NaN:", torch.isnan(cc).any().item())
assert cc.shape == (G, 2*K, params.N), f"期望 ({G},{2*K},{params.N})，得到 {cc.shape}"

# 4. 测试 SubspaceNetCC forward
model_cfg = (ModelGenerator()
             .set_model_type("SubspaceNetCC")
             .set_diff_method("root_music")
             .set_tau(8)
             .set_cc_params(G=G, K=K, L=L)
             .set_model(params))
model = model_cfg.model

# 5. 加 batch 维度
batch = cc.unsqueeze(0)  # [1, G, 2K, N]
print("[Batch input] shape:", batch.shape)

with torch.no_grad():
    doa, doa_all, roots, Rz = model(batch.float())

print("[Output] doa shape:", doa.shape, "  Rz shape:", Rz.shape)
assert Rz.shape == (1, params.N, params.N), f"Rz 形状错误: {Rz.shape}"
assert doa.shape == (1, params.M), f"doa 形状错误: {doa.shape}"

# 6. 检查 Rz 是否近似 Hermitian
Rz_np = Rz.detach().cpu().numpy()[0]
import numpy as np
herm_err = np.linalg.norm(Rz_np - Rz_np.conj().T) / np.linalg.norm(Rz_np)
print(f"[Rz Hermitian 误差] {herm_err:.4e}（< 1 即代表 gram_diagonal_overload 工作正常）")

# 7. 高 SNR 下的粗糙正确性
print(f"[True DOA (deg)]    {sm.doa * 180 / np.pi}")
print(f"[Predicted DOA (deg)] {doa[0].cpu().numpy() * 180 / np.pi}")

print("\n所有形状检查通过 ✓")
