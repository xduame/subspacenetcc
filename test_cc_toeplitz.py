"""验证 CC + Toeplitz + Root-MUSIC pipeline 数值正确性。"""
import sys

import numpy as np
from src.system_model import SystemModelParams
from src.signal_creation import Samples
from src.methods import (
    compute_zero_lag_cross_correlation,
    segmented_cross_correlation,
    toeplitz_reconstruction,
    CrossCorrToeplitzRootMUSIC,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. 配置（与 main.py 一致）
params = (
    SystemModelParams()
    .set_parameter("N", 16)
    .set_parameter("M", 3)
    .set_parameter("T", 200)
    .set_parameter("snr", 10)
    .set_parameter("signal_type", "NarrowBand")
    .set_parameter("signal_nature", "non-coherent")
    .set_parameter("eta", 0)
    .set_parameter("bias", 0.05)
    .set_parameter("sv_noise_var", 0)
)

L, G = 25, 8

# 2. 生成观测
np.random.seed(0)
sm = Samples(params)
sm.set_doa(None)
X, _, _, _ = sm.samples_creation(0, 1, 0, 1)
true_doa_deg = np.sort(sm.doa * 180 / np.pi)
print(f"True DOA (deg): {true_doa_deg}")

# 3. 测试零延迟互相关
r0 = compute_zero_lag_cross_correlation(X, ref_channel=0)
print(f"r0 shape: {r0.shape}, r0[0] (应近似实数功率): {r0[0]:.4f}")
assert r0.shape == (params.N,)
assert abs(r0[0].imag) < 1e-6, "r0[0] 应为实数"

# 4. 测试段平均
r_hat = segmented_cross_correlation(X, L=L, G=G, ref_channel=0)
print(f"r_hat shape: {r_hat.shape}")
assert r_hat.shape == (params.N,)

# 5. 测试 Toeplitz 重构
R_T = toeplitz_reconstruction(r_hat)
print(f"R_T shape: {R_T.shape}")
assert R_T.shape == (params.N, params.N)

# 验证 Hermitian
herm_err = np.linalg.norm(R_T - R_T.conj().T) / np.linalg.norm(R_T)
print(f"Hermitian 误差: {herm_err:.2e}")
assert herm_err < 1e-10, "Toeplitz 重构应严格 Hermitian"

# 验证 Toeplitz（每条对角线值相同）
for k in range(1, params.N):
    diag = np.diag(R_T, k=k)
    assert np.allclose(diag, diag[0]), f"第 {k} 条上对角线不是常数"
print("Toeplitz 结构验证通过 ✓")

# 6. 测试完整 pipeline
cc_rmusic = CrossCorrToeplitzRootMUSIC(sm, L=L, G=G, ref_channel=0)
predictions, roots, _, _, M = cc_rmusic.estimate(X)
print(f"Predicted DOA (deg, sorted): {np.sort(predictions)}")
print(f"|preds| - 1 (应 < 1，根在单位圆内): {np.abs(roots[:M]) - 1}")

# 7. 与 ground truth 比较
err = np.sort(predictions) - true_doa_deg
print(f"逐角度误差 (deg): {err}")
rmspe_deg = np.sqrt(np.mean(err**2))
print(f"RMSPE (deg): {rmspe_deg:.3f}")
assert rmspe_deg < 5, "高 SNR/固定随机种子下 RMSPE 应在几度以内"

print("\n所有数值检查通过 ✓")
