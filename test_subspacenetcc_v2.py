"""验证重构后的 SubspaceNetCC：未训练即应≈经典 CC+Toeplitz 性能。"""
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from itertools import permutations

from src.system_model import SystemModelParams
from src.signal_creation import Samples
from src.data_handler import create_channel_cross_correlation_tensor
from src.models import SubspaceNetCC, torch_toeplitz_from_vector
from src.methods import (
    CrossCorrToeplitzRootMUSIC,
    segmented_cross_correlation,
    toeplitz_reconstruction,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Toeplitz 函数数值正确性
print("=== 1. Toeplitz 函数验证 ===")
r = torch.tensor([[1.0+0j, 2+1j, 3-1j, 4+2j]], dtype=torch.complex64)
T = torch_toeplitz_from_vector(r)
print(f"T[0]:\n{T[0]}")
herm_err = torch.norm(T - T.conj().transpose(-2, -1)).item()
print(f"Hermitian 误差: {herm_err:.2e}")
assert herm_err < 1e-5, "应严格 Hermitian"
# 检查 Toeplitz 结构：每条对角线常数
for k in range(-3, 4):
    diag = torch.diagonal(T[0], offset=k)
    if k != 0:
        var = (diag - diag[0]).abs().max().item()
        assert var < 1e-5, f"diag {k} 不是常数"
print("✓ Toeplitz + Hermitian 结构正确\n")

# 2. 未训练模型应≈经典 CC+Toeplitz（残差初始化保证）
print("=== 2. 未训练 SubspaceNetCC vs 经典 CC+Toeplitz ===")
np.random.seed(0); torch.manual_seed(0)

params = (SystemModelParams()
          .set_parameter("N", 16).set_parameter("M", 3).set_parameter("T", 200)
          .set_parameter("snr", 10).set_parameter("signal_type", "NarrowBand")
          .set_parameter("signal_nature", "non-coherent")
          .set_parameter("eta", 0).set_parameter("bias", 0).set_parameter("sv_noise_var", 0))

model = SubspaceNetCC(G=8, K=8, N=16, M=3, diff_method="root_music")
model.eval()
assert model.fc2.out_features == 2 * params.N, "残差头必须只输出 2N 维"
assert torch.count_nonzero(model.fc2.weight).item() == 0, "fc2 权重必须零初始化"
assert torch.count_nonzero(model.fc2.bias).item() == 0, "fc2 bias 必须零初始化"

def rmspe_deg(preds, truth):
    truth = np.sort(truth); preds = np.array(preds)
    best = float('inf')
    for p in permutations(preds):
        err = np.array(p) - truth
        err = ((err + 90) % 180) - 90
        v = np.sqrt(np.mean(err**2))
        if v < best: best = v
    return best

nn_errs, classical_errs = [], []
for trial in range(50):
    sm = Samples(params); sm.set_doa(None)
    X_np, _, _, _ = sm.samples_creation(0, 1, 0, 1)
    truth = sm.doa * 180 / np.pi

    # 未训练网络
    X_t = torch.tensor(X_np, dtype=torch.complex64)
    cc = create_channel_cross_correlation_tensor(X_t, K=8, G=8, L=25, ref_channel=0)
    cc_batch = cc.unsqueeze(0).float()
    with torch.no_grad():
        doa, _, _, Rz = model(cc_batch)

    if trial == 0:
        r_model = model._baseline_cc(cc_batch)[0].numpy()
        r_classical = segmented_cross_correlation(X_np, L=25, G=8, ref_channel=0)
        r_err = np.max(np.abs(r_model - r_classical))
        assert r_err < 1e-5, f"baseline_r 与经典 CC 不一致: {r_err:.2e}"

        R_expected = toeplitz_reconstruction(r_classical)
        R_expected += 1e-3 * np.eye(R_expected.shape[0])
        R_err = np.max(np.abs(Rz[0].numpy() - R_expected))
        assert R_err < 1e-4, f"Rz 与 Toeplitz baseline 不一致: {R_err:.2e}"

    nn_pred = np.sort(doa[0].numpy() * 180 / np.pi)
    if len(nn_pred) >= 3:
        nn_errs.append(rmspe_deg(nn_pred[:3], truth))

    # 经典基线
    cl = CrossCorrToeplitzRootMUSIC(sm, L=25, G=8, ref_channel=0)
    cl_pred, _, _, _, _ = cl.estimate(X_np)
    if len(cl_pred) >= 3:
        classical_errs.append(rmspe_deg(cl_pred[:3], truth))

nn_med = np.median(nn_errs)
cl_med = np.median(classical_errs)
print(f"未训练 SubspaceNetCC v2:  median {nn_med:.3f}°")
print(f"经典 CC+Toeplitz:        median {cl_med:.3f}°")
print(f"差距:                    {abs(nn_med - cl_med):.4f}°")
assert abs(nn_med - cl_med) < 1.0, \
    f"未训练网络和经典基线差距过大 ({abs(nn_med-cl_med):.3f}°)，残差初始化可能没生效"
print("✓ 残差初始化生效，起步即达到经典基线水准\n")

# 3. 参数量对比
print("=== 3. 参数量统计 ===")
n_params = sum(p.numel() for p in model.parameters())
old_v1_params = 173345
old_free_output_dim = 2 * params.N * params.N
new_residual_dim = 2 * params.N
print(f"新 SubspaceNetCC v2: {n_params:,} 参数")
print(f"  (旧版 K=8 N=16 时 173,345 参数，76% 在 projection FC 黑盒里)")
print(f"  总参数压缩比: {old_v1_params / n_params:.1f}×")
print(f"  输出自由度压缩比: {old_free_output_dim / new_residual_dim:.1f}×")
assert n_params < old_v1_params / 4, "V2 head 应显著小于旧版自由矩阵模型"
assert old_free_output_dim / new_residual_dim == 16, "输出维度应为 16× 压缩"

print("\n=== 所有检查通过 ✓ ===")
