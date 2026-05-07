# SubspaceNetCC

SubspaceNetCC 是一个用于 DOA（Direction of Arrival）估计的研究代码仓库。当前项目定位是：

- 复现 SubspaceNet 相关流程；
- 在复现代码基础上实验“通道互相关输入”（channel cross-correlation input）改造；
- 对比传统子空间方法、SubspaceNet 系列模型，以及新增的 CC-Toeplitz Root-MUSIC 流程。

> [!IMPORTANT]
> 本 README 只描述当前代码中已经存在的功能、配置和限制，不包含未验证实验结果、性能指标或论文结论。

## 版本文档

- 当前文档化版本：`v0.1.0`。
- 版本历史见 [CHANGELOG.md](CHANGELOG.md)。
- `v0.1.0` 复现状态与限制见 [docs/releases/v0.1.0.md](docs/releases/v0.1.0.md)。

## 当前实现范围

当前代码中可以确认的功能包括：

- 合成阵列观测数据生成：`SystemModelParams`、`SystemModel`、`Samples` 支持窄带/宽带、相干/非相干信号、SNR、阵元数量、快拍数、阵列偏差和 steering vector 噪声等参数。
- SubspaceNet 复现相关模型：`SubspaceNet`、`SubspaceNetEsprit`、`DeepRootMUSIC`、`DeepAugmentedMUSIC`、`DeepCNN`。
- 通道互相关输入模型：`SubspaceNetCC` 接收 `[batch_size, G, 2K, N]` 实值张量，将输出整理为复数 surrogate covariance `Rz`，再接可微 `root_music` 或 `esprit`。
- 通道互相关数据预处理：`create_channel_cross_correlation_tensor` 将 `[N, T]` 观测按 `G` 个非重叠片段切分，每段生成 `[2K, N]` 的实部/虚部互相关特征。
- 传统子空间方法：`MUSIC`、`RootMUSIC`、`Esprit`、`MVDR`。
- CC-Toeplitz Root-MUSIC baseline：先用参考通道零延迟互相关估计一列，再构造 Hermitian Toeplitz 矩阵并运行 Root-MUSIC。
- 训练与评估流程：`TrainingParams`、`train`、`evaluate`、`evaluate_dnn_model`、`evaluate_model_based`。
- 两个 CC 相关检查脚本：`test_cc_input.py` 和 `test_cc_toeplitz.py`。

## 代码结构

```text
.
├── main.py                 # 主实验入口：配置数据、模型、训练、评估
├── test_cc_input.py        # SubspaceNetCC 输入形状与 forward sanity check
├── test_cc_toeplitz.py     # CC + Toeplitz + Root-MUSIC pipeline 检查
├── src/
│   ├── system_model.py     # 阵列与系统参数
│   ├── signal_creation.py  # 合成信号、噪声和观测
│   ├── data_handler.py     # 数据集生成、加载、协方差/互相关特征
│   ├── models.py           # SubspaceNet、SubspaceNetCC 和其他模型
│   ├── methods.py          # MUSIC、Root-MUSIC、ESPRIT、MVDR、CC-Toeplitz
│   ├── training.py         # 训练参数、训练循环、模型保存
│   ├── evaluation.py       # DNN、增强方法和传统方法评估
│   ├── criterions.py       # RMSPE/MSPE 损失
│   ├── plotting.py         # 谱图与 root 图绘制
│   └── utils.py            # 随机种子、Root-MUSIC 工具、矩阵工具
├── pyEnv/
│   ├── SubspaceNetEnv.yaml # Conda 环境定义，包含 torch
│   └── requirements.txt    # pip 依赖列表；当前 torch 被注释
└── data/                   # 本地数据、权重和输出目录；通常不应提交
```

## 环境准备

推荐使用仓库提供的 Conda 环境：

```bash
conda env create -f pyEnv/SubspaceNetEnv.yaml
conda activate SubspaceNetEnv
```

也可以使用 pip 安装依赖：

```bash
pip install -r pyEnv/requirements.txt
```

但需要注意，`pyEnv/requirements.txt` 中的 `torch==2.0.1` 目前是注释状态，而代码会直接导入 `torch`。如果使用 pip 路线，需要按本机 CUDA/CPU 环境另行安装 PyTorch。

## 最小运行命令

从仓库根目录运行主入口：

```bash
python main.py
```

`main.py` 当前默认配置为：

- `scenario_data_path = "uniform_bias_spacing"`；
- `commands["LOAD_DATA"] = True`；
- `commands["TRAIN_MODEL"] = False`；
- `commands["EVALUATE_MODE"] = True`；
- 系统参数：`N=16`、`M=3`、`T=200`、`snr=10`、`signal_type="NarrowBand"`、`signal_nature="non-coherent"`；
- 模型：`SubspaceNetCC` + `root_music`；
- CC 参数：`G=8`、`K=8`、`L=25`、`ref_channel=0`；
- 数据规模：`samples_size=100000`，测试集比例 `train_test_ratio=0.1`。

因此，默认 `python main.py` 会尝试从 `data/datasets/uniform_bias_spacing` 加载数据，并从 `data/weights/final_models` 加载对应模型权重。若这些本地文件不存在，需要先在 `main.py` 中调整 `commands` 以生成数据、训练模型，或改成已有数据/权重匹配的配置。

更小的 CC 相关检查命令：

```bash
python test_cc_input.py
python test_cc_toeplitz.py
```

这两个脚本分别检查 SubspaceNetCC 输入/输出形状，以及 CC-Toeplitz Root-MUSIC 的矩阵结构和固定随机种子下的 pipeline 行为。

## 当前验证状态

本次文档更新期间尝试运行：

```bash
python test_cc_input.py
python test_cc_toeplitz.py
```

当前本地 Python 环境缺少运行依赖：`test_cc_input.py` 在导入 `torch` 时失败，`test_cc_toeplitz.py` 在导入 `numpy` 时失败。因此，本 README 不记录任何本次运行得到的数值结果。

## 主要配置入口

实验配置集中在 `main.py`：

- `commands` 控制是否创建数据、加载数据、训练、保存模型和评估；
- `SystemModelParams()` 设置阵列、信号和噪声参数；
- `ModelGenerator().set_model_type(...)` 选择 `SubspaceNetCC`、`SubspaceNet`、`DA-MUSIC` 或 `DeepCNN` 等模型；
- `set_diff_method("root_music" | "esprit")` 选择 SubspaceNet 系列模型内部使用的可微子空间方法；
- `set_cc_params(G, K, L, ref_channel)` 设置 SubspaceNetCC 的通道互相关输入参数；
- `samples_size` 和 `train_test_ratio` 控制数据集规模。

## 当前限制

- 当前 README 不给出任何实验结果，因为仓库中没有可直接归档的、经复核的实验指标说明。
- `data/`、权重、输出和大规模数据在 `.gitignore` 中被忽略；全新克隆仓库时通常不会自动具备 `main.py` 默认加载所需的数据和权重。
- `main.py` 的默认样本规模是 `100000`，完整数据生成和训练成本较高；当前没有命令行参数或小样本 profile。
- `SubspaceNetCC` 的 `set_tau(8)` 调用不会设置 `tau`，因为 `ModelGenerator.set_tau` 对 `SubspaceNetCC` 直接返回；当前 CC 分支主要依赖 `G/K/L/ref_channel`。
- `CC-Toeplitz` baseline 的参数解析要求 `ref_channel=0`，因为实现假设用第一个 ULA 阵元估计 Toeplitz 第一列。
- `test_cc_input.py` 和 `test_cc_toeplitz.py` 依赖 `torch`、`numpy` 等环境；未安装依赖时会先在导入阶段失败。
- 部分中文注释在当前文件中存在编码显示问题，不影响这里对代码行为的描述，但会影响阅读体验。

## 下一步建议

- 给 `main.py` 增加 CLI 参数或独立配置文件，避免每次通过改源码切换实验。
- 增加小样本 smoke 配置，例如几十个样本的数据生成和一次前向/评估，便于快速验证环境。
- 补齐 SubspaceNetCC 数据生成、训练、评估的端到端说明，并明确每个权重文件对应的参数。
- 修复现有中文注释和测试脚本中的编码显示问题。
- 为 `create_channel_cross_correlation_tensor`、`SubspaceNetCC.forward`、`CrossCorrToeplitzRootMUSIC` 增加更稳定的自动化测试。
- 在获得可复核实验结果后，用单独文档记录实验配置、随机种子、数据版本、权重路径和指标，避免 README 混入未经验证的结论。
