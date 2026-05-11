# GoPro 训练配置说明

本文档汇总 `options/train/GoPro` 目录下所有训练配置文件的用途、相对 baseline 新增的模块，以及关键训练差异。

通用说明：

- 这些都是训练配置，`pretrain_network_g: ~` 表示默认从零训练，不是推理配置。
- 大多数配置使用 `ImageRestorationModel`、`NAFNetLocal`、`PSNRLoss`、`AdamW` 和 `TrueCosineAnnealingLR`。
- width64 系列的主干通常是 `width: 64`、`enc_blk_nums: [1, 1, 1, 28]`、`middle_blk_num: 1`、`dec_blk_nums: [1, 1, 1, 1]`。
- `NAFNet-width32.yml` 使用官方 GoPro LMDB 路径；部分 width64 配置使用 `/home/young/yaoyao-workdir/polar/data`；较新的 layerwise 和组合实验使用 `/workspace/polarv1/data/gopro_v1/flat`。

## 总览

| 配置文件 | 实验名 | 作用 | 新增模块 |
| --- | --- | --- | --- |
| `NAFNet-width32.yml` | `NAFNet-GoPro-width32` | 官方小模型 baseline。 | 无 |
| `NAFNet-width64.yml` | `NAFNet-GoPro-width64` | 本地 width64 baseline。 | 无 |
| `NAFNet-width64-gdpm.yml` | `NAFNet-GoPro-width64-GDPM` | 测试输入特征处的全局方向先验调制。 | GDPM |
| `NAFNet-width64-pa.yml` | `NAFNet-GoPro-width64-PA` | 在各阶段 NAFBlock gate 中加入 patch-aware 信息。 | PA |
| `NAFNet-width64-ipp.yml` | `NAFNet-GoPro-width64-IPP` | 在浅层 encoder gate 中加入插值 patch prior。 | IPP |
| `NAFNet-width64-dfpb.yml` | `NAFNet-GoPro-width64-DFPB` | 在所有 encoder 阶段后加入完整双频渐进模块。 | DFPB |
| `NAFNet-width64-fftdfpb.yml` | `NAFNet-GoPro-width64-FFTDFPB` | 只在最深 encoder 阶段加入 FFT 频域双频模块。 | FFTDFPB |
| `NAFNet-width64-wavedfpb.yml` | `NAFNet-GoPro-width64-WaveDFPB` | 在所有 encoder 阶段后加入小波双频模块。 | WaveDFPB |
| `NAFNet-width64-layerwise-dfpb.yml` | `NAFNet-GoPro-width64-LayerwiseDFPB` | 按 encoder 深度使用不同复杂度的频率模块。 | Layerwise DFPB |
| `NAFNet-width64-layerwise-wavedfpb.yml` | `NAFNet-GoPro-width64-LayerwiseWaveDFPB` | 按 encoder 深度使用不同复杂度的小波频率模块。 | Layerwise WaveDFPB |
| `NAFNet-width64-rmsa.yml` | `NAFNet-GoPro-width64-RMSA` | 在部分 decoder skip 处做旋转运动感知对齐。 | RMSA |
| `NAFNet-width64-nafgate-dfpb.yml` | `NAFNet-GoPro-width64-NAFGate-DFPB` | 浅层 NAFBlock 频率 gate + 深层 DFPB 的组合实验。 | NAFBlockFrequencyGate + DFPB |

## Baseline 配置

### `NAFNet-width32.yml`

作用：官方 GoPro 小模型 baseline。

主干结构：

- `type: NAFNetLocal`
- `width: 32`
- `enc_blk_nums: [1, 1, 1, 28]`
- `middle_blk_num: 1`
- `dec_blk_nums: [1, 1, 1, 1]`

新增模块：无。

关键训练设置：

- 使用官方 GoPro LMDB：`./datasets/GoPro/train/sharp_crops.lmdb`、`./datasets/GoPro/train/blur_crops.lmdb`、`./datasets/GoPro/test/target.lmdb`、`./datasets/GoPro/test/input.lmdb`。
- `num_gpu: 8`，`manual_seed: 42`。
- `total_iter: 200000`，`T_max: 200000`。
- `lr: 1e-3`，`val_freq: 2e4`。

### `NAFNet-width64.yml`

作用：本地 width64 baseline。

主干结构：

- `type: NAFNetLocal`
- `width: 64`
- `enc_blk_nums: [1, 1, 1, 28]`
- `middle_blk_num: 1`
- `dec_blk_nums: [1, 1, 1, 1]`

新增模块：无。配置中显式关闭：

- `use_gdpm: false`
- `use_pa: false`
- `use_dfpb: false`

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data` 下的 disk 数据。
- `num_gpu: 1`，`manual_seed: 10`。
- `total_iter: 400000`，`T_max: 400000`。
- `lr: 1e-3`，`val_freq: 2e4`。

## Gate 与先验模块

### `NAFNet-width64-gdpm.yml`

作用：测试 GDPM 对 width64 NAFNet 的影响。

新增模块：`GlobalDirectionalPriorModulation`。

模块位置：

- `use_gdpm: true`。
- GDPM 作用在 `intro` 卷积之后、encoder 之前。
- `use_pa: false`，`use_dfpb: false`。

设计意图：

- 在浅层特征中注入全局方向/运动先验。
- 用于观察全局方向信息是否能提升去模糊效果。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `lr: 1e-3`，`val_freq: 2e4`。

### `NAFNet-width64-pa.yml`

作用：测试 patch-aware gate。

新增模块：`PatchAwareGate`。

模块位置：

- `use_pa: true`。
- `pa_patch_size: 8`。
- 启用阶段：`enc1`、`enc2`、`enc3`、`enc4`、`middle`、`dec1`、`dec2`、`dec3`、`dec4`。
- `use_gdpm: false`，`use_dfpb: false`。

设计意图：

- 在 NAFBlock 的 gate 中引入 patch 级平均上下文。
- 这是一个全阶段 PA 消融，不是只加在浅层。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `lr: 1e-3`，`val_freq: 2e4`。

### `NAFNet-width64-ipp.yml`

作用：测试浅层 Interpolated Patch Prior。

新增模块：`IPPGate` 和 `InterpolatedPatchPrior`。

模块位置：

- `use_ipp: true`。
- `ipp_patch_size: 8`。
- `ipp_in_ffn: false`，因此只影响 NAFBlock 前半段 gate，不影响 FFN gate。
- 启用阶段：`enc1`、`enc2`。
- `use_gdpm: false`，`use_pa: false`，`use_dfpb: false`。

设计意图：

- 在浅层纹理和局部模糊特征更强的位置加入 patch prior。
- 控制改动范围，避免全网络引入过多额外先验。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `lr: 1e-3`，`val_freq: 2e4`。

## 频率模块实验

### `NAFNet-width64-dfpb.yml`

作用：在所有 encoder 阶段后加入完整 DFPB。

新增模块：`DualFrequencyProgressiveBlock`。

模块位置：

- `use_dfpb: true`。
- 启用阶段：`enc1`、`enc2`、`enc3`、`enc4`。
- `use_gdpm: false`，`use_pa: false`。

关键 DFPB 设置：

- `low_blocks: 1`
- `branch_expand_ratio: 2`
- `use_deformable_fusion: true`
- 低通核大小：`enc1: 9`，`enc2: 7`，`enc3: 5`，`enc4: 5`。

设计意图：

- 对 encoder 特征做低频/高频分解。
- 低频分支恢复结构，高频分支修正细节，再通过低频引导的 deformable fusion 融合回主干。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `lr: 1e-3`，`val_freq: 2e4`。

### `NAFNet-width64-fftdfpb.yml`

作用：只在最深 encoder 阶段测试 FFT 版本 DFPB。

新增模块：`FFTDualFrequencyProgressiveBlock`。

模块位置：

- `use_fftdfpb: true`。
- 启用阶段：`enc4`。
- `use_dfpb: false`，`use_gdpm: false`，`use_pa: false`。

关键 FFTDFPB 设置：

- `fft_radius_ratio: 0.23`
- `low_blocks: 1`
- `branch_expand_ratio: 2`
- `use_deformable_fusion: true`

设计意图：

- 在最低分辨率、语义和运动结构更集中的 `enc4` 上测试 Fourier 频域分解。
- 相比全阶段加入，计算量和干扰范围更小。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `lr: 1e-3`，`val_freq: 2e4`。

### `NAFNet-width64-wavedfpb.yml`

作用：在所有 encoder 阶段后加入小波版本 DFPB。

新增模块：`WaveletDualFrequencyProgressiveBlock`。

模块位置：

- `use_wavedfpb: true`。
- 启用阶段：`enc1`、`enc2`、`enc3`、`enc4`。
- `use_dfpb: false`，`use_fftdfpb: false`，`use_gdpm: false`，`use_pa: false`。

关键 WaveDFPB 设置：

- `low_blocks: 1`
- `branch_expand_ratio: 2`
- 全局 `use_deformable_fusion: true`。
- stage override 中 `enc1`、`enc2`、`enc3` 关闭 deformable fusion，`enc4` 保持开启。

设计意图：

- 用小波分解替代普通低通/高通分解。
- 浅层保持轻量，深层 `enc4` 保留 deformable fusion。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `lr: 1e-3`，`val_freq: 2e4`。

### `NAFNet-width64-layerwise-dfpb.yml`

作用：按 encoder 深度分层使用不同复杂度的 DFPB 风格频率模块。

新增模块：通过 `use_layerwise_dfpb` 启用 `FrequencyAwareBlock`。

模块位置：

- `use_layerwise_dfpb: true`。
- 启用阶段：`enc1`、`enc2`、`enc3`、`enc4`。
- `use_dfpb: false`，`use_fftdfpb: false`，`use_wavedfpb: false`。

分层 tier：

- `enc1: 1`
- `enc2: 1`
- `enc3: 2`
- `enc4: 3`

关键设置：

- `low_kernel_size: 5`
- `spatial_hidden: 16`
- `reduction: 4`
- `low_blocks: 1`
- `branch_expand_ratio: 2`
- `use_deformable_fusion: true`
- 低通核大小：`enc1: 9`，`enc2: 7`，`enc3: 5`，`enc4: 5`。

设计意图：

- 让浅层使用轻量频率提示，深层使用更完整的频率建模。
- 降低全阶段完整 DFPB 的开销和过强干扰。

关键训练设置：

- 使用 `/workspace/polarv1/data/gopro_v1/flat`。
- 训练集为 LMDB crop，验证集为 disk 文件夹。
- `batch_size_per_gpu: 8`。
- `lr: 7e-4`，`val_freq: 1e4`。

### `NAFNet-width64-layerwise-wavedfpb.yml`

作用：按 encoder 深度分层使用小波频率模块。

新增模块：通过 `use_layerwise_wavedfpb` 启用 `WaveletFrequencyAwareBlock`。

模块位置：

- `use_layerwise_wavedfpb: true`。
- 启用阶段：`enc1`、`enc2`、`enc3`、`enc4`。
- `use_dfpb: false`，`use_fftdfpb: false`，`use_wavedfpb: false`，`use_layerwise_dfpb: false`。

分层 tier：

- `enc1: 1`
- `enc2: 1`
- `enc3: 2`
- `enc4: 3`

关键设置：

- `spatial_hidden: 16`
- `reduction: 4`
- `low_blocks: 1`
- `branch_expand_ratio: 2`
- `use_deformable_fusion: true`
- stage override 中保留 `enc4.use_deformable_fusion: true`。

设计意图：

- 对标 `NAFNet-width64-layerwise-dfpb.yml`，但把频率分解换成小波形式。
- 用于比较小波分解和普通学习低通分解哪个更适合分层频率建模。

关键训练设置：

- 使用 `/workspace/polarv1/data/gopro_v1/flat`。
- 训练集为 LMDB crop，验证集为 disk 文件夹。
- `batch_size_per_gpu: 8`。
- `lr: 7e-4`，`val_freq: 1e4`。

## Skip Fusion 实验

### `NAFNet-width64-rmsa.yml`

作用：在部分 decoder skip 处进行 Rotary Motion-Aware Skip Alignment。

新增模块：`RotaryMotionAwareSkipAlignment`。

模块位置：

- `use_rmsa: true`。
- 启用阶段：`dec1`、`dec2`。
- `use_dfpb: false`，`use_gdpm: false`，`use_pa: false`。

关键 RMSA 设置：

- `hidden_ratio: 1.0`
- `pos_bands: 4`
- `scale_limit: 0.5`

设计意图：

- 在 skip fusion 前对 encoder skip 和 decoder feature 做运动感知对齐。
- 只放在 `dec1`、`dec2`，因为这些深层 decoder 特征分辨率更低，建模运动结构更经济。

关键训练设置：

- 使用 `/home/young/yaoyao-workdir/polar/data`。
- `batch_size_per_gpu: 8`。
- `lr: 7e-4`，`val_freq: 1e4`。

## 组合实验

### `NAFNet-width64-nafgate-dfpb.yml`

作用：浅层 NAFBlock 内部频率 gate + 深层完整 DFPB 的组合实验。

新增模块：

- `NAFBlockFrequencyGate`
- `DualFrequencyProgressiveBlock`

模块位置：

- `use_nafblock_freq_gate: true`。
- NAFBlock 频率 gate 启用阶段：`enc1`、`enc2`、`enc3`。
- `use_dfpb: true`。
- DFPB 启用阶段：`enc4`。
- `use_gdpm: false`，`use_pa: false`，`use_ipp: false`，`use_fftdfpb: false`，`use_wavedfpb: false`，`use_layerwise_dfpb: false`，`use_layerwise_wavedfpb: false`。

NAFBlock 频率 gate 设置：

- 默认 `freq_gate_low_kernel_size: 5`
- `freq_gate_hidden_channels: 16`
- stage 低通核大小：`enc1: 9`，`enc2: 7`，`enc3: 5`。

DFPB 设置：

- `low_kernel_size: 5`
- `low_blocks: 1`
- `branch_expand_ratio: 2`
- `use_deformable_fusion: true`

设计意图：

- 浅层用轻量频率 gate 处理纹理、边缘和局部高频线索。
- 深层只在 `enc4` 使用完整 DFPB 做结构级频率建模。
- 这是组合实验，不是 baseline。

关键训练设置：

- 使用 `/workspace/polarv1/data/gopro_v1/flat`。
- 训练集为 LMDB crop，验证集为 disk 文件夹。
- `batch_size_per_gpu: 8`。
- `lr: 7e-4`，`val_freq: 1e4`。

## 快速对比

| 配置组 | 数据形式 | 学习率 | 验证频率 | 主要变化 |
| --- | --- | --- | --- | --- |
| `NAFNet-width32.yml` | 官方 GoPro LMDB | `1e-3` | `2e4` | 小模型 baseline |
| `NAFNet-width64.yml`、`gdpm`、`pa`、`ipp`、`dfpb`、`fftdfpb`、`wavedfpb` | `/home/young/.../polar/data` disk | `1e-3` | `2e4` | baseline 和早期模块消融 |
| `rmsa` | `/home/young/.../polar/data` disk | `7e-4` | `1e4` | skip fusion 消融 |
| `layerwise-dfpb`、`layerwise-wavedfpb`、`nafgate-dfpb` | `/workspace/polarv1/...` LMDB train + disk val | `7e-4` | `1e4` | 较新的频率/组合模块实验 |

