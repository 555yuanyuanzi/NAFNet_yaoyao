# 分层 WaveDFPB 消融设计

## 目标

新增一条不影响现有消融的 WaveDFPB 分层路线：

```text
enc1 / enc2: Haar 高频能量引导空间注意力
enc3: Haar 高低频子带通道重标定
enc4: 完整 WaveDFPB
```

这条路线对应配置：

```text
options/train/GoPro/NAFNet-width64-layerwise-wavedfpb.yml
```

它与已有配置相互独立：

- `NAFNet-width64-wavedfpb.yml` 保持完整 WaveDFPB 路线，只通过 `stage_kwargs` 控制浅层是否启用可变形融合。
- `NAFNet-width64-layerwise-dfpb.yml` 保持原始 DFPB 的分层路线，仍使用 learnable low-pass 分解。
- 新配置只启用 `use_layerwise_wavedfpb`，不复用 `use_wavedfpb` 或 `use_layerwise_dfpb`。

## 分层策略

| stage | tier | 频率划分 | 处理方式 | 可变形融合 |
|---|---:|---|---|---|
| enc1 | 1 | Haar 小波 | 高频能量生成空间注意力 | 否 |
| enc2 | 1 | Haar 小波 | 高频能量生成空间注意力 | 否 |
| enc3 | 2 | Haar 小波 | 高/低频子带通道重标定 | 否 |
| enc4 | 3 | Haar 小波 | 完整 WaveDFPB | 是 |

## 配置接口

```yaml
network_g:
  use_wavedfpb: false
  use_layerwise_dfpb: false
  use_layerwise_wavedfpb: true
  layerwise_wavedfpb_stages: [enc1, enc2, enc3, enc4]
  layerwise_wavedfpb_kwargs:
    spatial_hidden: 16
    reduction: 4
    low_blocks: 1
    branch_expand_ratio: 2
    use_deformable_fusion: true
    tier_map:
      enc1: 1
      enc2: 1
      enc3: 2
      enc4: 3
```

## 推荐消融顺序

| 步骤 | 配置修改 | 目的 |
|---|---|---|
| 1 | `layerwise_wavedfpb_stages: [enc4]` | 验证完整 WaveDFPB 深层收益 |
| 2 | `layerwise_wavedfpb_stages: [enc3, enc4]` | 验证中层子带重标定是否带来增益 |
| 3 | `layerwise_wavedfpb_stages: [enc1, enc2, enc3, enc4]` | 验证浅层 Haar 高频空间注意力是否有额外收益 |

## 与普通 WaveDFPB 的区别

普通 `WaveletDualFrequencyProgressiveBlock` 在每个启用 stage 都执行：

```text
Haar 分解 -> 低频修复 -> 高频整流 -> 融合
```

分层 WaveDFPB 只在 `tier=3` 使用完整流程。浅层和中层只把 Haar 分解作为退化感知或子带加权信号，避免过早对高分辨率纹理做强分支修复。
