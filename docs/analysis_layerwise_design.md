# 分层频率感知设计：从浅到深的渐进策略

## 设计哲学

核心问题：**浅层和深层的频率分解应该做不同的事。**

现有的 `DualFrequencyProgressiveBlock` 包含四个阶段：
1. 低通分解（AdaptiveLowPassExtractor）
2. 低频独立修复（LowFrequencyRestorer）
3. 高频独立整流（HighFrequencyRectifier）
4. 可变形卷积融合（LowGuidedDeformableFusion）

在 enc4 上这四步都有意义。但浅层的问题出在 **第 2、3、4 步**，而不是第 1 步。浅层的分解本身没问题，问题在于分解后的独立处理不合理。

因此我们的策略是：**所有层共享第 1 步（分解），但对分解后的处理做层级化递进。**

```
             enc1          enc2          enc3          enc4
分解         ✅ 低通分解    ✅ 低通分解    ✅ 低通分解    ✅ 低通分解
处理策略     空间注意力     子带加权       轻量分支处理   完整分支+DCN
复杂度       极轻           轻             中             重
新增参数     ~0.3C²         ~2C²           ~6C²           ~20C²（现有）
```

---

## 三级架构设计

### Tier 1：频率引导空间注意力（enc1, enc2）

**核心思想**：不修改高/低频内容，只用频率信息告诉网络 **空间上哪里需要更多关注**。

```
输入 x → LowPass → x_low
              ↓
       x_high = x - x_low
              ↓
       spatial_energy = AvgPool_channel(|x_high|²)  ← 退化程度的空间地图
              ↓
       attention = σ(Conv(spatial_energy))
              ↓
输出 = x + γ · (attention ⊙ x)
```

**为什么这在浅层合理？**

- 在浅层，高频能量强的空间位置通常对应：模糊边界、噪声区域、纹理密集区
- 这就是一个 **退化感知的空间注意力**：模糊重的地方让网络投入更多注意力
- 不做任何频率成分的修改，只做注意力引导，完全不会误杀有效纹理
- 与 NAFBlock 的通道注意力（SCA）形成互补：SCA 做通道选择，Tier 1 做空间选择

```python
class FreqGuidedSpatialAttention(nn.Module):
    """
    Tier 1: 频率引导空间注意力。
    
    利用高频能量作为退化程度的空间指示器，
    生成空间注意力图来引导网络关注退化严重的区域。
    不修改频率成分本身，避免浅层误杀有效纹理。
    """

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.low_pass = AdaptiveLowPassExtractor(channels, kernel_size=kernel_size)
        
        # 从高频能量图生成空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = self.low_pass(x)
        x_high = x - x_low
        
        # 高频能量图 [B, 1, H, W]
        energy = (x_high ** 2).mean(dim=1, keepdim=True)
        attn = self.spatial_attn(energy)  # [B, 1, H, W]
        
        return x + self.gamma * (attn * x)
```

> [!NOTE]
> 这个模块只增加了 ~(C×k² + 16×9×2 + 1) ≈ C×25 + 290 个参数，极其轻量。
> 而完整的 DFPB 在同样的通道数下有 ~20C² 个参数。

---

### Tier 2：频率子带重标定（enc3）

**核心思想**：分解后不做独立修复，但让网络学会 **对高低频分别做通道级加权**，然后直接重组。

```
输入 x → LowPass → x_low
                     ↓
              x_high = x - x_low
                     ↓
         ┌──────────┴──────────┐
    w_low = SE(x_low)     w_high = SE(x_high)
         ↓                      ↓
    x_low' = w_low ⊙ x_low  x_high' = w_high ⊙ x_high
         └──────────┬──────────┘
                    ↓
              x_fused = x_low' + x_high'
                    ↓
         输出 = x + γ · Conv1×1(x_fused)
```

**为什么这在中层合理？**

- enc3 的分辨率为 H/4 × W/4，5×5 核的等效感受野 ≈ 20×20 像素，已经接近但还不足以覆盖模糊核
- 此时高/低频的语义分界开始变得有意义（低频≈局部结构，高频≈中频细节）
- 通道注意力 (SE) 的作用是：**保留有用的频率通道，抑制退化严重的频率通道**
- 不做空间级别的修改（那是 Tier 3 的事），只做通道级选择
- 与 Tier 1 的区别：Tier 1 做空间注意力（哪里重要），Tier 2 做通道注意力（哪个频带重要）

```python
class FreqSubbandRecalibration(nn.Module):
    """
    Tier 2: 频率子带重标定。
    
    将特征分解为高/低频后，对每个子带做独立的通道注意力加权，
    然后重新组合。不做空间级修改，避免中层频率处理过于激进。
    """

    def __init__(self, channels: int, kernel_size: int = 5, reduction: int = 4) -> None:
        super().__init__()
        self.low_pass = AdaptiveLowPassExtractor(channels, kernel_size=kernel_size)
        
        mid = max(channels // reduction, 8)
        
        # 低频通道注意力
        self.low_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        # 高频通道注意力
        self.high_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = self.low_pass(x)
        x_high = x - x_low
        
        x_low = x_low * self.low_se(x_low)
        x_high = x_high * self.high_se(x_high)
        
        fused = self.proj(x_low + x_high)
        return x + self.gamma * fused
```

> [!TIP]
> Tier 2 的参数量约 ~C×k² + 4×(C/r)×C + C² ≈ C×25 + 2C² + C²，约为完整 DFPB 的 **1/6**。

---

### Tier 3：完整双频处理（enc4）

**保持现有的 `DualFrequencyProgressiveBlock` 不变**。这一层已经验证有效 (+0.25~0.3)。

在 enc4 的 H/8 × W/8 分辨率上：
- 5×5 核的等效感受野 ≈ 40×40 像素，匹配模糊核尺度
- 低频 = 全局语义结构，高频 = 语义边界跳变
- 独立分支处理有明确的语义目标
- 可变形卷积融合可以做空间对齐

---

## 统一接口：FrequencyAwareBlock

为了方便在 NAFNet 中使用，设计一个统一的工厂类：

```python
class FrequencyAwareBlock(nn.Module):
    """
    Unified frequency-aware block with layer-adaptive behavior.
    
    tier=1: Spatial attention guided by high-freq energy (enc1, enc2)
    tier=2: Subband channel recalibration (enc3)
    tier=3: Full dual-frequency progressive block (enc4)
    """

    def __init__(
        self,
        channels: int,
        tier: int,
        low_kernel_size: int = 5,
        # Tier 3 专用参数
        low_blocks: int = 1,
        branch_expand_ratio: int = 2,
        use_deformable_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.tier = tier

        if tier == 1:
            self.block = FreqGuidedSpatialAttention(channels, kernel_size=low_kernel_size)
        elif tier == 2:
            self.block = FreqSubbandRecalibration(channels, kernel_size=low_kernel_size)
        elif tier == 3:
            self.block = DualFrequencyProgressiveBlock(
                channels,
                low_kernel_size=low_kernel_size,
                low_blocks=low_blocks,
                branch_expand_ratio=branch_expand_ratio,
                use_deformable_fusion=use_deformable_fusion,
            )
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tier == 3:
            return self.block(x, return_aux=False)
        return self.block(x)
```

### 在 NAFNet 中的集成方式

```python
# 在 NAFNetPhysicsInformed.__init__ 中：
self.freq_blocks = nn.ModuleList()
tier_map = {0: 1, 1: 1, 2: 2, 3: 3}  # enc层索引 → tier
chan = width
for i, num in enumerate(enc_blk_nums):
    self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
    self.freq_blocks.append(FrequencyAwareBlock(chan, tier=tier_map[i]))
    self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
    chan *= 2

# 在 forward 中：
for encoder, freq_block, down in zip(self.encoders, self.freq_blocks, self.downs):
    x = encoder(x)
    x = freq_block(x)  # ← 每层一个频率感知模块
    encs.append(x)
    x = down(x)
```

---

## 设计不冗余的关键原则

### 与 NAFBlock 的职责划分

```
                    NAFBlock                  FrequencyAwareBlock
                    ────────                  ───────────────────
关注什么？          通道间的交互              频域的结构-细节分离
注意力类型          通道注意力 (SCA)           空间注意力(T1) / 频带注意力(T2) / 频带处理(T3)
空间建模            3×3 DW Conv               低通核(自适应截止频率)
门控机制            SimpleGate (通道对半门控)   高低频分离后的加权/门控
参数共享            每个 block 独立            共享 AdaptiveLowPassExtractor 设计
```

> [!IMPORTANT]
> **不冗余的关键**：NAFBlock 完全不知道「频率」的概念，它的 SimpleGate 是通道维度的门控，SCA 是全局均值池化后的通道加权。FrequencyAwareBlock 引入的是 NAFBlock 缺失的 **频域感知能力**。
>
> 两者的正交性体现在：
> - NAFBlock 回答「哪些通道的特征重要？」
> - FrequencyAwareBlock 回答「这些特征中，哪些频率成分/空间位置需要特殊对待？」

### 避免梯度冲突的设计

三个 tier 的目标是 **一致的**（帮助恢复），但操作方式是 **正交的**：

| Tier | 修改维度 | 修改内容 | 不修改什么 |
|------|----------|----------|-----------|
| 1 | 空间 | 注意力权重 | 不改通道、不改频率内容 |
| 2 | 通道 | 高/低频子带的通道权重 | 不改空间分布 |
| 3 | 空间+通道+频率 | 完整的结构修复+细节整流 | — |

浅层（Tier 1）的梯度只通过空间注意力传播，不会与 Tier 3 在频率处理路径上产生冲突。

---

## 参数量与计算量对比

以 width=32 为例（enc1/2/3/4 通道数 = 32/64/128/256）：

| 层 | Tier | 方案 | 新增参数 | 对比：全部用 Tier 3 |
|----|------|------|----------|-------------------|
| enc1 (C=32) | 1 | 空间注意力 | ~1.1K | 48K (↓ 44×) |
| enc2 (C=64) | 1 | 空间注意力 | ~1.9K | 185K (↓ 97×) |
| enc3 (C=128) | 2 | 子带重标定 | ~35K | 720K (↓ 21×) |
| enc4 (C=256) | 3 | 完整 DFPB | ~2.8M | 2.8M (基准) |
| **总计** | — | — | **~2.84M** | **3.75M** |

> 分层设计比全层 Tier 3 **少了约 24% 的参数**，同时避免了浅层的性能退化。

---

## Decoder 侧是否也需要？

考虑到 NAFNet 的 skip connection 结构：

```
decoder4 ← up(bottleneck) + enc4_skip → decoder blocks
decoder3 ← up(dec4_out) + enc3_skip → decoder blocks
decoder2 ← up(dec3_out) + enc2_skip → decoder blocks
decoder1 ← up(dec2_out) + enc1_skip → decoder blocks
```

**建议：decoder 侧不加频率模块**，原因：

1. decoder 接收的是 encoder 的 skip feature + 上采样特征的 **和**，这个混合特征的频率特性比 encoder 更复杂
2. encoder 侧的 FrequencyAwareBlock 已经对 skip feature 做了频率感知处理，这个信息通过 skip connection 自然传递
3. decoder 的职责是融合多尺度信息并逐步恢复细节，与频率分解的目标有重叠
4. 减少参数，避免过拟合

如果实验表明 decoder 也有收益空间，建议仅在 **dec4**（最深层 decoder）尝试 Tier 2。

---

## 推荐实验路线

| 步骤 | 实验 | 预期 | 失败回退 |
|------|------|------|---------|
| 1 | enc4 保持 Tier 3（现有 baseline） | +0.25~0.3 | — |
| 2 | enc3 加 Tier 2 | 额外 +0.05~0.1 | 移除 enc3 模块 |
| 3 | enc1+enc2 加 Tier 1 | 额外 +0.02~0.05 | 移除，不影响步骤 2 |
| 4 | 如果步骤 2-3 有效，微调 tier 边界 | 探索性 | — |

> [!TIP]
> 每步增加的参数量很小，不需要改变学习率或训练策略。建议每步独立验证后再叠加。
