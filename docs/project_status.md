# NAFNet 频率增强模块项目状态

## 项目目标

本项目的目标是在 NAFNet 去运动模糊框架上引入高低频分离与融合模块，提升 GoPro 等去模糊任务的恢复质量，同时避免模块只在单一位置生效而显得像经验性 trick。

当前重点不是简单堆叠模块，而是验证以下问题：

- 高低频分离是否真的能带来稳定收益。
- 模块应该放在编码器浅层、编码器深层、middle，还是解码器。
- 浅层和深层是否应该采用不同的融合策略。
- 可变形融合是否能作为模块的核心贡献，而不是单纯增加参数量。

## 已完成内容

已经实现并接入了三类 DFPB 变体。

1. 原始 DFPB

代码位置：`basicsr/models/dfpb.py`

核心思路是使用可学习的空间低通滤波器提取低频分量，再用原特征减去低频得到高频分量。低通滤波器由 depthwise 卷积实现，并通过 softmax 归一化卷积核，使其更接近平滑滤波器。

需要注意的是，这种方法是频率启发的空间域方法，不是严格意义上的 FFT 频域方法。

2. FFTDFPB

代码位置：`basicsr/models/fftdfpb.py`

该版本使用 FFT 显式进行频率划分，通过中心圆形低频 mask 得到低频分量，再用原特征减去低频得到高频分量。它比原始 DFPB 更接近真正的频域方法。

当前已经删除了不需要的 `low_kernel_size` 参数，使 FFTDFPB 的配置更干净。主要控制参数是 `fft_radius_ratio` 或 `radius_ratio`。

3. WaveDFPB

代码位置：`basicsr/models/wavedfpb.py`

该版本使用一级 Haar 小波分解进行高低频划分：

- `LL` 分量作为低频信息。
- `LH`、`HL`、`HH` 分量作为高频信息。
- 小波分解后再重建回与输入相同的特征尺寸。

WaveDFPB 的优势是高低频分离方式是固定的、可解释的，不需要像卷积核大小或 FFT 半径那样针对不同层手动设置尺度参数。

## 关键架构

NAFNet 主干已经扩展支持三种模块：

代码位置：`basicsr/models/archs/NAFNet_arch.py`

支持的开关包括：

- `use_dfpb`
- `use_fftdfpb`
- `use_wavedfpb`

对应的 stage 配置包括：

- `dfpb_stages`
- `fftdfpb_stages`
- `wavedfpb_stages`

对应的模块参数包括：

- `dfpb_kwargs`
- `fftdfpb_kwargs`
- `wavedfpb_kwargs`

同时支持 `stage_kwargs`，可以对不同 stage 单独设置参数。例如可以让浅层关闭可变形融合，而深层保留可变形融合。

当前推荐的 WaveDFPB 配置是：

```yaml
network_g:
  use_wavedfpb: true
  wavedfpb_stages: [enc1, enc2, enc3, enc4]
  wavedfpb_kwargs:
    low_blocks: 1
    branch_expand_ratio: 2
    use_deformable_fusion: true
    stage_kwargs:
      enc1:
        use_deformable_fusion: false
      enc2:
        use_deformable_fusion: false
      enc3:
        use_deformable_fusion: false
      enc4:
        use_deformable_fusion: true
```

对应配置文件：

`options/train/GoPro/NAFNet-width64-wavedfpb.yml`

这个设计的含义是：

- enc1、enc2、enc3 使用小波高低频分离，但不使用可变形融合，避免在高分辨率浅层过度扰动细节。
- enc4 保留完整 WaveDFPB，包括低频引导的可变形融合，用于深层语义和结构特征的对齐与增强。

## 可变形融合的作用

可变形融合主要用于解决运动去模糊中的局部错位问题。

运动模糊会导致边缘、纹理和结构在空间上发生不一致。普通卷积只能在固定采样位置融合特征，而可变形卷积可以根据输入内容预测 offset 和 mask，从而在融合高低频特征时进行自适应采样。

在当前模块中，可变形融合更适合解释为：

- 低频分支提供稳定结构和轮廓引导。
- 高频分支提供边缘、纹理和细节。
- 可变形融合根据低频结构引导高频细节进行局部对齐。

因此它不是单纯增加复杂度，而是服务于去运动模糊中的非均匀模糊和局部错位。

## 当前问题

1. 原始 DFPB 只在 enc4 或 middle 上效果明显

已有实验现象是，DFPB 加在 enc4 能提升约 0.2 到 0.3 dB，加在 middle 也有一定提升，但加在 enc1、enc2、enc3 效果不好，甚至多层加入后可能低于 baseline。

这说明模块不是简单越多越好。可能原因包括：

- 浅层特征分辨率高，固定大小的空间低通卷积不一定适合。
- 浅层主要保留局部纹理和边缘，过强的高低频重组可能破坏原始细节。
- 深层特征分辨率低、语义更强，更适合做结构级频率建模。
- 多层加入会改变主干特征分布，训练难度和优化路径都会变化。

2. 空间低通滤波不够像真正的频域方法

原始 DFPB 的低通滤波器本质上仍然是 CNN 卷积，只能说是频率启发或低通先验，不能严格说使用了显式频域增强。

如果论文中强调频域建模，FFTDFPB 或 WaveDFPB 更容易解释。

3. FFT 半径存在层间尺度问题

FFTDFPB 的 `radius_ratio` 控制低频区域大小。不同层的特征分辨率不同，同一个半径比例在不同 stage 上可能对应不同的实际频率含义。

这会带来额外的超参数选择问题。

4. WaveDFPB 更干净，但仍需要消融证明

WaveDFPB 用 Haar 小波固定划分高低频，不需要给不同层设置卷积核大小或 FFT 半径，因此更适合做多层统一设计。

但它是否应该每层都加、浅层是否关闭可变形融合、深层是否保留完整模块，都需要实验确认。

## 当前推荐实验路线

为了避免模块设计显得像 trick，建议按以下顺序做消融：

1. Baseline NAFNet

不加任何 DFPB，用作基准。

2. WaveDFPB enc4

只在 enc4 加完整 WaveDFPB，验证深层高低频建模是否有效。

3. WaveDFPB enc1-enc4，shallow no deform

enc1、enc2、enc3 使用 WaveDFPB 但关闭可变形融合，enc4 保留完整可变形融合。

这是当前最推荐的主实验设置。

4. WaveDFPB enc1-enc4，all no deform

所有编码层都关闭可变形融合，用来验证小波高低频分离本身的贡献。

5. WaveDFPB enc1-enc4，all deform

所有编码层都开启可变形融合，用来验证浅层可变形融合是否会破坏细节或拖慢训练。

6. 原始 DFPB 与 FFTDFPB 对比

用于说明不同高低频提取方式的差异：

- 原始 DFPB：空间低通卷积。
- FFTDFPB：显式 FFT 频域划分。
- WaveDFPB：Haar 小波多分量划分。

## 下一步计划

短期建议先围绕 WaveDFPB 做实验，因为它的故事更清晰，配置也更干净。

优先完成：

- 训练 `NAFNet-width64-wavedfpb.yml` 当前配置。
- 记录 GoPro PSNR、SSIM、参数量和推理速度。
- 对比 baseline、enc4-only、enc1-enc4 shallow no deform。
- 如果 shallow no deform 有稳定收益，再补 all no deform 和 all deform 消融。

论文叙述可以围绕以下主线展开：

NAFNet 的深层编码器具有较强结构建模能力，但普通卷积对运动模糊造成的频率混叠和局部错位缺少显式约束。为此，引入小波高低频分解，将稳定结构信息和细节纹理分开建模；浅层保留轻量频率分解以减少对细节的扰动，深层使用低频引导的可变形融合来对齐高频细节，从而提升非均匀运动模糊下的恢复能力。

## 已验证内容

当前已经完成基础代码检查：

- `dfpb.py`、`fftdfpb.py`、`wavedfpb.py`、`NAFNet_arch.py` 可以通过 Python 编译检查。
- WaveDFPB 在小规模通道数下可以正常实例化。
- `use_deformable_fusion: false` 时，不再创建不必要的 `offset_mask` 分支，模块更干净，参数量和计算量更低。

需要注意的是，完整 width64 模型在当前 Windows 环境中实例化时可能遇到 page file 内存问题，这更像是本地环境限制，不一定是代码逻辑错误。
