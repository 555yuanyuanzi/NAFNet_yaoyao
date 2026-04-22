# NAFNet 模块设计总览

## 文档目的

这份文档用于统一说明当前仓库中 3 个主要外挂模块的：

- 定义是什么
- 具体怎么设计
- 放在 NAFNet 里起什么作用
- 更适合解决什么问题

当前覆盖的模块包括：

- `PA`：Patch Averaging
- `GDPM`：Global Directional Prior Modulation
- `DFPB`：Dual-Frequency Progressive Block

相关代码位置：

- `basicsr/models/PA.py`
- `basicsr/models/GDPM.py`
- `basicsr/models/dfpb.py`
- `basicsr/models/archs/NAFNet_arch.py`

## 总体定位

这 3 个模块的目标不同，不应该混为一谈：

| 模块 | 核心定位 | 更偏什么 | 当前最适合的作用 |
| --- | --- | --- | --- |
| `PA` | 局部低频结构提取 | 结构、平滑、低通 | 轻量结构先验 |
| `GDPM` | 全局频谱提示 | 退化感知、条件调制 | 全局 prompt / conditioner |
| `DFPB` | 低频到高频的渐进恢复 | 结构到细节、融合 | 主恢复块 |

一句话概括：

- `PA` 更像一个轻量结构工具
- `GDPM` 更像一个全局退化提示器
- `DFPB` 更像一个真正的方法模块

---

## 1. PA：Patch Averaging

### 1.1 模块定义

`PA` 的核心操作是：

1. 将特征图按固定 `patch_size x patch_size` 划分为不重叠 patch
2. 对每个 patch 求平均
3. 再把 patch 均值 broadcast 回原 patch 区域

在代码中，对应 [PatchAveraging](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/PA.py:5>)。

它本质上是一个 **patch 级的局部低通/结构聚合算子**。

### 1.2 当前实现

当前实现里：

- 默认 `patch_size = 8`
- 输入要求 `H` 和 `W` 都能被 `patch_size` 整除
- 否则当前上层逻辑会跳过该次 `PA` 调用

对应代码：

- [PatchAveraging.forward](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/PA.py:10>)
- [PatchAwareGate.forward](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/archs/NAFNet_arch.py:38>)

### 1.3 在 NAFNet 中的接法

当前不是直接用 `PA(x)` 替换原特征，而是把它接在 `NAFBlock` 第一段 `SimpleGate` 位置：

1. `x` 经过 `conv1 -> conv2`
2. 在 gate 前拆成 `x1, x2`
3. 对 `x1` 施加 `PA`
4. 用 `x = (x1 + pa_scale * PA(x1)) * x2`

其中 `pa_scale` 是零初始化的可学习参数，用来控制 `PA` 分支的参与强度。

### 1.4 设计动机

`PA` 的设计动机不是增强高频，而是：

- 提供更稳定的局部结构聚合
- 引入 patch 级低频先验
- 帮助浅层或中层特征更关注结构，而不是只盯局部像素扰动

### 1.5 用处

`PA` 更适合解决：

- 局部结构不稳定
- 模糊区域轮廓不够稳
- 浅层特征过于噪声化、纹理化

它不太适合直接承担：

- 高频重建
- 最终细节锐化
- 全局退化建模

### 1.6 优点与局限

优点：

- 很轻
- 易插入现有 CNN block
- 有明确的结构先验含义

局限：

- 天然偏平滑
- 放在 decoder 时容易压细节
- 当前实现受 `patch_size` 与特征图尺寸整除关系限制

### 1.7 当前建议

如果把 `PA` 作为模块验证：

- 更适合先放 encoder
- 不建议第一版就大量放在 decoder
- 如果测试是 GoPro 全图 `1280x720`，当前最稳的是浅层 encoder

---

## 2. GDPM：Global Directional Prior Modulation

### 2.1 模块定义

`GDPM` 是一个 **全局频谱提示模块**，不是直接做恢复的分支。

它从输入模糊图像中提取频谱先验，然后将先验转成一个低维 prompt，对早期特征进行有界调制。

对应代码：

- [GlobalDirectionalPriorModulation](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/GDPM.py:10>)

### 2.2 当前设计流程

当前 `GDPM` 的主流程是：

1. 输入模糊图像
2. 下采样到固定 `prior_size`
3. 做 FFT 和 `fftshift`
4. 取 `log magnitude`
5. 做径向均值归一化
6. 在中高频环带上做方向池化
7. 提取低/高频统计
8. 得到一个小的频谱提示向量
9. 用 MLP 生成 `gamma`
10. 用 `gamma` 对特征做残差缩放

### 2.3 当前先验向量

当前 prompt 向量由两部分组成：

- `direction_prior`
- `band_prior`

具体为：

- `direction_prior`: 若干方向上的频谱对比
- `band_prior`: `[high_energy, high_minus_low]`

即：

- `prior = [direction_prior, high_energy, high_minus_low]`

### 2.4 在 NAFNet 中的接法

当前 `GDPM` 放在 `intro` 之后、进入 encoder 之前：

1. `x = intro(inp)`
2. `x = gdpm(inp, x)`
3. 再进入 encoder

这是一个很典型的：

- 图像提退化先验
- 特征受先验调制

的 conditioner 接法。

### 2.5 设计动机

它要解决的问题是：

- NAFNet 本身偏局部卷积
- 对全局退化统计感知较弱
- motion blur 又常常带有方向性和频谱衰减特征

因此，`GDPM` 的目标不是重建图像，而是给主干一个：

- 全局方向提示
- 高频衰减强度提示
- 模糊严重程度提示

### 2.6 用处

`GDPM` 更适合解决：

- 主干缺少全局退化感知
- 不同样本模糊强度和方向差异大
- 希望用很轻的方式引入全局条件

它不太适合单独承担：

- 局部空间变化模糊的显式恢复
- 高频细节重建
- 大范围结构补偿

### 2.7 优点与局限

优点：

- 轻量
- 不直接和主干抢恢复权
- 训练相对稳
- 很适合作为 prompt 模块

局限：

- 提供的是全局频谱提示
- 对局部、空间变化模糊的建模是间接的
- 本质更像条件调制，不是主恢复器

### 2.8 当前建议

`GDPM` 最适合：

- 单独作为 `NAFNet + GDPM` 一组消融
- 或以后给更复杂模块提供全局退化 cue

---

## 3. DFPB：Dual-Frequency Progressive Block

### 3.1 模块定义

`DFPB` 是一个 **双频渐进恢复块**。

它不是简单做“高低频分开处理”，而是把恢复过程组织成：

1. 先恢复低频结构
2. 再在结构引导下纠偏高频细节
3. 最后用可变形方式融合

对应代码：

- [DualFrequencyProgressiveBlock](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/dfpb.py:230>)

### 3.2 组成部分

当前 `DFPB` 由 4 个子模块组成：

#### A. AdaptiveLowPassExtractor

对应：

- [AdaptiveLowPassExtractor](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/dfpb.py:15>)

作用：

- 用逐通道可学习低通核提取 `x_low`
- 用 `x_high = x - x_low` 得到高频残差

#### B. LowFrequencyRestorer

对应：

- [LowFrequencyRestorer](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/dfpb.py:61>)

作用：

- 恢复粗结构
- 稳定轮廓和大尺度模糊区域

#### C. HighFrequencyRectifier

对应：

- [HighFrequencyRectifier](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/dfpb.py:74>)

作用：

- 接收高频残差和低频恢复结果
- 预测一个空间 gate
- 用 gate 控制高频纠偏强度

它强调的是 **选择性纠偏**，不是盲目增强高频。

#### D. LowGuidedDeformableFusion

对应：

- [LowGuidedDeformableFusion](</c:/Users/86155/Desktop/shiyancode/NAFNet/basicsr/models/dfpb.py:175>)

作用：

- 在低频引导下对齐高频分支
- 以 motion-aware 的方式融合 base / low / high 三路特征

如果运行环境没有 `torchvision.ops.deform_conv2d`，当前实现会自动退化成普通卷积融合。

### 3.3 在 NAFNet 中的接法

当前 `DFPB` 已经支持按 stage 开关接入：

- `use_dfpb`
- `dfpb_stages`
- `dfpb_kwargs`

默认更推荐先放：

- `middle`

后续再试：

- `enc4 + middle`

### 3.4 设计动机

`DFPB` 的故事不是“用了频率分解”，而是：

- motion blur 对结构和细节的破坏方式不同
- 用同一条恢复路径同时处理两者容易失衡
- 高频恢复应该依赖低频结构
- 两个分支最终还要考虑局部对齐问题

### 3.5 用处

`DFPB` 更适合解决：

- 结构恢复和细节恢复目标冲突
- 高频细节恢复容易出现伪纹理或振铃
- 需要一个真正参与恢复的主模块，而不仅仅是 prompt

### 3.6 优点与局限

优点：

- 论文故事清楚
- 比单纯频域块更贴近运动去模糊任务
- 兼顾结构恢复、细节纠偏和融合

局限：

- 比 `PA` 和 `GDPM` 更重
- 可变形融合部分依赖运行环境
- 如果插得太多，训练风险会上升

### 3.7 当前建议

最合适的验证顺序是：

1. `NAFNet`
2. `NAFNet + GDPM`
3. `NAFNet + PA`
4. `NAFNet + DFPB(middle)`
5. `NAFNet + DFPB(enc4 + middle)`

---

## 4. 三个模块之间的关系

可以把它们理解成三种不同层次的设计：

### PA：结构工具

- 更轻
- 更局部
- 更像 block 内的小操作

### GDPM：全局退化提示

- 更轻
- 更全局
- 更像条件调制器

### DFPB：主恢复块

- 更完整
- 更偏方法主线
- 更适合作为论文主模块

如果后面要收束成一篇论文主线，当前最容易讲故事的是：

- `GDPM` 作为轻量全局退化提示
- `DFPB` 作为主恢复块

而 `PA` 更适合作为：

- 对照模块
- 低频结构先验工具
- 或训练期辅助约束的来源

---

## 5. 当前消融配置建议

建议至少保留下面几组：

1. baseline `NAFNet`
2. `NAFNet + PA`
3. `NAFNet + GDPM`
4. `NAFNet + DFPB`

如果后续需要更深入，再补：

5. `NAFNet + GDPM + DFPB`
6. `NAFNet + DFPB(enc4 + middle)`

这样更容易判断：

- 结构先验是否有用
- 全局频谱提示是否有用
- 双频渐进恢复是否真能带来增益
