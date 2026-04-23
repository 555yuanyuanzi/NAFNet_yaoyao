# Patch Statistics Gate

## 模块目的

这个消融模块是一条独立的结构先验分支，不会替代现有的 `PA` 路径。
它用于验证这样一种思路：在保留原始像素级逐元素门控的前提下，
向 gate 中注入更丰富的 patch 级结构先验。

## 核心思想

这个模块不再只使用 patch mean，而是同时提取两类 patch 统计量：

- patch mean
- patch contrast（每个 patch 内的平均绝对偏差）

这两种统计量先通过一个轻量级分组 `1x1` 卷积进行融合，
再加到 gate 的第一分支上，最后仍然执行原始的逐元素门控：

`out = (x1 + structure_prior) * x2`

这样做的含义是：

- 保留原始的像素级 gate
- 给 gate 增加粗粒度结构信息
- 同时让 gate 感知局部变化强度

## 代码位置

- 模块文件：`basicsr/models/PSG.py`
- 架构接线：`basicsr/models/archs/NAFNet_arch.py`
- 示例配置：`options/train/GoPro/NAFNet-width64-psg.yml`

## 仓库中的使用方式

启用该模块时，配置中需要设置：

- `use_psg: true`
- `psg_patch_size: 8`
- `psg_stages: [...]`

在跑这个消融时，建议保持 `use_pa: false`，避免在同一个实验里混用两种不同的结构先验模块。
