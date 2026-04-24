# Interpolated Patch Prior

## Goal

IPP is a lightweight refinement of plain patch averaging.
It keeps the patch-mean prior, but replaces hard repeated broadcasting with
smooth bilinear interpolation.

## Design

Given a feature map `x`, IPP performs:

- patch-wise average pooling with stride equal to `patch_size`
- bilinear upsampling back to the original resolution

The module output is:

`IPP(x) = U(AvgPool(x))`

where `U(.)` is bilinear interpolation.

To keep the operator robust at arbitrary inference sizes, the feature map is
first padded to a multiple of `patch_size`, then pooled and resized back to the
original resolution.

## NAF Gate Formulation

Inside the NAF gate, IPP is injected only into the first branch:

`out = (x1 + ipp_scale * IPP(x1)) * x2`

This keeps the branch identity-initialized through `ipp_scale = 0`.

## Why It Is Different From PA

Vanilla patch averaging restores the pooled prior with hard patch replication.
IPP instead reconstructs the prior with smooth interpolation, which avoids the
blocky patch boundaries introduced by repeated tiling.

## Files

- module: `basicsr/models/IPP.py`
- integration: `basicsr/models/archs/NAFNet_arch.py`
- example config: `options/train/GoPro/NAFNet-width64-ipp.yml`
