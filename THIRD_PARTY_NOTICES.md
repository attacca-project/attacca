# Third-party notices

The MIT License in [LICENSE](LICENSE) covers original Attacca material only.
Third-party material listed below remains under its own terms.

## MineStudio

- Source: https://github.com/CraftJarvis/MineStudio, commit `278aa8553668d591339dbf30d281594ed06ee882`.
- License: MIT License, Copyright (c) 2024 Shaofei Cai.
- Full license text: [LICENSES/MineStudio-MIT.txt](LICENSES/MineStudio-MIT.txt).
- Use: simulator, offline training utilities, and data modules. MineStudio is installed separately.
  `patches/minestudio.patch` supplies simulator launch, reset, and callback changes.
  Original MineStudio material remains subject to its MIT license.

## ROCKET-2

- Source: https://github.com/CraftJarvis/ROCKET-2, commit `5ea1b3b2e4a4b43af8f00a75f232bb490ec8fb6a`.
  The pinned upstream repository does not include a license file.
- The files below are adapted from ROCKET-2 and credited to its authors. They are not covered by this repository's
  MIT License and remain subject to the ROCKET-2 authors' terms.
- Adapted files:
  - `src/attacca/models/model.py`: `CrossViewRocket` and `ActionEmbeddingLayer`, adapted from upstream `model.py`
    with target grounding and phase conditioning.
  - `src/attacca/models/cfg_wrapper.py`: `CFGWrapper`, adapted from upstream `cfg_wrapper.py`.
  - `src/attacca/training/trainer.py`: training structure adapted from upstream `train.py`.
  - `configs/attacca.yaml`: model and optimization settings adapted from upstream `config.yaml`.
- Training initialization: `phython96/ROCKET-2-1x-22w` on Hugging Face, released under the MIT License according to its model card.
  The Attacca model weights, distributed separately on Hugging Face, are fine-tuned from this initialization
  and released under the MIT License (see the model card).

## Minecraft engine

MineStudio downloads the Minecraft engine, an MCP-Reborn build named `mcprec-6.13.jar`.
The engine is not redistributed in this code repository.
The renderer patch in `src/attacca/worlds/human_engine/` is compiled and applied to the downloaded engine at run time;
the patched engine is written to a local cache.

## Visual backbones

The model uses the timm backbones `vit_base_patch16_224.dino` and `vit_tiny_patch16_224.augreg_in21k_ft_in1k`.
Training may download their pretrained weights from Hugging Face.
The complete Attacca inference model includes its visual-backbone tensors, so loading that model does not download them again.
Their pretrained weights are released under the Apache License 2.0 on Hugging Face. The Attacca model file contains them,
and those tensors remain under that license.
