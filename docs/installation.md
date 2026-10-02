# Installation

Linux, Python 3.10, Java 8, PyTorch with CUDA 12.8, and an NVIDIA GPU. The default training recipe uses three GPUs.

```bash
git clone https://github.com/attacca-project/attacca.git && cd attacca
conda create -n minestudio -c conda-forge python=3.10 openjdk=8 && conda activate minestudio
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/CraftJarvis/MineStudio.git ../MineStudio
git -C ../MineStudio checkout 278aa8553668d591339dbf30d281594ed06ee882
git -C ../MineStudio apply "$PWD/patches/minestudio.patch"
pip install -e ../MineStudio -e .
```

MineStudio downloads the Minecraft engine on first use.
On a headless machine, install `xvfb` (`sudo apt-get install -y xvfb xauth`) and run the chain evaluation under `xvfb-run -a`.
`MINESTUDIO_GPU_RENDER=0` renders Minecraft in software while the policy runs on the GPU.
