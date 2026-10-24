# MEUNet-CS (AAAI 2026)

Official PyTorch implementation of **Breaking Measurement Barriers: From Compressed Sensing to Deep Reconstruction**.

Gang Qu, Ping Wang, Siming Zheng, and Xin Yuan.

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37815) | [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37815/41777)

## Overview

MEUNet is a measurement-enhanced unfolding network for image compressed sensing. It combines asymmetric Kronecker compressed sensing (AKCS), which improves sensing incoherence, with measurement-aware cross-attention (MACA), which uses measurement information to guide reconstruction. A shared convolution-Transformer denoiser provides efficient multi-stage reconstruction. The paper evaluates the method on Set11, CBSD68, and real single-pixel imaging data.

## Installation

The reference environment is Linux with an NVIDIA GPU, Python 3.13.2, PyTorch 2.6.0, and the CUDA 12.6 PyTorch build. Package versions in `requirement.txt` are taken from the working `SPI` environment.

```bash
git clone https://github.com/Gang-Qu/MEUNet-CS.git
cd MEUNet-CS
conda create -n meunet python=3.13.2 -y
conda activate meunet
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

An NVIDIA driver compatible with the CUDA 12.6 PyTorch build is required. Use Python 3.13 with the pinned packages; NumPy 2.1.3 is not intended for Python 3.14.

## Repository Structure

```text
MEUNet-CS/
  model/MEUNet.py       # Network architecture
  Weight/epoch_306.pth # Pretrained checkpoint at approximately 10% sampling
  opts.py             # Model, dataset, and optimization settings
  train.py            # Training
  test.py             # Evaluation and reconstruction image export
  utils.py            # Data loading, metrics, and checkpoint utilities
  para_FLOPS.py       # Computational complexity profiling script
  requirement.txt     # Reference package versions
```

## Data Preparation

Prepare the training images (BSDS400) and benchmark images (e.g., Set11 or CBSD68) in separate directories. Datasets are not included. Supply the directories using `--train_data_path` and `--benchmark_path`.

The supplied loaders use the Y channel of OpenCV's YCrCb conversion. The evaluation script supports images matching `args.size`, or images with twice that height and width, which are split into four patches.

## Pretrained Model and Testing

The included `Weight/epoch_306.pth` checkpoint uses `size=[256, 256]`, `meas_size=[81, 81]`, and seven unfolding stages. Its effective sampling ratio is `81 * 81 / (256 * 256)`, approximately 10%.

**Before testing this checkpoint, change the `--meas_size` default in `opts.py` from `[128, 128]` to `[81, 81]`.** Keep the other architecture settings at their defaults. The current argument parser does not parse list-valued options from the command line; edit `opts.py` for image sizes, measurement sizes, and block lists. Changing `--cr` alone does not change the sensing matrix dimensions.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --test_weight_path ./Weight/epoch_306.pth \
  --benchmark_path /path/to/Set11 \
  --device cuda
```

PSNR and SSIM are printed and saved in `test_results/log/`. Ground-truth/reconstruction image pairs are saved in `test_results/<decoder_type>/1/test/`.

The checkpoint loader reports the numbers of model, pretrained, and loaded parameter entries. They should all match for this checkpoint; mismatched measurement dimensions can otherwise silently skip checkpoint entries.

## Training

Set the desired image and measurement sizes in `opts.py`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_data_path /path/to/BSDS400 \
  --benchmark_path /path/to/Set11 \
  --device cuda
```

Checkpoints and logs are saved under `weights/` and `log/`. Set `--pretrained_model_path /path/to/checkpoint.pth` to initialize from pretrained weights. The released script defaults are configurable; consult the paper for the experimental training schedule.

## Citation

```bibtex
@article{qu2026breaking,
  title   = {Breaking Measurement Barriers: From Compressed Sensing to Deep Reconstruction},
  author  = {Qu, Gang and Wang, Ping and Zheng, Siming and Yuan, Xin},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {11},
  pages   = {8631--8639},
  year    = {2026},
  doi     = {10.1609/aaai.v40i11.37815}
}
```

## Contact

For questions, please open a GitHub issue or contact [Gang Qu](mailto:qugang@westlake.edu.cn).
