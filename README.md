# Symmetry Discovery with Deep Learning

[![arXiv](https://img.shields.io/badge/arXiv-2112.05722-b31b1b.svg)](https://arxiv.org/abs/2112.05722)
[![DOI](https://img.shields.io/badge/DOI-10.1103%2FPhysRevD.105.096031-blue)](https://doi.org/10.1103/PhysRevD.105.096031)

Code accompanying the paper:

> **Symmetry Discovery with Deep Learning**  
> Krish Desai, Benjamin Nachman, Jesse Thaler  
> *Physical Review D* **105**, 096031 (2022)  
> [10.1103/PhysRevD.105.096031](https://doi.org/10.1103/PhysRevD.105.096031)

---

## Overview

This repository provides a machine-learning framework for automatically **discovering symmetries in data** using a Generative Adversarial Network (GAN) approach. A trainable, parameterized transformation (e.g., a rotation matrix) is placed inside the generator. The discriminator is trained to distinguish the original data from the transformed data. When the discriminator can no longer tell them apart, the learned transformation is a symmetry of the dataset.

The method is validated on synthetic Gaussian datasets with analytically known symmetries and then applied to real particle-physics data from the [LHC Olympics](https://lhco2020.github.io/homepage/) challenge.

---

## Repository Structure

```
symmetrydiscovery/
├── GaussianExperiments/        # Symmetry discovery on Gaussian distributions
│   ├── 1DGaussianAnalytic.ipynb
│   ├── 1DGaussianNumeric.ipynb
│   ├── 2DGaussianAGL2Plots.ipynb
│   ├── 2DGaussianNumeric.ipynb
│   ├── 2DGaussianO2.ipynb
│   ├── 2DGaussianGL2data.txt
│   └── 2DGaussianAGL2Slices/   # Parameter scans for the affine GL(2) group
│
├── LHCOlympics/                # Experiments on LHC collision data
│   ├── LHCO/
│   │   ├── LHCO.py             # 2-parameter SO(2) rotation discovery
│   │   ├── LHCO6.py            # 6-parameter SO(4) rotation discovery
│   │   ├── LHCO.sh / LHCO6.sh  # SLURM batch scripts
│   │   └── *.ipynb             # Jupyter notebook versions + loss analysis
│   ├── LHCOZ2.py               # Discrete Z2 symmetry discovery
│   └── *.ipynb                 # Additional analysis notebooks
│
├── MSE/                        # MSE-based symmetry discovery variant
│   ├── MSE7.py
│   └── *.ipynb
│
├── SymmetryDiscoveryMap/       # Comprehensive symmetry mapping
│   ├── 2DGaussian.ipynb
│   └── Augmentation.ipynb
│
├── CITATION.bib
└── CITATION.cff
```

---

## Method

The core idea is to learn the parameters of a transformation **T(θ)** such that:

1. **T(θ) · x ≈ x in distribution** — the discriminator cannot distinguish `T(θ)·x` from `x`.  
2. **T(θ) · T(θ) · x ≈ x** — the transformation is approximately an involution (group closure regularizer).

A custom Keras layer (`MyLayer`) implements the trainable transformation. Different experiments use different parameterizations:

| Experiment | Transformation | Parameters |
|---|---|---|
| 1D/2D Gaussian | SO(2) rotation | 1–2 angles |
| 2D Gaussian (GL2) | Affine GL(2) | 6 parameters |
| LHC data (LHCO) | 2D rotation on 4-vectors | θ₁, θ₂ |
| LHC data (LHCO6) | 6D rotation (SO(4) subgroup) | θ₁–θ₆ |
| LHC data (Z2) | Discrete reflection | c, s |

### Loss Function

```
L_total = L_GAN + α · L_symmetry
```

- **L_GAN**: binary cross-entropy between the discriminator outputs for real and transformed data.  
- **L_symmetry**: mean-squared error between `T(T(x))` and `x`, encouraging involutive group structure.  
- **α** (default 0.1): weight balancing the two terms.

### Network Architecture

**Generator** — a single `MyLayer` that applies the learned orthogonal transformation to the input.

**Discriminator** — a small feedforward network:
```
Input → Dense(25, relu) → Dense(25, relu) → Dense(1, sigmoid)
```

---

## Requirements

- Python ≥ 3.7  
- TensorFlow ≥ 2.2 (GPU recommended for LHC experiments)  
- NumPy  
- pandas  
- scikit-learn  
- matplotlib  
- Jupyter (for notebooks)

Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib jupyter
```

---

## Data

The LHC experiments (`LHCO.py`, `LHCO6.py`, `LHCOZ2.py`) require the LHC Olympics R&D dataset:

```
events_anomalydetection_DelphesPythia8_v2_qcd_features.h5
```

This file can be downloaded from the [LHC Olympics Zenodo record](https://zenodo.org/record/6466204).  
Place it in `LHCOlympics/LHCO/` before running the scripts.

Gaussian experiments generate synthetic data internally — no extra files needed.

---

## Usage

### Running a Python script directly

```bash
# 2D rotation discovery on LHC data
python LHCOlympics/LHCO/LHCO.py

# 6D rotation discovery on LHC data
python LHCOlympics/LHCO/LHCO6.py

# Discrete Z2 symmetry on LHC data
python LHCOlympics/LHCOZ2.py

# MSE-based rotation discovery
python MSE/MSE7.py
```

### Running on a SLURM cluster

```bash
sbatch LHCOlympics/LHCO/LHCO.sh
sbatch LHCOlympics/LHCO/LHCO6.sh
```

### Jupyter notebooks

```bash
jupyter notebook
```

Open any `.ipynb` file to reproduce individual figures or run interactive experiments.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Desai:2022dml,
  author  = {Desai, Krish and Nachman, Benjamin and Thaler, Jesse},
  title   = {Symmetry Discovery with Deep Learning},
  journal = {Phys. Rev. D},
  volume  = {105},
  pages   = {096031},
  year    = {2022},
  doi     = {10.1103/PhysRevD.105.096031},
  eprint  = {2112.05722},
  archivePrefix = {arXiv}
}
```

---

## License

See repository for license details.
