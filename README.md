# Cross-modal Representation Shift Refinement for Point-supervised Video Moment Retrieval (ACM TOIS 2026)

> Official PyTorch implementation of **DRONE**, a point-supervised VMR framework that mitigates cross-modal representation shift via Pseudo-Frame Temporal Alignment (PTA) and Curriculum-Guided Semantic Refinement (CSR).

## Authors

**Kun Wang**<sup>1</sup>, **Yupeng Hu**<sup>1</sup>\*, **Hao Liu**<sup>1</sup>, **Jiang Shao**<sup>1</sup>, **Liqiang Nie**<sup>2</sup>

<sup>1</sup> School of Software, Shandong University, Jinan, China  
<sup>2</sup> School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China
\* Corresponding author

## Paper and Links

- Paper (ACM TOIS 2026): https://dl.acm.org/doi/10.1145/3786606

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Checkpoints](#checkpoints)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)
- [Contact](#contact)

## Updates

- [04/2026] Initial public code release.

---

## Introduction

This repository contains the implementation of **DRONE: Cross-modal Representation Shift Refinement for Point-supervised Video Moment Retrieval** (ACM TOIS 2026).

Point-supervised Video Moment Retrieval (VMR) aims to localize the temporal segment in a video that matches a natural language query using only point-level supervision. DRONE addresses the cross-modal representation shift issue in this setting by introducing **Pseudo-Frame Temporal Alignment (PTA)** and **Curriculum-Guided Semantic Refinement (CSR)**, which progressively improve temporal alignment and semantic consistency between video and text representations.

This repository currently provides:

- training code for point-supervised VMR
- evaluation code for benchmark datasets
- checkpoint files for quick reproduction
- utilities for dataset preparation and experiment management

---

## Highlights

- Supports three standard datasets: `activitynetcaptions`, `charadessta`, `tacos`.
- Includes training and evaluation scripts.
- Includes checkpoint folders for quick reproduction:
  - `Act_ckpt/`
  - `Cha_ckpt/` (C3D / I3D / VGG)
  - `TACoS_ckpt/`

---

## Method Overview

![Pipeline](./assets/pipeline.png)

---

## Repository Structure

```text

|- assets/
|  |- pipeline.png
|- src/
|  |- config.yaml
|  |- dataset/
|  |  |- dataset.py
|  |  |- generate_glance.py
|  |  |- generate_duration_glance.py
|  |- experiment/
|  |  |- train.py
|  |  |- eval.py
|  |- model/
|  |  |- model.py
|  |  |- building_blocks.py
|  |- utils/
|  |  |- utils.py
|  |  |- vl_utils.py
|- README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iLearn-Lab/DRONE.git
cd DRONE
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install numpy scipy pyyaml tqdm
```

---

## Dataset Preparation

### 1. Prepare features and raw annotations

Follow [ViGA](https://github.com/r-cui/ViGA)'s dataset preparation for:
- ActivityNet Captions
- Charades-STA
- TACoS


### 2. Check path configuration

This repo currently contains local paths (for example under `data/...`), mainly in:

- `src/config.yaml`
- `src/utils/utils.py`

Before running, replace them with your local dataset root and feature paths.

---

## Checkpoints

The cloud links of checkpoints: [Google Drive](https://drive.google.com/drive/folders/1QAocP9-h3YrtJuF0xbKRheFiZBQ4L2TI?usp=sharing).

---

## Training
ActivityNet Captions:
```bash
python -m src.experiment.train --task activitynetcaptions
```
Charades-STA:
```bash
python -m src.experiment.train --task charadessta
```
TACoS:
```bash
python -m src.experiment.train --task tacos
```

---

## Evaluation

Evaluate a trained experiment folder:

```bash
python -m src.experiment.eval --exp path/to/your/experiment_folder
```

The folder should contain:

- `config.yaml`
- `model_best.pt`


---

## Results

### Charades-STA (I3D)

| Method | Venue | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---:|---:|---:|---:|
| ViGA | SIGIR 2022 | 71.21 | 45.05 | 20.27 | 44.57 |
| CFMR | MM 2023 | - | 48.14 | 22.58 | - |
| SG-SCI | MM 2024 | 70.30 | 52.07 | 27.23 | 46.77 |
| **DRONE (Ours)** | TOIS 2026 | **71.48** | **53.68** | **27.34** | **47.43** |

### Charades-STA (VGG)

| Method | Venue | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---:|---:|---:|---:|
| ViGA | SIGIR 2022 | 60.22 | 36.72 | 17.20 | 38.62 |
| D3G | ICCV 2023 | - | 41.64 | 19.60 | - |
| **DRONE (Ours)** | TOIS 2026 | **61.21** | **42.37** | **20.54** | **40.25** |

### Charades-STA (C3D)

| Method | Venue | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---:|---:|---:|---:|
| ViGA | SIGIR 2022 | 56.85 | 35.11 | 15.11 | 36.35 |
| **DRONE (Ours)** | TOIS 2026 | **57.12** | **37.90** | **17.69** | **37.12** |

### ActivityNet Captions (C3D)

| Method | Venue | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---:|---:|---:|---:|
| ViGA | SIGIR 2022 | **59.61** | 35.79 | 16.96 | 40.12 |
| CFMR | MM 2023 | - | 36.97 | 17.33 | - |
| D3G | ICCV 2023 | 58.25 | 36.68 | **18.54** | - |
| **DRONE (Ours)** | TOIS 2026 | 59.47 | **37.68** | 17.47 | **40.30** |

### TACoS (C3D)

| Method | Venue | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---:|---:|---:|---:|
| ViGA | SIGIR 2022 | 19.62 | 8.85 | 3.22 | 15.47 |
| CFMR | MM 2023 | 25.44 | 12.82 | - | - |
| D3G | ICCV 2023 | 27.27 | 12.67 | 4.70 | - |
| SG-SCI | MM 2024 | **37.47** | 20.59 | 8.27 | 23.83 |
| **DRONE (Ours)** | TOIS 2026 | 36.12 | **21.57** | **8.70** | **23.91** |

---

## Citation

```bibtex
@article{wang2026cross,
  title={Cross-Modal Representation Shift Refinement for Point-supervised Video Moment Retrieval},
  author={Wang, Kun and Hu, Yupeng and Liu, Hao and Shao, Jiang and Nie, Liqiang},
  journal={ACM Transactions on Information Systems},
  volume={44},
  number={3},
  pages={1--30},
  year={2026},
  publisher={ACM New York, NY}
}
```

---

## Acknowledgement
- This implementation and data organization are inspired by [ViGA](https://github.com/r-cui/ViGA).
- Thanks to all collaborators and contributors of this project.
---

## License

This project is released under the Apache License 2.0.

---

## Contact
**If you have any questions, feel free to contact me at khylon.kun.wang@gmail.com**.