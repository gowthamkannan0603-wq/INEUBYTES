# Task 1 – Computer Vision Using CNN Models
**Author:** Gowtham K | **Reg No:** INBT020009 | **Course ID:** AIINB10626

---

## Overview
This task implements and compares two CNN models on the CIFAR-10 image classification dataset:

| | Part A | Part B |
|---|---|---|
| Architecture | AlexNet-style CNN | Custom CNN with Residual Blocks |
| Augmentation | None | Flip + Shift + Rotation |
| Batch Norm | No | Yes |
| Dropout | No | Yes (0.4 + 0.3) |
| LR Schedule | Fixed Adam | Cosine Decay |
| Skip Connections | No | Yes |

---

## Dataset
- **CIFAR-10**: 60,000 images (32×32 RGB) across 10 classes
- **Split**: 40,000 train / 10,000 val / 10,000 test (fixed seed=42, same across both parts)
- **Preprocessing**: Pixel normalisation to [0, 1], no augmentation in Part A

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Part A — AlexNet-style baseline
cd partA
python train_cifar10_cnn.py

# Part B — Custom improved CNN
cd partB
python train_custom_cnn.py
```

---

## Part A — Architecture
```
Input (32×32×3)
→ Conv2D(64, 3×3) + ReLU → MaxPool(2×2)
→ Conv2D(128, 3×3) + ReLU → MaxPool(2×2)
→ Conv2D(256, 3×3) + ReLU
→ Conv2D(256, 3×3) + ReLU
→ Conv2D(128, 3×3) + ReLU → MaxPool(2×2)
→ Flatten
→ Dense(256) + ReLU
→ Dense(256) + ReLU
→ Dense(10) + Softmax
```
- **Target**: ≥ 70% test accuracy

---

## Part B — Architecture (Key Improvements)
```
Input (32×32×3)
→ Conv2D(64) + BN + ReLU → MaxPool        [Entry Block]
→ ResidualBlock(128) → MaxPool             [Stage 1]
→ ResidualBlock(256) → MaxPool             [Stage 2]
→ Conv2D(256) + BN + ReLU                  [Stage 3]
→ GlobalAveragePooling
→ Dropout(0.4) → Dense(256) + ReLU
→ Dropout(0.3) → Dense(10) + Softmax
```

### Why Each Change Improves Performance
1. **Batch Normalisation** — normalises activations, prevents internal covariate shift, enables faster/stabler training
2. **Residual (skip) connections** — gradient flows back unobstructed, enables effective training of deeper layers
3. **Data Augmentation** — flips + shifts + rotations increase effective dataset size and teach spatial invariance
4. **Global Average Pooling** — replaces flat Dense head; cuts parameters in the classifier, reduces overfitting
5. **Dropout** — stochastic regularisation that creates an implicit ensemble of sub-networks
6. **Cosine LR decay** — smooth learning rate annealing improves final convergence quality

---

## Expected Outputs
Each script saves results to a `results/` folder:
- `partA_results.png` — accuracy/loss curves + confusion matrix
- `partB_results.png` — accuracy/loss curves + confusion matrix + comparison bar chart
- `partA_stats.npy` — saved metrics from Part A (used by Part B for comparison)

---

## Deliverables Checklist
- [x] Python code (Part A + Part B)
- [x] requirements.txt
- [x] README
- [ ] Architecture diagrams (generate from model.summary() or draw.io)
- [ ] Performance table (fill after running)
- [ ] Confusion matrices (auto-saved as PNG)
- [ ] Google Doc report
