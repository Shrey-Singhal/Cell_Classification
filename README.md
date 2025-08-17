# Cell Classification with CNNs (Histopathology)

End-to-end deep learning project that classifies microscopic cell image patches into:
- **Task 1 (4-class):** fibroblast (0), inflammatory (1), epithelial (2), others (3)
- **Task 2 (binary):** cancerous vs non-cancerous *(epithelial ≡ cancerous in this setup)*

Built with **TensorFlow/Keras**. Includes data preparation, augmentation, **patient-aware splits** (to avoid leakage), model training, hyperparameter tuning, and evaluation.


## Dataset

- **Images:** `patch_images/` (**27×27 RGB** patches; across **99 patients**)
- **Labels / metadata (included):**
  - `data/data_labels_mainData.csv`
  - `data/data_labels_extraData.csv`

## Tasks

- **4-class classification** - cellType ∈ {0: fibroblast, 1: inflammatory, 2: epithelial, 3: others}

- **Binary cancer detection** - isCancerous ∈ {0, 1} (epithelial treated as cancerous)

## Methodology
**Preprocessing & Splits**

- Merge `data_labels_mainData.csv` and `data_labels_extraData.csv`

- **Patient-aware split** (to prevent leakage): Train 60% / Val 20% / Test 20% by patientID.

- Image normalization.

- **ImageDataGenerator** augmentations on train only (e.g., rotations, flips). Validation/Test are not augmented.


**Imbalance Handling**

- Added extra malignant samples (epithelial) from the extra labels CSV.

- Train-time augmentation to increase effective sample diversity.

**Models & Tuning**

- Baseline CNN (Keras Sequential): Conv → ReLU → MaxPooling → Flatten → Dense (softmax/sigmoid).

- Keras Tuner (Bayesian) explored:

  - number of conv blocks, filters, dense units

  - dropout, L2 weight decay

  - optimizer / learning rate (SGD/Adam variants)

- Trained with multiple schedules (e.g., 50, 200, 500 epochs).

## Results (from this project)

### Task 1 — 4-class (categorical accuracy)

| Setup          | Train Acc | Val/Test Acc        |
|----------------|-----------|---------------------|
| Base (50 ep)   | ≈ 0.59    | ≈ **0.64** (test)   |
| Tuned (50 ep)  | ≈ 0.62    | ≈ **0.68** (test)   |
| Tuned (500 ep) | ≈ 0.665   | **≈ 0.730** (val)   |

*Trend:* Initial underfitting reduced after tuning and longer training.

### Task 2 — Binary (accuracy)

| Setup           | Train Acc | Val/Test Acc         |
|-----------------|-----------|----------------------|
| Base (50 ep)    | —         | **≈ 0.824** (binary) |
| Tuned (200 ep)  | ≈ 0.982   | **≈ 0.857** (test)   |
| Tuned (500 ep)  | ≈ 0.971   | **≈ 0.864** (val)    |

*Trend:* Early overfitting moderated via regularization/tuning; validation accuracy improved.

> Learning-curve plots (loss/accuracy) for the key runs are included in the notebook.

# Future Work

- Transfer learning (ResNet/EfficientNet) to boost accuracy on small patches.

- Additional regularization/schedulers: label smoothing, cosine LR decay, early stopping; try mixup/cutout.

- Patient-level cross-validation for tighter generalization estimates.

- Lightweight demo (Streamlit) or inference API (Flask/FastAPI).

- Experiment tracking with MLflow or Weights & Biases.