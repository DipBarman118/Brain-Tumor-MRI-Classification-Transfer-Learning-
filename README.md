# Brain Tumor MRI Classification (EfficientNetB3)

A deep learning pipeline that classifies brain MRI scans into four categories —
**glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using a
fine-tuned EfficientNetB3 backbone. The final model reaches **95.50% accuracy**
on a held-out test set of 1,311 images.

## Problem Statement

Manual review of brain MRI scans for tumor detection and subtyping is
time-consuming and requires radiological expertise that isn't always available,
especially in resource-constrained settings. This project builds an automated
classifier that takes a single MRI slice and predicts one of four classes,
aiming to support (not replace) radiological screening by flagging likely tumor
type with a confidence score, including test-time augmentation (TTA) for more
robust single-image predictions.

## Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, by Masoud Nickparvar)
- **License:** CC0-1.0 (Public Domain)
- **Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`
- **Split:** 5,712 images in `Training/` (further split 85/15 into train/val),
  1,311 images in `Testing/`

If you use this dataset, please credit the original Kaggle source above.

## Architecture

- **Backbone:** EfficientNetB3 (ImageNet-pretrained, `include_top=False`)
- **Input size:** 300×300×3 (EfficientNetB3's native resolution)
- **Head:** GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) →
  Dropout(0.3) → Dense(4, softmax)
- **Augmentation:** random horizontal flip, rotation (±0.2), zoom (±0.2),
  translation (±0.1), contrast (±0.2) — applied in-graph via a Keras
  `Sequential` augmentation block
- **Class imbalance:** handled via computed class weights (inverse frequency)
  rather than resampling

### Two-Stage Training

1. **Stage 1 — Frozen backbone:** EfficientNetB3 weights frozen, only the
   classification head trained. Adam @ 1e-4, up to 20 epochs, early stopping
   on `val_loss` (patience 7).
2. **Stage 2 — Fine-tuning:** last 100 layers of the backbone unfrozen,
   trained end-to-end. Adam @ 1e-5, up to 25 epochs, early stopping on
   `val_loss` (patience 10).

Both stages use `ModelCheckpoint` (best `val_loss`) and `ReduceLROnPlateau`.

### Inference

Single-image prediction uses **test-time augmentation**: the input image plus
a horizontal flip and two rotations are all run through the model, and the
softmax outputs are averaged. Predictions below 60% confidence are flagged as
`(uncertain)`.

## Results (Test Set, n=1,311)

**Overall accuracy: 95.50%**

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| glioma       | 0.9532    | 0.9500 | 0.9516   |
| meningioma   | 0.9479    | 0.8922 | 0.9192   |
| notumor      | 0.9949    | 0.9728 | 0.9838   |
| pituitary    | 0.9146    | 1.0000 | 0.9554   |

Full classification report and confusion matrix: [`results/`](results/)

The main confusion is meningioma being misclassified as pituitary (23 cases) —
see `results/classification_report.md` for a short error analysis.

## Repository Structure

```
.
├── ML_Project.ipynb              # Full training + evaluation notebook (Colab)
├── class_names.json              # Class label order used by the model
├── requirements.txt              # Python dependencies
├── results/
│   ├── confusion_matrix.png      # Test-set confusion matrix
│   └── classification_report.md  # Full precision/recall/F1 breakdown
├── LICENSE
└── README.md
```

## Running the Notebook

1. Open `ML_Project.ipynb` in Google Colab (ideally with a GPU runtime).
2. You'll need a Kaggle API token (`kaggle.json`) to download the dataset —
   generate one from your [Kaggle account settings](https://www.kaggle.com/settings)
   and upload it when prompted in Cell 2.
   **Never commit `kaggle.json` to this repo** — it's a personal credential.
3. Run cells in order. Training (both stages) takes roughly 30–40 minutes on a
   T4 GPU.
4. The trained model and `class_names.json` are saved to `/content/` at the end
   of the notebook; the final cell lets you upload a new image for prediction.

## Notes on Reproducing Locally

If running outside Colab, install dependencies with:

```bash
pip install -r requirements.txt
```

Versions in `requirements.txt` are reasonable compatible ranges, not the exact
frozen versions from the original Colab run (those weren't logged in the
notebook output).

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The dataset itself is CC0-1.0 and is **not** redistributed here — download it
directly from the Kaggle link above.
