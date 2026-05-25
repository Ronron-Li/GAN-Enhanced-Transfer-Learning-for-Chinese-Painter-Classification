# GAN-Enhanced Transfer Learning for Chinese Painter Classification

Fine-grained classification of Chinese paintings by painter, using **StyleGAN2-ADA-generated samples as data augmentation** to address the small-dataset problem inherent to traditional-art classification.

**Final result: 43.1% → 85.9% test accuracy** (+42.8 pp over the naive baseline) on a 25-class painter-classification task with only ~400 original images.

**Stack:** PyTorch · ResNet-50 · StyleGAN2-ADA · TensorFlow/Keras · WikiArt

> Course project for **DS-UA 301 (Advanced Topics in Data Science), NYU**, 2024. Team: Ron Li, Fei Gao, Zhi Zeng.

---

## Results

| Method | Test Accuracy |
|---|---:|
| Naive ResNet-50 (no transfer, no GAN) | 43.1% |
| Fine-tuning from WikiArt-pretrained (no GAN) | 55.0% |
| Shallow Learning from WikiArt-pretrained (no GAN) | 56.2% |
| **Shallow Learning + GAN augmentation (freeze 43 layers)** | **85.9%** |

On the held-out original test set the GAN-augmented model classified **8/9 sampled images correctly** and reached **85.85% accuracy on the full original small dataset**.

---

## Motivation

Author-level classification of Chinese paintings is hard for two reasons:

1. **Per-painter data is scarce.** Our curated dataset has only ~400 paintings across 25 artists — roughly 13 training and 3 test images per class. Standard CNNs overfit immediately at this scale.
2. **Stylistic features are subtle.** Brushwork, ink density, and compositional conventions distinguish painters more than color or subject, so off-the-shelf ImageNet features transfer imperfectly.

This project tackles both: **transfer learning** from a Western-art-pretrained backbone to handle the domain shift, and **GAN-generated synthetic samples** to expand the per-class training set by ~20×.

---

## Pipeline

```
WikiArt (81k+ paintings, 27 genres)
        │
        ▼
[Stage 1]  Pre-train ResNet-50 on WikiArt genre classification
        │  (early-stopped at epoch 6 to prevent overfit)
        │  → domain-adapted "art" feature extractor
        ▼
Chinese paintings dataset (25 artists, ~400 images)
        │
        ├──▶ [Stage 2a] Transfer-learning baselines
        │     - Fine-tune entire network with LR schedule  → 55.0%
        │     - Shallow learning (freeze backbone, train FC only)  → 56.2%
        │
        └──▶ [Stage 2b] StyleGAN2-ADA data augmentation
              - Fine-tune from FFHQ-256 checkpoint on Chinese paintings
              - Train ~100k images per per-artist model
              - Generate ~300 synthetic samples per artist
              - Combined dataset: 312 real + 6,863 synthetic = 7,175 samples
              - Sweep freeze depth ∈ {10, 13, 16, …, 49} layers
              - Best: freeze first 43 layers → 90.15% on the augmented test split,
                85.9% on the original held-out small test set
```

---

## Key implementation details

- **Backbone:** `torchvision.models.resnet50(pretrained=True)`, FC head replaced with `nn.Linear(2048, num_classes)`.
- **Input pipeline:** images resized to 224×224, ImageNet mean/std normalization, batch size 32, serialized as `.pt` tensors for fast Colab loading.
- **Pre-training:** WikiArt genre classification, Adam `lr=1e-4`, early stopping at epoch 6 (val accuracy ≈ 71%, training diverges past this point).
- **Fine-tuning baseline:** custom step-LR schedule (`0.01 → 0.001 → 1e-4 → 1e-5` at epochs 5/10/15), Adam, 20 epochs.
- **Shallow-learning baseline:** freeze all backbone layers, optimize only `model.fc.parameters()`, Adam `lr=1e-3`, 15 epochs.
- **Ablation:** swept `freeze_layers ∈ {10, 13, 16, 19, …, 49}`; freezing the first 43 layers was optimal.
- **GAN:** StyleGAN2-ADA-PyTorch (NVlabs), per-artist generators fine-tuned from `stylegan2-ffhq-256x256.pkl`. CUDA 11.0 + PyTorch 1.7.1 environment built via conda inside Colab.
- **Evaluation:** accuracy, precision, recall, F1 (sklearn); accuracy-vs-frozen-layers and accuracy-vs-augmentation-ratio plots.

---

## Repository contents

| Notebook | Purpose |
|---|---|
| `preprocessData.ipynb` | Build painter-labeled dataset; resize to 224×224, ImageNet-normalize, serialize as `.pt` tensors |
| `Base_model_training.ipynb` | Pre-train ResNet-50 on WikiArt genres as the domain-adapted backbone |
| `Shallow_transfer_learning.ipynb` | Freeze backbone, train only the final FC layer on Chinese painters (baseline → 56.2%) |
| `trainingChineseArtOnRestnet50 (1).ipynb` | End-to-end fine-tune ResNet-50 on Chinese painters with LR scheduling; PyTorch and TF/Keras variants (→ 55.0%) |
| `GANcHineseART2.ipynb` | StyleGAN2-ADA setup (CUDA 11 / PyTorch 1.7.1), fine-tune FFHQ-256 checkpoint, generate synthetic samples |
| `GAN_shallow_learning.ipynb` | Sweep frozen-layer count over 7,175-sample augmented dataset (→ best: freeze 43 layers, **85.9%**) |

---

## How to reproduce

The notebooks were written for Google Colab + Google Drive. To rerun:

1. Place Chinese painters image folders under `Dataset/<painter_name>/*.jpg`.
2. Run `preprocessData.ipynb` to produce `image_tensors.zip` and `image_labels.csv`.
3. Run `Base_model_training.ipynb` to pre-train the WikiArt backbone.
4. Run `Shallow_transfer_learning.ipynb` / `trainingChineseArtOnRestnet50.ipynb` for the transfer-learning baselines.
5. Run `GANcHineseART2.ipynb` to fine-tune StyleGAN2-ADA and generate synthetic samples (requires a CUDA 11 GPU; Colab T4 works).
6. Run `GAN_shallow_learning.ipynb` for the augmented-classifier freeze-depth sweep.

---

## Datasets

- **WikiArt** (via `kagglehub`: `steubk/wikiart`) — 81,000+ paintings, 27 genres, used for backbone pre-training.
- **Chinese painters dataset** — 25 artists, ~400 paintings, assembled and labeled by the team. Not redistributed in this repo due to source-image licensing.

---

## Next steps

- Per-layer learning-rate tuning instead of a uniform schedule.
- Robustness testing on web-scraped images with different lighting and backgrounds.
- Compare StyleGAN2-ADA augmentation against diffusion-model augmentation.

---

## Credits

Course project for **DS-UA 301 — NYU Center for Data Science** (2024).
Team: Ron Li, Fei Gao, Zhi Zeng.
StyleGAN2-ADA implementation: [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

---

## License

MIT — see [`LICENSE`](LICENSE).
