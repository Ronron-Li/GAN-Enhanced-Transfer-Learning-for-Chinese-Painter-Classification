# GAN-Enhanced-Transfer-Learning-for-Chinese-Painter-Classification
DS301 Project_NYU
# GAN-Enhanced Transfer Learning for Chinese Painter Classification

Fine-grained classification of Chinese paintings by painter, with **StyleGAN2-ADA-generated samples used as a data-augmentation strategy** to address the small-dataset problem inherent to traditional-art classification.

**Stack:** PyTorch · ResNet-50 · StyleGAN2-ADA · TensorFlow/Keras · WikiArt

> Course project for **DS-GA 3001 (Advanced Topics in Data Science: Deep Learning), NYU**, 2024.

---

## Motivation

Author-level classification of Chinese paintings is hard for two reasons:

1. **Per-painter data is scarce.** Even canonical masters have only tens to a few hundred extant attributed works in clean digital form — orders of magnitude less than what modern CNNs typically train on.
2. **Stylistic features are subtle.** Brushwork, ink density, and compositional conventions distinguish painters more than color or subject, so off-the-shelf ImageNet features transfer imperfectly.

This project tackles both: **transfer learning** from a Western-art-pretrained backbone to handle the domain shift, and **GAN-generated synthetic samples** to expand the per-class training set.

---

## Pipeline

```
WikiArt (Western art, multi-genre)
        │
        ▼
[Stage 1]  Pre-train ResNet-50 on WikiArt genres
        │  (general "art" feature extractor)
        ▼
Chinese painters dataset (10 painters)
        │
        ├──▶ [Stage 2a] Transfer learning
        │     - Shallow: freeze backbone, train FC head
        │     - Deep:    progressively unfreeze layers
        │
        └──▶ [Stage 2b] StyleGAN2-ADA augmentation
              - Fine-tune FFHQ-256 checkpoint on Chinese paintings
              - Generate synthetic samples per painter
              - Mix synthetic + real, retrain classifier
```

---

## Repository contents

| Notebook | Purpose |
|---|---|
| `preprocessData.ipynb` | Build painter-labeled dataset; resize to 224×224, ImageNet-normalize, serialize as tensor `.pt` files |
| `Base_model_training.ipynb` | Pre-train ResNet-50 on WikiArt genres as the domain-adapted backbone |
| `Shallow_transfer_learning.ipynb` | Freeze backbone, train only the final FC layer on Chinese painters (baseline) |
| `trainingChineseArtOnRestnet50 (1).ipynb` | End-to-end fine-tune ResNet-50 on Chinese painters; LR scheduling, both PyTorch and TF/Keras variants |
| `GANcHineseART2.ipynb` | StyleGAN2-ADA setup (CUDA 11 / PyTorch 1.7.1), fine-tune FFHQ-256 checkpoint, generate synthetic samples |
| `GAN_shallow_learning.ipynb` | Sweep frozen-layer count, retrain on real + GAN-augmented data, report accuracy vs. freeze depth |

---

## Key implementation details

- **Backbone:** `torchvision.models.resnet50(pretrained=True)`, FC head replaced with `nn.Linear(2048, num_classes)`.
- **Input pipeline:** 224×224, ImageNet mean/std normalization, batch size 32.
- **Optimizer:** Adam, `lr=1e-3` for head-only training, `lr=1e-4` for full fine-tuning.
- **Loss:** `CrossEntropyLoss`.
- **Ablation:** trained with `freeze_layers ∈ {0, 1, 2, …, all}` to find the accuracy-optimal freeze depth.
- **GAN:** StyleGAN2-ADA-PyTorch (NVlabs), fine-tuned from `stylegan2-ffhq-256x256.pkl` to converge with limited Chinese-painting data.
- **Evaluation:** accuracy, precision, recall, F1 (sklearn); accuracy-vs-frozen-layers and accuracy-vs-augmentation-ratio plots.

---

## How to reproduce

The notebooks were written for Google Colab + Google Drive. To rerun:

1. Place the Chinese painters image folders under `Dataset/<painter_name>/*.jpg`.
2. Run `preprocessData.ipynb` to produce `image_tensors.zip` and `image_labels.csv`.
3. Run `Base_model_training.ipynb` to pre-train the WikiArt backbone.
4. Run `Shallow_transfer_learning.ipynb` / `trainingChineseArtOnRestnet50.ipynb` for the transfer-learning baselines.
5. Run `GANcHineseART2.ipynb` to fine-tune StyleGAN2-ADA and generate synthetic samples (requires a CUDA 11 GPU; T4 on Colab works).
6. Run `GAN_shallow_learning.ipynb` for the augmented-classifier sweep.

---

## Datasets

- **WikiArt** (via `kagglehub`: `steubk/wikiart`) — backbone pre-training.
- **Chinese painters dataset** — 10 painters, assembled and labeled by the team. Not redistributed in this repo due to source-image licensing.

---

## Team & credits

Course project for **DS-GA 3001 — NYU Center for Data Science** (2024).
StyleGAN2-ADA implementation: [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

---

## License

MIT — see [`LICENSE`](LICENSE).
