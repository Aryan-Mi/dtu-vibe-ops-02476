# DTU Course: MLOps 02476 (Winter 2026)

## Group (36) Members

- Aryan Mirzazadeh
- Mohamad Marwan Summakieh
- Trinity Sara McConnachie Evans
- Vladyslav Horbatenko
- Yuen Yi Hui

---

## Project goal
The goal of this project is to build an end-to-end, reproducible MLOps pipeline that can **classify dermatoscopic images of skin lesions** and estimate whether a lesion is **likely malignant vs. benign**. The primary focus is not “the biggest model”, but demonstrating a well-engineered workflow: data handling, training, evaluation, tracking, testing, packaging, and deployment-ready inference.

## Data (initial, may change)
We will start with the **HAM10000** dataset (Harvard Dataverse), a widely used benchmark of dermatoscopic images. The original labels are multi-class (seven lesion types); we will initially map them into a **binary target (malignant vs. benign)** for a simpler first version, and optionally expand to **multi-class** once the pipeline is configured and stable.

Key data steps:
- Deterministic train/val/test splits (seeded)
- Standardized preprocessing (resize, normalization)
- Augmentation for robustness (flip/rotate/color jitter, etc.)
- Handling class imbalance (weighted loss and/or sampling)

## Frameworks & how we include them
- **PyTorch** as the core training backend
- **PyTorch Lightning** to structure training loops, logging, checkpoints, and reproducibility
- **MONAI and/or torchvision** for medical-imaging friendly transforms + strong vision utilities
- **Hugging Face ecosystem** (where helpful) to pull pretrained vision backbones and standardize model configs
- **Streamlit** as a lightweight UI front-end to upload an image and get a predicted class + confidence output.

## Model plan (architecture brainstorm)
We will implement 3 tiers:
1. **Baseline CNN (from scratch)**
   Repeated blocks: Conv(3×3) → BatchNorm → ReLU → MaxPool(2), then a classifier head.
2. **Residual CNN (ResNet-style)**
   Residual blocks: Conv(3×3) → BN → ReLU → Conv(3×3) → BN, skip connection + ReLU, then global average pooling + FC.
3. **Transfer learning (expected best performance)**
   EfficientNet-family and/or modern backbones (e.g., ConvNeXt / ViT equivalents) initialized from pretrained weights, then fine-tuned on HAM10000.

## MLOps plan (course-aligned)
We will operationalize the pipeline using:
- A clean repository structure + reusable modules (data, models, training, inference)
- Reproducible environments (pinned dependencies) and containerization (Docker)
- Experiment tracking (metrics, configs, artifacts), plus model checkpointing
- Automated tests (data loading, shape checks, smoke-train step) and **CI** via GitHub Actions
- A simple deployment story: local inference service + Streamlit UI (and optional cloud deployment if time permits)

## Expected deliverables
- Training pipeline with tracked experiments and reproducible runs
- Evaluations (accuracy, F1, AUROC, confusion matrix) + error analysis
- Streamlit demo for interactive inference
- Documentation in this README + clear “how to run” instructions
