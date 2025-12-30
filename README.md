## Multi-Modal Fake News Detection with Trust-Aware Triage

This repository contains the full implementation of a multimodal fake news detection system that integrates textual, visual, and external evidential signals with trust-aware uncertainty scoring and cost-aware human verification triage.

The codebase is designed for research reproducibility, ablation studies, and real-world deployment experiments.

### Key Contributions

Multimodal classifier combining text + image encoders with hierarchical fusion

Event-adversarial training to reduce topic leakage

Hyperlink trust modeling via Domain Trust Factor (DTF)

External evidence retrieval:

Reverse image search

Text retrieval

Optional verifier model

Three decision signals:

PTS (Preliminary Trustworthiness Score)

CMC (Combined Metric of Classification)

CMPU (Combined Metric of Prediction Uncertainty)

Cost-aware triage for selective human verification

This repo supports end-to-end training, evaluation, inference, and ablations.

### Repository Structure
.
├── configs/                # YAML experiment configs
├── data/                   # ignored by git (datasets, images, indices)
├── scripts/                # dataset preparation & utilities
├── src/mmfnd/              # core library
│   ├── cli/                # train / eval / predict CLIs
│   ├── datasets/           # CSV + image loaders
│   ├── models/             # text, vision, fusion, adversary
│   ├── evidence/           # retrieval, verifier, caching
│   ├── trust/              # DTF computation
│   └── scoring/            # PTS, CMC, CMPU, triage
├── outputs/                # ignored (runs, checkpoints, reports)
├── .gitignore
├── pyproject.toml
└── README.md

### Citation

If you use this codebase, cite the corresponding paper:
```
@article{yourpaper2025,
  title={Multimodal Fake News Detection with Trust-Aware Triage},
  author={...},
  year={2025}
}
```

