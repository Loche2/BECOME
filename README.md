# BECOME
Implementation of the paper "Budget and Frequency Controlled Cost-Aware Model Extraction Attack on Sequential Recommenders"

## Requirements

see requirements.txt

## 1. Train the Black-Box Recommender Model

```bash
python train.py
```

Trained black-box recommenders are stored under ./experiments/model-code/dataset-code/models/best_acc_model.pth

## 2. Extract trained Black-Box Recommender model

```bash
python distill.py
```

Extracted surrogate models and metrics are stored under ./experiments/distillation_rank/distillation-specification/dataset-code/
