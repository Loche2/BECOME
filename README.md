# BECOME

Code repository of the CIKM '26 paper "Budget and Frequency Controlled Cost-Aware Model Extraction Attack on Sequential Recommenders"

## Introduction

<img src=visualize/framework.jpg>

In this paper, we propose a novel approach, named Budget and Frequency Controlled Cost-Aware Model Extraction Attack (BECOME), for extracting black-box sequential recommenders, which extends the standard extraction framework with two cost-aware innovations: Feedback-Driven Dynamic Budgeting period-ically evaluates the victim model to refine query allocation and steer sequence generation adaptively. Rank-Aware Frequency Controlling integrates frequency constraints with ranking guidance in the next-item sampler to select high-value items and broaden information coverage.

## Requirements

see requirements.txt

## Usage
### 1. Train the Black-Box Recommender Model

```bash
python train.py
```

Trained black-box recommenders are stored under ./experiments/model-code/dataset-code/models/best_acc_model.pth

### 2. Extract trained Black-Box Recommender model

```bash
python distill.py
```

Extracted surrogate models and metrics are stored under ./experiments/distillation_rank/distillation-specification/dataset-code/

## Citing BECOME

If you find this repo useful for your research, pleae cite our paper.

```
@inproceedings{zhou2025become,
  title={Budget and Frequency Controlled Cost-Aware Model Extraction Attack on Sequential Recommenders},
  author={Lei Zhou, Min Gao, Zongwei Wang, and Yibing Bai},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM ’25)},
  year={2025}
  doi={10.1145/3746252.3761032}
}
```

## Acknowledgments

We thank the authors of [RecSys-Extraction-Attack](https://github.com/Yueeeeeeee/RecSys-Extraction-Attack) for making their code available, which forms the basis of our implementation. We also gratefully acknowledge the authors of [Transformers](https://github.com/huggingface/transformers) from Hugging Face and [BERT4Rec](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) by Jaewon Chung.
