# Pre-trained models

The following table summarizes the performance of the pre-trained models and the
URLs of checkpoints and configurations. The scores are slightly different from
the paper due to refactoring of the code, but the trends are generally the same.
We set `--seed_everything 0` for both training and evaluation.

Name|Accuracy (RGB)|Accuracy (ALpha)|F-score (RGB)|F-score (Alpha)|URL|Note
---|---|---|---|---|---|---
Stats (mode)|0.717|0.891|0.003|0.219|[ckpt](https://storage.googleapis.com/ailab-public/webcolor/checkpoints/Stats.ckpt), [config](https://storage.googleapis.com/ailab-public/webcolor/configs/Stats.yaml)|Evaluate with ``--model.sampling false``.
Stats (sampling)|0.621|0.821|0.004|0.207|[ckpt](https://storage.googleapis.com/ailab-public/webcolor/checkpoints/Stats.ckpt), [config](https://storage.googleapis.com/ailab-public/webcolor/configs/Stats.yaml)|Evaluate with ``--model.sampling true``.
NAR|0.774|0.929|0.078|0.677|[ckpt](https://storage.googleapis.com/ailab-public/webcolor/checkpoints/NAR.ckpt), [config](https://storage.googleapis.com/ailab-public/webcolor/configs/NAR.yaml)|
CVAE|0.773|0.929|0.069|0.664|[ckpt](https://storage.googleapis.com/ailab-public/webcolor/checkpoints/CVAE.ckpt), [config](https://storage.googleapis.com/ailab-public/webcolor/configs/CVAE.yaml)|
Upsampler|-|-|-|-|[ckpt](), [config]()|Used together with all the other models.
