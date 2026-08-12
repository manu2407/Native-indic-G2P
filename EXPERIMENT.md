# T4 experiment protocol

## Fixed v1 inputs

The included manifest fixes 64,676 training words and 3,357 development words.
It also includes the 3,215-word blind split used for the already reported v1
result. Verify the manifest before training:

```bash
python neural_data.py
```

Do not use the included blind split to choose hyperparameters, epochs, or
architectures. Use it only to reproduce the existing result. A cleaned v2
dataset needs a new blind split.

## First T4 run

Run the notebook's medium configuration:

```bash
python neural_g2p.py train \
  --device cuda \
  --output models/neural_g2p_t4_v1.pt \
  --epochs 1000 \
  --patience 30 \
  --learning-rate 2e-4 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --d-model 256 \
  --heads 8 \
  --layers 4 \
  --feedforward 1024 \
  --dropout 0.1 \
  --input-mode native
```

The checkpoint is rewritten only when development full-schema exactness
improves. Download it from Colab after the run; `models/*.pt` is ignored so a
large model is never accidentally committed.

## Record after every run

Save the command, the best epoch, development metrics, checkpoint filename,
and Git commit in the GitHub issue or release notes. This is sufficient to
compare experiments without adding a custom experiment-tracking system.
