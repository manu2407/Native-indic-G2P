# Native Indic G2P — Colab training

This repository is the minimal Google Colab training package for the Hindi
hybrid G2P model. The CPU data-building work is already complete.

## Included

- `train_r2_t4.ipynb`: the only notebook needed for training.
- `data/hindi_g2p_v2_1m_r2.tar.gz`: the sealed r2 dataset bundle.
- Neural and native runtime code.
- Compact native schwa and prosody models.

The dataset contains:

- 701,549 training words
- 39,033 development words
- 38,829 blind words
- zero word overlap between partitions

The compressed dataset SHA-256 is:

```text
f66de01174569ed7b9ed59ff6d150f3d766017980e4de1c0a54b3b0987641d17
```

## Train

1. Open [`train_r2_t4.ipynb`](train_r2_t4.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run every cell in order.
4. Approve Google Drive mounting when prompted.

The notebook clones this repository, verifies and extracts the sealed dataset,
checks the train/dev/blind manifest, and trains a 256-dimensional four-layer
Transformer. The best development checkpoint is written directly to:

```text
MyDrive/native-indic-g2p/models/neural_g2p_r2_t4.pt
```

Training uses `--max-length 384` because the longest observed hybrid input is
353 symbols. The 1,000 epoch value is a ceiling; early stopping uses patience
30 and retains the best development checkpoint. If Colab disconnects, rerun
the notebook and it resumes from the checkpoint stored in Drive.

The blind split is included for one final evaluation after model choices are
settled. Do not use blind results to tune the architecture or hyperparameters.
