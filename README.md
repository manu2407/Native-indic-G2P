# Colab training package

This folder is a self-contained copy of the files required to train and
evaluate the neural Hindi G2P model on Google Colab. It intentionally excludes
the full research history, raw corpus downloads, local virtual environments,
and large neural checkpoints.

## Contents

- Python runtime: `neural_g2p.py` plus the native candidate pipeline it uses.
- Frozen training, development, and observed blind label files.
- The sealed manifest that verifies their SHA-256 digests and word-disjoint
  partitions.
- Compact native schwa and prosody artifacts required by `--input-mode native`.
- `native_indic_g2p_t4.ipynb`, a Colab notebook that installs PyTorch, checks
  the GPU, verifies the dataset, and starts a resumable T4 training run.

The included data is about 41 MiB. It is enough to reproduce the current
neural experiment. The observed blind set has already been used for one
reported result; do not tune model choices against it.

## Run in Google Colab

1. In Colab select **Runtime → Change runtime type → T4 GPU**.
2. Clone the repository and open `colab/native_indic_g2p_t4.ipynb`, or upload
   this `colab/` folder alone.
3. Run the notebook cells in order.
4. Download `models/neural_g2p_t4_v1.pt` before the Colab runtime ends.

The notebook uses 1,000 epochs as a maximum, not a target. Development-set
early stopping (`--patience 30`) keeps the best checkpoint automatically.
See [EXPERIMENT.md](EXPERIMENT.md) for the fixed inputs and run-recording
protocol.

## Important boundary

The neural data eligibility implementation currently allows any Unicode
letters/marks even though the manifest describes Devanagari-only data. The
included v1 package is frozen for reproducibility; do not silently edit it.
Create a cleaned v2 dataset and a fresh blind split before making final claims
about later models.
