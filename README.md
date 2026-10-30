# SER-CP

Core implementation of **Sample-Level Evidence Response and Complementary Prediction for Low-Quality Multi-View Classification**.

SER-CP combines class-wise source evidence correction, Dempster–Shafer aggregation, a joint prediction branch, and sample-adaptive posterior mixing. The paper configuration is `patient_posterior_natural` with hidden width 128. Historical configuration names are retained to preserve the frozen implementation.

## Files

- `dualpath_model.py`: fusion model and prediction interface.
- `dualpath_losses.py`: final NLL, auxiliary supervision, and reference-based safety loss.
- `dependencies/`: the two unchanged source modules used by the model. Their SHA-256 values are checked when imported.
- `tests/`: synthetic CPU tests for probabilities, masks, gradients, mixing, and losses.
- `SOURCE_SHA256.json`: release file checksums.

## Setup and test

Python 3.9 or later is required. Install PyTorch for your environment, then install the remaining dependencies.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests -p no:cacheprovider
```

The release was tested on CPU with Python 3.10.9, PyTorch 1.12.1, NumPy 1.22.4, scikit-learn 1.2.1, and pytest 7.1.2. All 40 tests passed.

## Inference interface

```python
import torch
from dualpath_model import build_model

model = build_model(
    'patient_posterior_natural', input_dim=512,
    views=4, classes=4, hidden=128,
).eval()

# Synthetic inputs demonstrate shapes, not a trained prediction.
features = torch.randn(2, 4, 512)        # batch, views, feature dimension
present = torch.ones(2, 4, dtype=torch.bool)
with torch.no_grad():
    output = model(features, present)
    probabilities = output['probabilities']
    joint_weight = output['gate']
```

`features` must be finite floating-point tensors with a common feature dimension across view slots. `present` indicates available views and must retain at least one view per sample. Preprocessing and feature extraction are external to the fusion head. Load a compatible trained state dictionary for real predictions.

The final probability is `(1 - gate) * p_DS + gate * p_joint`. The output also includes source evidence, branch evidence, and four-state diagnostics. Inference does not take labels or corruption identities.

## Training interface

Use `dualpath_losses.compute_loss` with integer class labels, an availability mask, and frozen reference probabilities. The paper defaults are auxiliary weight 0.3 and safety weight 0.5. During paired conflict training, `aux_present` excludes replaced views from isolated source supervision. Reference predictions must be prepared without evaluation-patient leakage and detached from the optimization graph.

This compact release contains the model and loss, not the complete dataset-specific preprocessing, reference-model assembly, training queues, or evaluation pipeline. The historical command-line entry point inside `dependencies/train_legacy256_paired.py` is not the SER-CP experiment runner. Only its model definition is imported here.

## Scope

No datasets, patient identifiers, feature arrays, checkpoints, experimental predictions, server configuration, or credentials are included. All tests use generated tensors. The frozen model and loss files are copied unchanged from the experimental implementation.
