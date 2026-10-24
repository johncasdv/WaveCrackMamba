# Training and evaluation

The scripts in this directory implement the experiment configurations used for DeepCrack and Crack500. See the repository-root `README.md` for the tested environment, dataset layout, training command, and evaluation protocol.

`run_pair.py` runs one dataset. `run_all.py` runs both datasets sequentially on one GPU. `evaluate_best.py` evaluates a completed run's validation-selected checkpoint on the held-out test set. Dataset paths and run output paths are supplied at runtime; benchmark data are not included.
