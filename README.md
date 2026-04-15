# EMGFlow

This open-source branch is aligned with the paper `EMGFlow: Robust and Efficient Surface Electromyography Synthesis via Flow Matching`.

Following the paper scope, the public release keeps three core parts:

- `emgflow/model/`: conditional generative models for sEMG synthesis used in the paper
- `emgflow/datasets/`: Ninapro dataset loading and preprocessing utilities
- `EMG_fidelity/`: fidelity and downstream-evaluation package used to assess synthetic EMG quality
- `preprocessing/`: raw NinaPro preprocessing scripts for converting source files into repository-ready formats

Experiment-specific runners, ablation workspaces, temporary outputs, figures, and unrelated playground code are intentionally excluded from this branch.

## Scope

The released code corresponds to the paper's core pipeline:

1. Sliding-window sEMG preprocessing and dataset loading
2. Conditional EMG generation with Flow Matching and baseline generators
3. Evaluation through feature-based fidelity metrics and downstream utility

The public model implementations are limited to the paper's main generative baselines:

- EMGFlow / Flow Matching
- DDPM with DDIM or ancestral DDPM sampling
- WGAN-GP

## Environment

The original development environment used Python 3.12.

```bash
conda activate py312
pip install -e .
```

Main dependencies:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `PyWavelets`
- `einops`

## Dataset Roots

Dataset roots are configured with environment variables:

```bash
export NINAPRO_DB2_ROOT=/path/to/DB2_npy
export NINAPRO_DB4_ROOT=/path/to/DB4_npy
export NINAPRO_DB6_ROOT=/path/to/DB6_npy
export NINAPRO_DB7_ROOT=/path/to/DB7_npy
```

## Repository Layout

```text
emgflow/
  model/
  datasets/
  config/path.py
EMG_fidelity/
preprocessing/
```

## Notes

- This branch is a core-code release, not the full internal research workspace.
- The paper mainly benchmarks three datasets; the public loaders kept here cover the repository's core Ninapro loaders.
- The `preprocessing/` scripts are provided as utility scripts and may require local path edits before use.
- Pretrained checkpoints, cached outputs, and experiment artifacts are not included.

## License

License selection has not been finalized yet.
