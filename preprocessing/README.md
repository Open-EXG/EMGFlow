# Preprocessing

This folder contains the raw-data preprocessing scripts used to convert NinaPro source files into the structured `npy` or Hugging Face dataset formats used by this repository.

Included scripts:

- `process_db2.py`
- `process_db4.py`
- `process_db6.py`
- `process_db7.py`
- `read_db2.py`
- `read_db4.py`
- `read_db7.py`
- `verify_db4.py`
- `verify_db7.py`
- `verivy_db2.py`
- `verivy_db6.py`
- `download_db2.sh`
- `download_db6.sh`
- `DB7_README.md`

## Usage Notes

- These scripts were copied from the original internal preprocessing workspace.
- Paths are now configured through environment variables instead of hard-coded machine-specific paths.
- DB2, DB4, and DB7 processing scripts export Hugging Face dataset artifacts.
- DB6 processing exports structured `npy` files by subject/day/session.

## Environment Variables

Processing scripts:

- `EMGFLOW_DB2_ZIP_SOURCE_DIR`
- `EMGFLOW_DB2_TARGET_DIR`
- `EMGFLOW_DB2_DATASET_NAME`
- `EMGFLOW_DB4_ZIP_SOURCE_DIR`
- `EMGFLOW_DB4_TARGET_DIR`
- `EMGFLOW_DB4_DATASET_NAME`
- `EMGFLOW_DB6_SOURCE_DIR`
- `EMGFLOW_DB6_TARGET_DIR`
- `EMGFLOW_DB7_ZIP_SOURCE_DIR`
- `EMGFLOW_DB7_TARGET_DIR`
- `EMGFLOW_DB7_DATASET_NAME`

Read and verification scripts:

- `EMGFLOW_DB2_DATASET_DIR`
- `EMGFLOW_DB4_DATASET_DIR`
- `EMGFLOW_DB7_DATASET_DIR`

Download scripts:

- `EMGFLOW_DB2_DOWNLOAD_DIR`
- `EMGFLOW_DB6_DOWNLOAD_DIR`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

## Typical Workflow

1. Download or place the raw NinaPro files under your local raw-data directory.
2. Export the source and target environment variables for the dataset you want to process.
3. Run the processing script for the dataset you need.
4. Optionally run the paired `read_*` or `verify_*` script to inspect the generated outputs.

Example:

```bash
export EMGFLOW_DB4_ZIP_SOURCE_DIR=/path/to/Ninapro_DB4
export EMGFLOW_DB4_TARGET_DIR=/path/to/DB4_npy
export EMGFLOW_DB4_DATASET_DIR=/path/to/DB4_npy/emg_db4_dataset

python preprocessing/process_db4.py
python preprocessing/verify_db4.py
python preprocessing/read_db4.py
```
