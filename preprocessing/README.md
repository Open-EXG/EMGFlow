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
- Most scripts use absolute local paths at the top of the file.
- Before running them, update the path configuration in each script to match your machine.
- DB2, DB4, and DB7 processing scripts export Hugging Face dataset artifacts.
- DB6 processing exports structured `npy` files by subject/day/session.

## Typical Workflow

1. Download or place the raw NinaPro files under your local raw-data directory.
2. Edit the source and target path constants in the corresponding `process_db*.py` script.
3. Run the processing script for the dataset you need.
4. Optionally run the paired `read_*` or `verify_*` script to inspect the generated outputs.
