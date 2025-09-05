#!/bin/bash

# This script sets up the environment for running BioEMu

# Enable ColabFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Change venv directory to the current conda environment
export BIOEMU_COLABFOLD_DIR=$CONDA_PREFIX

# Git clone the required repositories
# git clone https://github.com/microsoft/bioemu.git
# git clone https://github.com/microsoft/bioemu-benchmarks.git

# Create a conda environment for BioEMu
# conda create -n se3diff python=3.12 --yes
# conda activate se3diff

pip install uv

# Install required packages with correct versions
uv pip install -e .
uv pip install dm-haiku --upgrade
uv pip install numpy --upgrade
uv pip install pandas --upgrade

# Patch ColabFold
SITE_PACKAGES_DIR=$(python -c "import site; print(next(p for p in site.getsitepackages() if 'site-packages' in p))")
SCRIPT_DIR=$(pwd)/bioemu/src/bioemu/colabfold_setup
patch "${SITE_PACKAGES_DIR}/alphafold/model/modules.py" "${SCRIPT_DIR}/modules.patch"
patch "${SITE_PACKAGES_DIR}/colabfold/batch.py" "${SCRIPT_DIR}/batch.patch"
touch "${CONDA_PREFIX}"/.COLABFOLD_PATCHED

# Install BioEmu and BioEmu Benchmarks
uv pip install -e bioemu
uv pip install -e bioemu-benchmarks
