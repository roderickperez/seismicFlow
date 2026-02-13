# seismicFlow

A repository for seismic flowline extraction and analysis. This project is a fork of [Adelved/seismic-flow](https://github.com/Adelved/seismic-flow).

## Project Overview

This project provides tools for processing 2D seismic data, specifically for:
- Extracting surfaces and horizons.
- Calculating structure tensors for gradient analysis.
- Segmenting flowlines using clustering techniques.

## Folder Structure

- `F3_2Dline.npy`: Sample 2D seismic data in NumPy format.
- `extract_surfaces.py`: Main script for surface extraction.
- `flow_utils.py`: Utility functions and `Surface` class definition.
- `generate_surfaces_2D.py`: Script for generating 2D surfaces.
- `segment_flowlines.py`: Script for clustering and segmenting flowlines.
- `requirements.txt`: List of required Python packages.
- `environment.yml`: Conda environment specification.
- `pyproject.toml`: `uv` project configuration.

## Requirements

The project requires Python 3.12 and several scientific libraries. 
- **Core**: `numpy`, `scipy`, `matplotlib`, `shapely`, `scikit-learn`, `structure-tensor`
- **GPU Acceleration**: `torch`, `torchvision`, `torchaudio` (for CUDA-enabled tasks)

## Installation

### Using `uv` (Recommended)

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Initialize the environment and install dependencies**:
    ```bash
    uv sync
    ```
3.  **Verify GPU access**:
    ```bash
    uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

## Credits

This project is based on the work by **Adelved** in the original [seismic-flow](https://github.com/Adelved/seismic-flow) repository.
