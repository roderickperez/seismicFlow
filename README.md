# seismicFlow

A repository for seismic flowline extraction and analysis. This project is a fork of [Adelved/seismic-flow](https://github.com/Adelved/seismic-flow).

## Project Overview

This project provides tools for processing 2D seismic data, specifically for:
- Extracting surfaces and horizons.
- Calculating structure tensors for gradient analysis.
- Segmenting flowlines using clustering techniques.

## Scientific Background & Methodology

Based on "Exploring seismic data in the flowline domain" (Adelved et al., The Leading Edge, March 2025), the methodology treats seismic reflections as a fluid flow problem.

### A) Step-by-Step Methodology
1. **Calculate Dip and Azimuth (The Vector Field)**:
   - Uses the **Gradient Structure Tensor** to calculate the gradient vector of the seismic image.
   - Perform Eigen-decomposition to find eigenvalues and eigenvectors.
   - The eigenvector corresponding to the smallest eigenvalue points parallel to the reflections, forming the "velocity field" for flow simulation.

2. **Connect the Lines (Flowline Integration)**:
   - Seed points are placed at fixed increments across the seismic section.
   - The **fourth-order Runge-Kutta (RK4)** method is used to trace the path of particles through the vector field.
   - Resulting "flowlines" follow the path of minimum amplitude change.

3. **K-Means Clustering**:
   - Clusters flowlines based on geometric features (specifically $y$-coordinates).
   - This automatically separates the seismic section into distinct stratigraphic sequences.

4. **Surface Selection**:
   - Calculates an **Overlap Score**. Convergence/overlap heavily indicates unconformities (erosion surfaces).
   - Surfaces are ranked by this score for final selection.

### B) GPU Acceleration Strategies
- **Structure Tensor**: Parallelized using a convolutional approach or GPU kernels (CuPy/PyTorch) to compute gradients and Eigen-decomposition for the whole matrix simultaneously.
- **RK4 Integration**: An "embarrassingly parallel" problem where each flowline is independent. Multiple threads can trace thousands of lines simultaneously using GPU texture memory for spatial locality.
- **K-Means**: Massive GPU acceleration is possible using libraries like FAISS or RAPIDS (cuML).

### C) 3D Expansion
- **Pseudo-3D (2.5D)**: Traces high-quality lines on one slice, then uses those points as seeds for orthogonal tracing, creating "ribbons" to define surfaces.
- **Volumetric 3D**:
   - Compute a 3x3 Structure Tensor for every voxel.
   - The smallest eigenvector points in the direction of the reflection plane's dip.
   - RK4 becomes 3-dimensional ($dx, dy, dz$), propagating a "front" of points to define a 2D surface manifold.

## Folder Structure

- `F3_2Dline.npy`: Sample 2D seismic data in NumPy format.
- `extract_surfaces.py`: Main script for surface extraction.
- `flow_utils.py`: Utility functions and `Surface` class definition.
- `generate_surfaces_2D.py`: Script for generating 2D surfaces.
- `segment_flowlines.py`: Script for clustering and segmenting flowlines.
- `requirements.txt`: List of required Python packages.
- `pyproject.toml`: `uv` project configuration.

## Requirements

The project requires Python 3.12 and several scientific libraries. 
- **Core**: `numpy`, `scipy`, `matplotlib`, `shapely`, `scikit-learn`, `structure-tensor`
- **GPU Acceleration**: `torch`, `torchvision`, `torchaudio`, `cupy-cuda12x`

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

## Seismic Processing Workflow

The project follows a step-by-step pipeline for 2D seismic analysis:

### 1. Data Loading
- **Input**: 2D Seismic volume in `.npy` (NumPy) format.
- **Default File**: `F3_2Dline.npy` (located in the project root).
- **Structure**: Array of shape `[samples, traces]`.

### 2. Structural Analysis (Structure Tensor)
- The code calculates the **Structural Gradient Tensor** to identify local orientations (dip/azimuth) of seismic events.
- Tools like `structure_tensor_2d` and `eig_special_2d` are used to compute the vector field (gradient directions).
- **GPU Acceleration**: This step automatically uses **CuPy** (via NVIDIA GPU) for faster computation if `cupy` is installed.

### 3. Surface Extraction (Horizon Picking)
- Surfaces (horizons) are extracted by tracing through the computed vector field using the **Runge-Kutta (RK4)** integration method.
- Seed points are automatically selected at specified trace intervals.
- File: `extract_surfaces.py`

### 4. Selection and Pruning
- Extracted surfaces are filtered and pruned based on overlap thresholds to ensure only unique horizons are kept.
- File: `generate_surfaces_2D.py`

### 5. Segmentation (K-Means Clustering)
- Surfaces are grouped into stratigraphic layers using **K-Means clustering** based on their coordinates and features.
- A final segmentation map is produced showing the layered structure of the seismic section.
- File: `segment_flowlines.py`

---

## Technical Setup (GPU & Libraries)

### GPU Acceleration (NVIDIA A5000)
To ensure maximum performance on your NVIDIA A5000, we use:
- **PyTorch**: For future deep learning or tensor operations.
- **CuPy**: For high-speed structure tensor calculations.

### Maintenance & Updates

- **Activate Environment**: `source .venv/bin/activate`
- **Sync/Install All**: `uv sync`
- **Install New Library**: `uv add <library-name>`
- **Upgrade All Libraries**: `uv lock --upgrade && uv sync`
- **Update Specific Library**: `uv add <library-name> --upgrade`

> [!NOTE]
> **Visualization in Non-Interactive Environments**: The scripts are configured to automatically save results as `.png` images in the `output/` directory. This is useful for users running the code via SSH or WSL without a GUI.
> - **Horizon Extraction**: `output/extracted_surfaces.png`
> - **Surface Generation**: `output/generated_surfaces.png`
> - **Segmentation Map**: `output/segmentation_map.png`

## Detailed Instructions to Make it Work

1.  **Prepare Data**: Ensure you have a `.npy` file named `F3_2Dline.npy` (provided in the repo).
2.  **Activate Environment**: 
    ```bash
    source .venv/bin/activate
    ```
3.  **Execute**: Run any of the main scripts using `uv run python <script_name>.py`.
4.  **View Results**: Locate the generated images in the `output/` folder and open them using your preferred image viewer.

## Credits

This project is based on the work by **Adelved** in the original [seismic-flow](https://github.com/Adelved/seismic-flow) repository.
