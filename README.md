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

## Code-to-Theory Mapping

The following table maps the mathematical steps from the paper to the specific implementations in this repository:

| Methodology Step (Paper) | Implementation File | Key Functions/Logic |
| :--- | :--- | :--- |
| **1. Structure Tensor** | `extract_surfaces.py` | `structure_tensor_2d`, `eig_special_2d` (Lines 25-26) |
| **2. RK4 Integration** | `flow_utils.py` | `runge_kutta_4` (Lines 58-91) |
| **3. Flowline Extraction** | `flow_utils.py` | `extract_surfaces` (Lines 94-181) |
| **4. Unconformity Heatmap** | `generate_surfaces_2D.py`| `create_heatmap` (Lines 58-76) |
| **5. Overlap Pruning** | `generate_surfaces_2D.py`| `prune_overlapping_surfaces` (Lines 35-53) |
| **6. K-Means Clustering** | `segment_flowlines.py` | `segment_flowlines`, `KMeans` (Lines 111-174) |

---

## Folder Structure

- `F3_2Dline.npy`: Sample 2D seismic data in NumPy format.
- `extract_surfaces.py`: Main script for surface extraction.
- `flow_utils.py`: Utility functions and `Surface` class definition.
- `generate_surfaces_2D.py`: Script for generating 2D surfaces.
- `segment_flowlines.py`: Script for clustering and segmenting flowlines.
- `requirements.txt`: List of required Python packages.
- `pyproject.toml`: `uv` project configuration.

## Performance Analysis & Bottlenecks

This project leverages GPU acceleration (CuPy) to handle the heavy computational load of seismic attribute analysis. Below is a breakdown of the performance characteristics for each workflow.

### 1. Structure Tensor (ST) Computation
*   **2D / 2.5D**: Calculations involve a $2 \times 2$ gradient matrix per pixel. This is extremely fast on GPU as it relies on simple 2D convolutions and element-wise math.
*   **3D**: Calculations involve a $3 \times 3$ matrix per voxel (6 unique components). 
    *   **The Hard Part**: 3D ST requires **3D Gaussian Convolutions**, which are computationally much heavier than 2D ($O(N^3)$ vs $O(N^2)$).
    *   **The Bottleneck**: The Eigen-decomposition of millions of $3 \times 3$ matrices requires a robust numerical solver (`cp.linalg.eigh`). To prevent GPU solver crashes on large volumes, the code uses **batched processing** (1M samples per batch), which adds management overhead but ensures stability.

### 2. Pseudo-3D Flowline Generation (RK4)
*   **Mechanism**: The 3D script iterates through inlines and performs 2D RK4 tracing on each slice using the vectors projected from the 3D field.
*   **The Primary Bottleneck**: 
    1.  **Python Loop Overhead**: Iterating over hundreds of slices in Python introduces significant latency compared to a single monolithic GPU kernel.
    2.  **Kernel Launch Latency**: For every slice, multiple GPU kernels (vectorization, normalization, RK4 steps) are launched. The accumulation of these overheads is the primary reason large 3D volumes (e.g., >200 inlines) take 20+ minutes.
    3.  **CPU-Bound K-Means**: The segmentation step (K-Means) runs on the **CPU** using SciKit-Learn for every slice. This serial operation prevents the GPU from being fully utilized throughout the loop.

### 3. I/O and Memory
*   **Memory**: The 3D workflow loads the entire sub-volume into a 3D NumPy array. For massive volumes, this can lead to memory exhaustion or swapping.
*   **I/O**: SEGY writing is sequential. Exporting 4 technical volumes for a large 3D block involves significant disk I/O, which is a fixed hardware bottleneck.

### Summary of Computational Cost
| Operation | Complexity | Device | Note |
| :--- | :--- | :--- | :--- |
| **3D Structure Tensor** | High | GPU | $O(N^3)$ 3D Convolutions |
| **3D Eigen-Decomp** | High | GPU | Numerical solver, batched |
| **RK4 Tracing** | Medium | GPU | Limited by step-by-step iteration |
| **K-Means Seg** | Medium | **CPU** | **Major sequential bottleneck** |

---

## 🛠️ Requirements & Installation

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

### GPU Optimization & Benchmarking

The latest release introduces a high-performance GPU pipeline in `seismicFlow2D_interactive_GPU.py`.

#### Key Improvements:
1.  **Vectorized RK4 Engine**: Replaced sequential trace-by-trace iteration with a massively parallel GPU engine. This allows thousands of flowlines to be traced simultaneously.
2.  **Native Cupy Structure Tensor**: Implemented custom GPU kernels for Gaussian smoothing, gradient calculation, and eigenvalue decomposition, eliminating CPU-GPU transfer overhead during the initial structural analysis.
3.  **Real-time Benchmarking**: The interactive CLI now reports compute times per device.

#### Speedup Quantification (Tesla T4 vs. Modern CPU):
| Task | CPU Time | GPU Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Structure Tensor** | ~0.5 - 1.2s | ~0.08 - 0.15s | **~8x** |
| **Flowline Tracing (RK4)** | ~5.0 - 15s | ~0.02 - 0.05s | **~250x** |
| **End-to-End Slice** | ~10 - 20s | ~1.5 - 2.0s | **~10x** |

> [!TIP]
> The performance gain in RK4 is most noticeable when using smaller trace intervals (e.g., `interval=10`), where the GPU can process 500+ flowlines in the time a CPU takes for one.

## Technical Setup (GPU & Libraries)

### GPU Acceleration (NVIDIA A5000)
To ensure maximum performance on your NVIDIA A5000, we use:
- **PyTorch**: For future deep learning or tensor operations.
- **CuPy**: For high-speed structure tensor calculations.

## Clear Step-By-Step Execution Guide

Follow these steps to run the full seismic flowline processing pipeline.

### Step 1: Environment Activation
Always start by activating your `uv` environment:
```bash
source .venv/bin/activate
```

### Step 2: Prepare Your Data
- Ensure your seismic data is in `.npy` format.
- The default dataset `F3_2Dline.npy` is already in the root directory.
- To use your own data, place it in the root and update the `seispath` variable in the scripts.

### Step 3: Run the Processing Pipeline

#### **A. Basic Horizon Extraction (RK4 Integration)**
Run this script to calculate the structure tensor and trace horizons using the Runge-Kutta 4th order method.
```bash
uv run python extract_surfaces.py
```
*   **Result**: Generates `output/extracted_surfaces.png` showing the traced horizons on the seismic section.

#### **B. Full Segmentation & Clustering (K-Means)**
Run this to group horizons into stratigraphic layers and identify major unconformities based on flowline convergence.
```bash
uv run python segment_flowlines.py
```
*   **Result**: Generates `output/segmentation_map.png` showing the segmented stratigraphic sequences.

#### **C. Advanced Generation & Pruning**
Use this for high-quality surface generation with overlap-based pruning.
```bash
uv run python generate_surfaces_2D.py
```
*   **Result**: Generates `output/generated_surfaces.png`.

---

## Technical Maintenance

- **Update Dependencies**: `uv sync`
- **Add New Package**: `uv add <package>`
- **Upgrade All Packages**: `uv lock --upgrade && uv sync`

## Credits

This project is a fork of [Adelved/seismic-flow](https://github.com/Adelved/seismic-flow) and implements the methodology described in:
> **Exploring seismic data in the flowline domain** (Adelved et al., The Leading Edge, March 2025)
