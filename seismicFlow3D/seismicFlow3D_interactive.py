import os
import sys
import time
import numpy as np
import segyio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# Add parent directory to path to import flow_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_utils import *

console = Console()

def print_rich_banner():
    banner_text = """[bold cyan]
 ███████╗ ███████╗ ██╗ ███████╗ ███╗   ███╗ ██╗  ██████╗ ███████╗ ██╗      ██████╗  ██╗    ██╗
 ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗ ████║ ██║ ██╔════╝ ██╔════╝ ██║     ██╔═══██╗ ██║    ██║
 ███████╗ █████╗   ██║ ███████╗ ██╔████╔██║ ██║ ██║      █████╗   ██║     ██║   ██║ ██║ █╗ ██║
 ╚════██║ ██╔══╝   ██║ ╚════██║ ██║╚██╔╝██║ ██║ ██║      ██╔══╝   ██║     ██║   ██║ ██║███╗██║
 ███████║ ███████╗ ██║ ███████║ ██║ ╚═╝ ██║ ██║ ╚██████╗ ██║      ███████╗╚██████╔╝ ╚███╔███╔╝
 ╚══════╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝     ╚═╝ ╚═╝  ╚═════╝ ╚═╝      ╚══════╝ ╚═════╝   ╚══╝╚══╝ 

                           >>>>>>>>> [ 3D CASE ] <<<<<<<<<<<< 
    [/bold cyan]"""
    console.print(Panel.fit(banner_text, border_style="cyan"))

def main():
    print_rich_banner()
    
    # 0. Device selection
    if HAS_CUPY:
        device_choice = Prompt.ask("Select processing device", choices=["cpu", "gpu"], default="gpu")
    else:
        device_choice = "cpu"
        console.print("[bold red]CuPy not found. Using CPU (Will be slow).[/bold red]")
        
    console.print(f"Using [bold magenta]{device_choice.upper()}[/bold magenta] for performance heavy operations")
    
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    segy_path = os.path.join(base_dir, "seismicData/segy/1_Original_Seismics.sgy")
    output_dir = os.path.join(base_dir, "seismicFlow3D/output")
    figures_dir = os.path.join(base_dir, "seismicFlow3D/figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(segy_path):
        console.print(f"[bold red]Error: File not found at {segy_path}[/bold red]")
        return

    # 1. Scanning SEGY
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[cyan]Scanning SEGY headers...", total=None)
        with segyio.open(segy_path, ignore_geometry=True) as s:
            ilines = s.attributes(segyio.TraceField.INLINE_3D)[:]
            xlines = s.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            unique_ilines = np.sort(np.unique(ilines))
            unique_xlines = np.sort(np.unique(xlines))
            num_samples = s.samples.size
            progress.update(task, description="Headers Scanned!")

    # 2. Information Table
    table = Table(title="Volume Information")
    table.add_column("Dimension", style="cyan"); table.add_column("Size", style="magenta"); table.add_column("Range", style="yellow")
    table.add_row("UNIQUE INLINES", str(len(unique_ilines)), f"{unique_ilines.min()} - {unique_ilines.max()}")
    table.add_row("UNIQUE XLINES", str(len(unique_xlines)), f"{unique_xlines.min()} - {unique_xlines.max()}")
    table.add_row("TIME SAMPLES", str(num_samples), "N/A")
    console.print(table)

    # 3. Sub-Volume Selection
    scope = Prompt.ask("Process [bold]FULL[/bold] volume or [bold]SUB-VOLUME[/bold]?", choices=["full", "sub-volume"], default="sub-volume")
    
    if scope == "sub-volume":
        console.print(f"\n[bold yellow]Select Sub-Volume for 3D Analysis:[/bold yellow]")
        
        s_start = IntPrompt.ask(f"Start Inline", default=int(unique_ilines.min()))
        s_end = IntPrompt.ask(f"End Inline", default=int(unique_ilines.max()))
        
        x_start = IntPrompt.ask(f"Start Xline", default=int(unique_xlines.min()))
        x_end = IntPrompt.ask(f"End Xline", default=int(unique_xlines.max()))

        t_start = IntPrompt.ask("Start Sample Index", default=0)
        t_end = IntPrompt.ask("End Sample Index", default=num_samples-1) + 1
    else:
        s_start, s_end = int(unique_ilines.min()), int(unique_ilines.max())
        x_start, x_end = int(unique_xlines.min()), int(unique_xlines.max())
        t_start, t_end = 0, num_samples

    # Filtered indices
    slices = unique_ilines[(unique_ilines >= s_start) & (unique_ilines <= s_end)]
    x_range = unique_xlines[(unique_xlines >= x_start) & (unique_xlines <= x_end)]
    
    if len(slices) == 0 or len(x_range) == 0:
        console.print("[bold red]Invalid range selected. Exiting.[/bold red]")
        return

    # 4. Loading 3D Volume
    console.print("\n[cyan]Loading 3D Volume into Memory...[/cyan]")
    # Initialize volume: (D, H, W) -> (Inline, Xline, Time)
    # Wait, segyio usually reads trace by trace.
    # We need to construct the volume carefully.
    
    # We want: volume[inline_idx, xline_idx, time]
    n_inlines = len(slices)
    n_xlines = len(x_range)
    n_times = t_end - t_start
    
    volume_3d = np.zeros((n_inlines, n_xlines, n_times), dtype=np.float32)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), console=console) as p:
        task = p.add_task("[green]Reading Traces...", total=n_inlines)
        
        with segyio.open(segy_path, ignore_geometry=True) as s:
            # Create a lookup for valid xlines in this sub-volume
            # To speed up, we can iterate inlines, find traces, filter xlines.
            
            # This might be slow if we iterate line by line.
            # Faster: Find all traces in the block?
            # segyio is fast at reading contiguous blocks if sorted.
            
            for i, il in enumerate(slices):
                # Find traces with this inline
                # And xline within range
                
                # Using header lookups can be slow.
                # Assuming standard sorted file:
                # Traces for inline `il` are likely contiguous.
                
                # Use boolean mask on arrays read earlier?
                mask = (ilines == il) & (xlines >= x_start) & (xlines <= x_end)
                if not mask.any(): 
                    p.advance(task)
                    continue
                    
                trace_indices = np.where(mask)[0]
                # Sort by xline
                trace_xlines = xlines[trace_indices]
                sort_idx = np.argsort(trace_xlines)
                trace_indices = trace_indices[sort_idx]
                trace_xlines = trace_xlines[sort_idx]
                
                # Read traces
                raw_traces = np.stack([s.trace[idx][t_start:t_end] for idx in trace_indices])
                
                # Map to volume matrix
                # trace_xlines contains xlines present.
                # volume index j corresponds to x_range[j]
                
                # We need to map xline value to index 0..n_xlines-1
                # x_range is sorted unique xlines requested.
                # Use searchsorted for fast index lookup
                
                valid_idx = np.searchsorted(x_range, trace_xlines)
                
                # Assign
                volume_3d[i, valid_idx, :] = raw_traces
                
                p.advance(task)
                
    console.print(f"Volume Loaded. Shape: {volume_3d.shape} (Inlines, Xlines, Time)")
    
    # 5. Compute 3D Structure Tensor
    console.print("\n[cyan]Computing 3D Structure Tensor on GPU...[/cyan]")
    sigma, rho = 1.0, 2.0
    
    # Send to GPU
    vol_gpu = cp.asarray(volume_3d)
    
    # Compute Tensor
    S_3d = gpu_structure_tensor_3d(vol_gpu, sigma, rho)
    
    # Compute Magnitude (3D Gradient Magnitude)
    # S = [Izz, Iyy, Ixx, ...]
    mag_3d_gpu = cp.sqrt(S_3d[0] + S_3d[1] + S_3d[2])
    mag_3d = mag_3d_gpu.get()
    
    # Extract Normal Vector
    normals_3d = eig_special_3d_gpu(S_3d) # (3, Inlines, Xlines, Time)
    
    console.print("3D Attributes Computed.")
    
    # 6. Flowline Generation (Pseudo-3D)
    num_clusters = 10
    sample_int = 100
    
    console.print(f"\n[cyan]Generating Flowlines (Pseudo-3D) along Inlines...[/cyan]")
    
    # Output containers
    segmentation_3d = np.zeros_like(volume_3d, dtype=np.float32)
    flowlines_3d = np.zeros_like(volume_3d, dtype=np.float32)
    orientation_3d = np.zeros_like(volume_3d, dtype=np.float32)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as p:
        task = p.add_task("Tracing Flowlines...", total=n_inlines)
        
        for i in range(n_inlines):
            # Extract Slice
            slice_data = vol_gpu[i].get().T # (T, X)
            
            # Extract Normal Components for this slice
            nx = normals_3d[1, i].get().T # (T, X)
            nt = normals_3d[2, i].get().T # (T, X)
            
            # Construct 2D Vector Field for this slice
            # Flow is orthogonal to normal within the plane.
            vectors = np.stack([-nx, nt]) 
            
            # Calculate 2D Orientation for this slice (for parity with 2D export)
            orient_slice = np.arctan2(vectors[0], vectors[1])
            orientation_3d[i] = orient_slice.T
            
            # Run Extraction
            surfaces, _ = extract_surfaces(slice_data, vectors, [sample_int], mode='both', device='gpu')
            
            # Accumulate Flowlines
            flow_slice = np.zeros_like(slice_data)
            for surf in surfaces:
                y, x = surf.path[:, 0].astype(int), surf.path[:, 1].astype(int)
                m = (y >= 0) & (y < flow_slice.shape[0]) & (x >= 0) & (x < flow_slice.shape[1])
                flow_slice[y[m], x[m]] = 1.0
            
            # K-Means Segmentation
            res_mask = np.zeros_like(slice_data)
            if len(surfaces) >= num_clusters:
                X, _ = surface_to_feature_vector(surfaces)
                X_scaled = StandardScaler().fit_transform(X)
                clusterer = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_scaled)
                for ind, surf in enumerate(surfaces): surf.kmeans_label = clusterer.labels_[ind]
                
                # Sorting
                sorted_indices = np.append(np.argsort(clusterer.cluster_centers_[:, 1]), [num_clusters + 1, num_clusters + 2])
                label_mapping = {old: new for new, old in enumerate(sorted_indices)}
                for surf in surfaces: surf.kmeans_label = label_mapping[surf.kmeans_label]
                sorted_tops = sort_tops(slice_data, surfaces, clusterer)
                interp_tops = [Surface(interpolate_boundary(top, slice_data.shape)) for top in sorted_tops]
                _, YM = np.meshgrid(np.arange(slice_data.shape[1]), np.arange(slice_data.shape[0]))
                lmap_inv = {v: k for k, v in label_mapping.items()}
                for j in range(len(interp_tops) - 1):
                    y1, y2 = interp_tops[j].path[:, 0], interp_tops[j+1].path[:, 0]
                    if j in lmap_inv: res_mask[np.where((YM > y1) & (YM < y2))] = lmap_inv[j]

            # Store in 3D volume (transpose back)
            segmentation_3d[i] = res_mask.T
            flowlines_3d[i] = flow_slice.T
            
            p.advance(task)

    console.print("[bold green]3D Processing Complete![/bold green]")

    # 7. Exports
    console.print("\n[yellow]Preparing for Export... gathering headers.[/yellow]")
    
    trace_indices_export = []
    
    with segyio.open(segy_path, ignore_geometry=True) as s:
        data_to_export_seg = []
        data_to_export_flow = []
        data_to_export_mag = []
        data_to_export_orient = []
        
        for i, il in enumerate(slices):
            mask = (ilines == il) & (xlines >= x_start) & (xlines <= x_end)
            if not mask.any(): continue
            
            t_idxs = np.where(mask)[0]
            tx = xlines[t_idxs]
            sort_ord = np.argsort(tx)
            t_idxs = t_idxs[sort_ord]
            
            trace_indices_export.extend(t_idxs.tolist())
            
            tx_sorted = tx[sort_ord]
            vol_x_indices = np.searchsorted(x_range, tx_sorted)
            
            # Extract traces
            data_to_export_seg.append(segmentation_3d[i, vol_x_indices, :])
            data_to_export_flow.append(flowlines_3d[i, vol_x_indices, :])
            data_to_export_mag.append(mag_3d[i, vol_x_indices, :])
            data_to_export_orient.append(orientation_3d[i, vol_x_indices, :])
            
        flat_seg = np.concatenate(data_to_export_seg, axis=0).T
        flat_flow = np.concatenate(data_to_export_flow, axis=0).T
        flat_mag = np.concatenate(data_to_export_mag, axis=0).T
        flat_orient = np.concatenate(data_to_export_orient, axis=0).T
        
    if Confirm.ask("Export 3D Segmentation?"):
        save_segy_volume(flat_seg, os.path.join(output_dir, "segmentation_3d.sgy"), segy_path, trace_indices_export)
        if Confirm.ask("Inspect Headers?"): inspect_segy_headers(os.path.join(output_dir, "segmentation_3d.sgy"))

    if Confirm.ask("Export 3D Gradient Tensor (Magnitude)?"):
        save_segy_volume(flat_mag, os.path.join(output_dir, "gradient_tensor_3d.sgy"), segy_path, trace_indices_export)

    if Confirm.ask("Export 3D Flowline Density?"):
        save_segy_volume(flat_flow, os.path.join(output_dir, "flowlines_3d.sgy"), segy_path, trace_indices_export)
        
    if Confirm.ask("Export 3D Vector Orientation?"):
        save_segy_volume(flat_orient, os.path.join(output_dir, "vector_orientation_3d.sgy"), segy_path, trace_indices_export)

    # 8. Visualization (3D Slicing)
    if Confirm.ask("\nVisualize 3D Results?"):
        # Select Axis
        view_axis = Prompt.ask("View Section", choices=["Inline", "Xline", "TimeSlice"], default="Inline")
        
        if view_axis == "Inline":
            idx = IntPrompt.ask(f"Select Inline ({slices.min()}-{slices.max()})", default=int(slices[len(slices)//2]))
            vol_idx = np.searchsorted(slices, idx)
            # Slice: (X, T) -> Transpose to (T, X) for plotting
            viz_seg = segmentation_3d[vol_idx].T
            viz_flow = flowlines_3d[vol_idx].T
            viz_seis = volume_3d[vol_idx].T
            viz_mag = mag_3d[vol_idx].T
            viz_orient = orientation_3d[vol_idx].T
            
        elif view_axis == "Xline":
            idx = IntPrompt.ask(f"Select Xline ({x_range.min()}-{x_range.max()})", default=int(x_range[len(x_range)//2]))
            vol_idx = np.searchsorted(x_range, idx)
            # Volume: (I, X, T)
            # Slice: volume[:, vol_idx, :] -> (I, T) -> Transpose (T, I)
            viz_seg = segmentation_3d[:, vol_idx, :].T
            viz_flow = flowlines_3d[:, vol_idx, :].T
            viz_seis = volume_3d[:, vol_idx, :].T
            viz_mag = mag_3d[:, vol_idx, :].T
            viz_orient = orientation_3d[:, vol_idx, :].T

        else: # TimeSlice
            idx = IntPrompt.ask(f"Select Time Index ({0}-{n_times-1})", default=n_times//2)
            # Plot (I, X)
            viz_seg = segmentation_3d[:, :, idx]
            viz_flow = flowlines_3d[:, :, idx]
            viz_seis = volume_3d[:, :, idx]
            viz_mag = mag_3d[:, :, idx]
            viz_orient = orientation_3d[:, :, idx]
            
        # Plot
        plt.figure(figsize=(15, 8))
        plt.subplot(231); plt.imshow(viz_seis, cmap='gray', aspect='auto'); plt.title(f"Seismic {view_axis} {idx}")
        plt.subplot(232); plt.imshow(viz_mag, cmap='viridis', aspect='auto'); plt.colorbar(label='Magnitude'); plt.title("Gradient Tensor Mag")
        plt.subplot(233); plt.imshow(viz_orient, cmap='twilight', aspect='auto'); plt.colorbar(label='Radians'); plt.title("Orientation")
        plt.subplot(234); plt.imshow(viz_flow, cmap='inferno', aspect='auto'); plt.title("Flow Density")
        plt.subplot(235); plt.imshow(viz_seg, cmap='jet', aspect='auto'); plt.title("Segmentation")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"QC_3D_{view_axis}_{idx}.png"))
        console.print(f"Saved QC figure to {figures_dir}")

if __name__ == "__main__":
    main()
