import os
import sys
import time
import numpy as np
import segyio
import matplotlib.pyplot as plt
import scipy.signal as SKP
from sklearn.cluster import KMeans as SKKMeans
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# Try to import RAPIDS cuML
try:
    import cuml
    from cuml.cluster import KMeans as CUMLKMeans
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

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
    
    # 0.5 Implementation selection
    if device_choice == "gpu":
        if HAS_CUML:
            use_cuml = Prompt.ask("Select Clustering Engine", choices=["sk-learn", "cuml-rapids"], default="cuml-rapids") == "cuml-rapids"
        else:
            use_cuml = False
            console.print("[yellow]RAPIDS cuML not found. Falling back to SciKit-Learn.[/yellow]")
        
        slab_size = IntPrompt.ask("Slab size (sections processed together)", default=25)
        slab_size = max(1, min(slab_size, 50))
    else:
        use_cuml = False
        slab_size = 1
        console.print("[yellow]Running in serial mode (CPU).[/yellow]")

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

    slices = unique_ilines[(unique_ilines >= s_start) & (unique_ilines <= s_end)]
    x_range = unique_xlines[(unique_xlines >= x_start) & (unique_xlines <= x_end)]
    if len(slices) == 0 or len(x_range) == 0:
        console.print("[bold red]Invalid range selected. Exiting.[/bold red]"); return

    # 4. Loading 3D Volume
    console.print("\n[cyan]Loading 3D Volume into Memory...[/cyan]")
    n_inlines, n_xlines, n_times = len(slices), len(x_range), t_end - t_start
    volume_3d = np.zeros((n_inlines, n_xlines, n_times), dtype=np.float32)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), console=console) as p:
        task = p.add_task("[green]Reading Traces...", total=n_inlines)
        with segyio.open(segy_path, ignore_geometry=True) as s:
            for i, il in enumerate(slices):
                mask = (ilines == il) & (xlines >= x_start) & (xlines <= x_end)
                if not mask.any(): p.advance(task); continue
                trace_indices = np.where(mask)[0]
                sort_idx = np.argsort(xlines[trace_indices])
                trace_indices = trace_indices[sort_idx]
                raw_traces = np.stack([s.trace[idx][t_start:t_end] for idx in trace_indices])
                valid_idx = np.searchsorted(x_range, xlines[trace_indices])
                volume_3d[i, valid_idx, :] = raw_traces
                p.advance(task)
    
    console.print(f"Volume Loaded. Shape: {volume_3d.shape} (Inlines, Xlines, Time)")
    
    # 5. Compute 3D Structure Tensor
    console.print("\n[cyan]Computing 3D Structure Tensor on GPU...[/cyan]")
    sigma, rho = 1.0, 2.0
    start_st = time.time()
    
    # Send to GPU and keep it there
    vol_gpu = cp.asarray(volume_3d)
    S_3d = gpu_structure_tensor_3d(vol_gpu, sigma, rho)
    mag_3d_gpu = cp.sqrt(S_3d[0] + S_3d[1] + S_3d[2])
    mag_3d = mag_3d_gpu.get()
    normals_3d = eig_special_3d_gpu(S_3d)
    st_time = time.time() - start_st
    console.print(f"3D Attributes Computed in [bold green]{st_time:.2f}s[/bold green].")
    
    # 6. Optimized Flowline Generation (Slab Processing - Tiled Mode)
    num_clusters = 10
    sample_int = 100
    segmentation_3d = np.zeros_like(volume_3d, dtype=np.float32)
    flowlines_3d = np.zeros_like(volume_3d, dtype=np.float32)
    orientation_3d = np.zeros_like(volume_3d, dtype=np.float32)
    
    total_flowlines = 0
    start_rk = time.time()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), 
                  TextColumn("| Extracted: {task.fields[total_count]}"), console=console) as p:
        task = p.add_task("Tracing & Segmenting (Slab Mode)...", total=n_inlines, total_count=0)
        
        for i in range(0, n_inlines, slab_size):
            # Process a slab of inlines using TILED strategy
            end_i = min(i + slab_size, n_inlines)
            actual_slab = end_i - i
            
            # 1. Project Vectors and Stack into Tiled Space (Keep on GPU)
            # Normal components: 0=Inline, 1=Xline, 2=Time
            # Inplane trace: Xline (Y) and Time (X)
            # Slab Normals: (actual_slab, 3, Xline, Time)
            slab_nx = normals_3d[1, i:end_i] # (N, Xline, Time)
            slab_nt = normals_3d[2, i:end_i] # (N, Xline, Time)
            
            # orthogonal vectors in (Time, Xline) plane for each slice (Image Y=Time, X=Xline)
            # Normal on plane (n_time, n_xline). Tangent = (n_xline, -n_time)
            # v[0] (Y-comp/Time) = n_xline = slab_nx (Index 1)
            # v[1] (X-comp/Xline) = -n_time = -slab_nt (Index 2)
            # Transpose to (N, 2, Time, Xline)
            v_tiled = cp.stack([slab_nx.transpose(0, 2, 1), -slab_nt.transpose(0, 2, 1)], axis=1) # (N, 2, T, X)
            v_mag = cp.sqrt(v_tiled[:, 0]**2 + v_tiled[:, 1]**2) + 1e-10
            v_tiled /= v_mag[:, cp.newaxis, :, :] 
            
            # Orientation Storage (arctan2(X, Y) convention)
            orient_slab = cp.arctan2(v_tiled[:, 1], v_tiled[:, 0])
            orientation_3d[i:end_i] = orient_slab.transpose(0, 2, 1).get()

            # TILE the vectors: (2, T, N*X)
            # Move axis 0 (N) and 3 (X) together: (2, T, N, X) -> (2, T, N*X)
            v_giant = v_tiled.transpose(1, 2, 0, 3).reshape(2, n_times, actual_slab * n_xlines)
            
            # TILE the seismic data: (T, N*X)
            slice_slab = vol_gpu[i:end_i].transpose(0, 2, 1) # (N, T, X)
            slice_giant = slice_slab.transpose(1, 0, 2).reshape(n_times, actual_slab * n_xlines)
            
            # 2. SEED Point Generation (CPU-bound but batched per slab)
            all_seeds_y, all_seeds_x, all_rel_x = [], [], []
            slice_giant_cpu = slice_giant.get()
            
            for k in range(actual_slab):
                # We can multithread this if needed, but per-slab is okay
                # Standard spacing
                seeds_x = np.arange(0, n_xlines - 1, sample_int)
                for sx in seeds_x:
                    trace = slice_giant_cpu[:, k * n_xlines + sx]
                    peaks, _ = SKP.find_peaks(np.abs(trace)) # Need to import scipy.signal as SKP
                    for py in peaks:
                        all_seeds_y.append(py)
                        all_seeds_x.append(k * n_xlines + sx)
                        all_rel_x.append(sx)
            
            if not all_seeds_x:
                p.update(task, advance=actual_slab)
                continue
                
            # 3. ONE MASSIVE GPU LAUNCH (RK4)
            seeds_y_gpu = cp.array(all_seeds_y, dtype=cp.float32)
            seeds_x_gpu = cp.array(all_seeds_x, dtype=cp.float32)
            
            h_val, steps_val = 1.0, n_xlines # approx across one slice
            
            all_x_f, all_y_f = gpu_rk4_vectorized(seeds_x_gpu, seeds_y_gpu, h_val, steps_val, v_giant)
            all_x_b, all_y_b = gpu_rk4_vectorized(seeds_x_gpu, seeds_y_gpu, h_val, steps_val, v_giant * -1)
            
            cpu_xf, cpu_yf = all_x_f.get(), all_y_f.get()
            cpu_xb, cpu_yb = all_x_b.get(), all_y_b.get()
            
            # 4. Map back to surfaces and perform K-Means (Simplified per inline for now)
            # Partition seeds by slab index
            slab_seeds = [[] for _ in range(actual_slab)]
            for s_idx in range(len(all_seeds_x)):
                k = all_seeds_x[s_idx] // n_xlines
                # Construct Surface
                pf = np.column_stack((cpu_yf[:, s_idx], cpu_xf[:, s_idx]))
                pb = np.column_stack((cpu_yb[:, s_idx], cpu_xb[:, s_idx]))
                # Keep only valid (non-nan) and stay within slice boundary
                pf = pf[~np.isnan(pf).any(axis=1)]
                pb = pb[~np.isnan(pb).any(axis=1)]
                
                # Filter by slab bounds (x must be within k*X and (k+1)*X)
                xf_min, xf_max = k * n_xlines, (k+1) * n_xlines
                mf = (pf[:, 1] >= xf_min) & (pf[:, 1] < xf_max)
                mb = (pb[:, 1] >= xf_min) & (pb[:, 1] < xf_max)
                pf = pf[mf]; pb = pb[mb]
                
                # Correct x coordinate relative to slice
                pf[:, 1] -= k * n_xlines
                pb[:, 1] -= k * n_xlines
                
                merged = np.concatenate([pb[::-1], pf])
                slab_seeds[k].append(Surface(merged, all_seeds_x[s_idx] % n_xlines))

            # 5. Segmentation per inline (Batchable if needed, but cuML is fast)
            for k in range(actual_slab):
                current_idx = i + k
                surfaces = slab_seeds[k]
                total_flowlines += len(surfaces)
                
                slice_data = slice_slab[k].get()
                flow_slice = np.zeros_like(slice_data)
                res_mask = np.zeros_like(slice_data)
                
                for surf in surfaces:
                    paths = surf.path.astype(int)
                    m = (paths[:, 0] >= 0) & (paths[:, 0] < flow_slice.shape[0]) & \
                        (paths[:, 1] >= 0) & (paths[:, 1] < flow_slice.shape[1])
                    flow_slice[paths[m, 0], paths[m, 1]] = 1.0
                
                if len(surfaces) >= num_clusters:
                    X, _ = surface_to_feature_vector(surfaces)
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    if use_cuml:
                        X_gpu = cp.asarray(X_scaled, dtype=cp.float32)
                        clusterer = CUMLKMeans(n_clusters=num_clusters, n_init=1, max_iter=300).fit(X_gpu)
                        labels = clusterer.labels_.get()
                        centers = clusterer.cluster_centers_.get()
                    else:
                        clusterer = SKKMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_scaled)
                        labels = clusterer.labels_
                        centers = clusterer.cluster_centers_

                    for idx, surf in enumerate(surfaces): surf.kmeans_label = labels[idx]
                    sorted_indices = np.append(np.argsort(centers[:, 1]), [num_clusters + 1, num_clusters + 2])
                    label_mapping = {old: new for new, old in enumerate(sorted_indices)}
                    for surf in surfaces: surf.kmeans_label = label_mapping.get(surf.kmeans_label, num_clusters)
                    
                    sorted_tops = sort_tops(slice_data, surfaces, clusterer)
                    interp_tops = [Surface(interpolate_boundary(top, slice_data.shape)) for top in sorted_tops]
                    lmap_inv = {v: k for k, v in label_mapping.items()}
                    _, YM = np.meshgrid(np.arange(slice_data.shape[1]), np.arange(slice_data.shape[0]))
                    for j in range(len(interp_tops) - 1):
                        y1, y2 = interp_tops[j].path[:, 0], interp_tops[j+1].path[:, 0]
                        if j in lmap_inv: res_mask[np.where((YM > y1) & (YM < y2))] = lmap_inv[j]
                
                segmentation_3d[current_idx] = res_mask.T
                flowlines_3d[current_idx] = flow_slice.T
                
            p.update(task, advance=actual_slab, total_count=total_flowlines)

    console.print(f"[bold green]3D Processing Complete in {time.time() - start_rk:.2f}s![/bold green]")

    # 7. Exports
    trace_indices_export = []
    with segyio.open(segy_path, ignore_geometry=True) as s:
        data_seg, data_flow, data_mag, data_orient = [], [], [], []
        for i, il in enumerate(slices):
            mask = (ilines == il) & (xlines >= x_start) & (xlines <= x_end)
            if not mask.any(): continue
            t_idxs = np.where(mask)[0]
            tx = xlines[t_idxs]; sort_ord = np.argsort(tx)
            t_idxs = t_idxs[sort_ord]; trace_indices_export.extend(t_idxs.tolist())
            vx_idxs = np.searchsorted(x_range, tx[sort_ord])
            data_seg.append(segmentation_3d[i, vx_idxs, :])
            data_flow.append(flowlines_3d[i, vx_idxs, :])
            data_mag.append(mag_3d[i, vx_idxs, :])
            data_orient.append(orientation_3d[i, vx_idxs, :])
        
        flat_seg = np.concatenate(data_seg, axis=0).T
        flat_flow = np.concatenate(data_flow, axis=0).T
        flat_mag = np.concatenate(data_mag, axis=0).T
        flat_orient = np.concatenate(data_orient, axis=0).T

    if Confirm.ask("Export 3D Segmentation?"):
        save_segy_volume(flat_seg, os.path.join(output_dir, "segmentation_3d.sgy"), segy_path, trace_indices_export)
    if Confirm.ask("Export 3D Gradient Tensor (Magnitude)?"):
        save_segy_volume(flat_mag, os.path.join(output_dir, "gradient_tensor_3d.sgy"), segy_path, trace_indices_export)
    if Confirm.ask("Export 3D Flowline Density?"):
        save_segy_volume(flat_flow, os.path.join(output_dir, "flowlines_3d.sgy"), segy_path, trace_indices_export)
    if Confirm.ask("Export 3D Vector Orientation?"):
        save_segy_volume(flat_orient, os.path.join(output_dir, "vector_orientation_3d.sgy"), segy_path, trace_indices_export)

    # 8. Optimized Visualization
    if Confirm.ask("\nVisualize 3D Results?"):
        view_axis = Prompt.ask("View Section", choices=["Inline", "Xline", "TimeSlice"], default="Inline")
        if view_axis == "Inline":
            idx = IntPrompt.ask(f"Select Inline ({slices.min()}-{slices.max()})", default=int(slices[len(slices)//2]))
            vol_idx = np.searchsorted(slices, idx)
            viz_seis, viz_mag, viz_orient, viz_flow, viz_seg = volume_3d[vol_idx].T, mag_3d[vol_idx].T, orientation_3d[vol_idx].T, flowlines_3d[vol_idx].T, segmentation_3d[vol_idx].T
        elif view_axis == "Xline":
            idx = IntPrompt.ask(f"Select Xline ({x_range.min()}-{x_range.max()})", default=int(x_range[len(x_range)//2]))
            vol_idx = np.searchsorted(x_range, idx); viz_seis, viz_mag, viz_orient, viz_flow, viz_seg = volume_3d[:, vol_idx, :].T, mag_3d[:, vol_idx, :].T, orientation_3d[:, vol_idx, :].T, flowlines_3d[:, vol_idx, :].T, segmentation_3d[:, vol_idx, :].T
        else:
            idx = IntPrompt.ask(f"Select Time Index (0-{n_times-1})", default=n_times//2)
            viz_seis, viz_mag, viz_orient, viz_flow, viz_seg = volume_3d[:, :, idx], mag_3d[:, :, idx], orientation_3d[:, :, idx], flowlines_3d[:, :, idx], segmentation_3d[:, :, idx]

        # RE-CALCULATE 2D MAGNITUDE FOR PARITY
        # This is the "Visual Correction" step - using 2D gradients for the QC plot intensity
        if view_axis == "Inline":
            temp_S = gpu_structure_tensor_2d(cp.asarray(viz_seis), sigma=1.0, rho=2.0)
            viz_mag_2d = cp.sqrt(temp_S[0] + temp_S[1]).get()
        else:
            viz_mag_2d = viz_mag

        amplitude_max = np.percentile(np.abs(viz_seis), 99)
        axis_str = view_axis.lower()

        plt.figure(figsize=(10, 8)); plt.imshow(viz_seis, vmin=-amplitude_max, vmax=amplitude_max, cmap='Greys', aspect='auto'); plt.colorbar(label='Amplitude'); plt.title(f'SEISMIC {view_axis.upper()} {idx}'); plt.savefig(os.path.join(figures_dir, f"slice_{axis_str}_{idx}.png"))
        plt.figure(figsize=(10, 8)); plt.imshow(viz_mag_2d, cmap='viridis', aspect='auto'); plt.colorbar(label='Magnitude'); plt.title(f"2D Gradient Magnitude - {view_axis} {idx}"); plt.savefig(os.path.join(figures_dir, f"vector_mag_{axis_str}_{idx}.png"))
        plt.figure(figsize=(10, 8)); plt.imshow(viz_orient, cmap='twilight', aspect='auto'); plt.colorbar(label='Radians'); plt.title(f"Vector Orientation - {view_axis} {idx}"); plt.savefig(os.path.join(figures_dir, f"vector_orientation_{axis_str}_{idx}.png"))
        plt.figure(figsize=(10, 8)); plt.imshow(viz_flow, cmap='binary', aspect='auto'); plt.title(f"Flowlines - {view_axis} {idx}"); plt.savefig(os.path.join(figures_dir, f"flowlines_{axis_str}_{idx}.png"))
        plt.figure(figsize=(10, 8)); plt.imshow(viz_seis, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto'); plt.imshow(np.where(viz_flow > 0, 1, np.nan), cmap='Reds_r', alpha=0.8, aspect='auto'); plt.title(f"Flowlines Overlay - {view_axis} {idx}"); plt.savefig(os.path.join(figures_dir, f"overlay_{axis_str}_{idx}.png"))
        
        cmap_choice = Prompt.ask("Choose colormap for flowlines", default="jet")
        plt.figure(figsize=(10, 8)); plt.imshow(viz_seis, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto', alpha=0.3); plt.imshow(np.where(viz_flow > 0, viz_seg, np.nan), cmap=cmap_choice, aspect='auto'); plt.colorbar(label='Segment ID'); plt.title(f"Colormap Flowlines - {view_axis} {idx}"); plt.savefig(os.path.join(figures_dir, f"colormap_{axis_str}_{idx}.png"))
        plt.figure(figsize=(10, 8)); plt.imshow(viz_seg, cmap='jet', aspect='auto'); plt.colorbar(label='Segment ID'); plt.title(f"Segmentation - {view_axis} {idx}"); plt.savefig(os.path.join(figures_dir, f"segmentation_{axis_str}_{idx}.png"))
        console.print(f"[bold green]Saved optimized QC suite to {figures_dir}[/bold green]")

if __name__ == "__main__":
    main()
