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

                           >>>>>>>>> [ 2.5D+ CASE ] <<<<<<<<<<<< 
    [/bold cyan]"""
    console.print(Panel.fit(banner_text, border_style="cyan"))

def visualize_qc(volume_3d, vol_seg, vol_mag, vol_orient, flow_features_il, flow_features_xl, 
                 slices_il, slices_xl, t_start, t_end, extraction_mode, figures_dir, device_choice, sample_int):
    """
    Interactive QC options for the 2.5D+ volume.
    """
    import matplotlib.pyplot as plt
    
    modes_to_ask = []
    if extraction_mode == 'both': modes_to_ask = ['inline', 'xline']
    elif extraction_mode == 'inline': modes_to_ask = ['inline']
    else: modes_to_ask = ['xline']
    
    for mode in modes_to_ask:
        if not Confirm.ask(f"\nVisualize a specific [bold]{mode}[/bold] slice?"):
            continue
            
        # Select Slice
        if mode == 'inline':
            valid_slices = slices_il
            prompt_text = "Select Inline index"
        else:
            valid_slices = slices_xl
            prompt_text = "Select Xline index"
            
        slice_val = IntPrompt.ask(prompt_text, choices=[str(int(s)) for s in valid_slices], default=int(valid_slices[len(valid_slices)//2]))
        
        # Find index in 3D volume
        # volume_3d is (Inlines, Xlines, Time)
        try:
            if mode == 'inline':
                idx = np.where(slices_il == slice_val)[0][0]
                seis_qc = volume_3d[idx, :, :].T # (Time, Xlines)
                seg_qc = vol_seg[idx, :, :].T
                mag_qc = vol_mag[idx, :, :].T
                orient_qc = vol_orient[idx, :, :].T
                
                # Filter flowlines for this slice
                surfs_qc = [s for s in flow_features_il if s.slice_index == idx]
                fname_suffix = f"inline_{slice_val}"
                
            else:
                idx = np.where(slices_xl == slice_val)[0][0]
                seis_qc = volume_3d[:, idx, :].T # (Time, Inlines)
                seg_qc = vol_seg[:, idx, :].T
                mag_qc = vol_mag[:, idx, :].T
                orient_qc = vol_orient[:, idx, :].T
                
                surfs_qc = [s for s in flow_features_xl if s.slice_index == idx]
                fname_suffix = f"xline_{slice_val}"

            console.print(f"[cyan]Generating detailed 7-figure suite for {mode} {slice_val}...[/cyan]")

            # Plotting
            figs = [
                (seis_qc, 'gray', f"Seismic Slice - {mode} {slice_val}", f"slice_{fname_suffix}", 'Amplitude'),
                (mag_qc, 'viridis', "Structure Tensor Magnitude", f"vector_mag_{fname_suffix}", 'Magnitude'),
                (orient_qc, 'twilight', "Vector Orientation", f"vector_orientation_{fname_suffix}", 'Radians'),
                (seg_qc, 'jet', "Segmentation Map", f"segmentation_{fname_suffix}", 'Cluster ID')
            ]
            
            for data, cmap, title, fname, cbar_label in figs:
                plt.figure(figsize=(10, 8))
                plt.imshow(data, cmap=cmap, aspect='auto')
                plt.colorbar(label=cbar_label)
                plt.title(title)
                plt.savefig(os.path.join(figures_dir, f"{fname}.png"))
                plt.close()
            
            # Specialty Plots
            # 5. Flowlines Geometry
            plt.figure(figsize=(10, 8))
            if surfs_qc: [plt.plot(s.path[:, 1], s.path[:, 0], 'k-', lw=0.5, alpha=0.5) for s in surfs_qc]
            plt.title(f"Flowlines Geometries - {mode} {slice_val}")
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(figures_dir, f"flowlines_{fname_suffix}.png"))
            plt.close()

            # 6. Overlay
            plt.figure(figsize=(10, 8))
            plt.imshow(seis_qc, cmap='gray', aspect='auto')
            if surfs_qc: [plt.plot(s.path[:, 1], s.path[:, 0], 'r-', lw=0.8, alpha=0.7) for s in surfs_qc]
            plt.title(f"Flowlines Overlay - {mode} {slice_val}")
            plt.savefig(os.path.join(figures_dir, f"overlay_{fname_suffix}.png"))
            plt.close()
            
            # 7. Colormap
            cmap_choice = Prompt.ask("Choose colormap for flowlines", choices=["viridis", "jet", "plasma", "magma", "inferno"], default="jet")
            plt.figure(figsize=(10, 8))
            plt.imshow(seis_qc, cmap='gray', aspect='auto', alpha=0.3)
            
            if surfs_qc:
                colors = plt.get_cmap(cmap_choice)(np.linspace(0, 1, len(surfs_qc)))
                for ind, s in enumerate(surfs_qc): 
                    plt.plot(s.path[:, 1], s.path[:, 0], color=colors[ind], lw=0.8)
                
                sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_choice), norm=plt.Normalize(vmin=0, vmax=len(surfs_qc)))
                plt.colorbar(sm, ax=plt.gca(), label='Surface Index')
            
            plt.title(f"Colormap Flowlines ({cmap_choice})")
            plt.savefig(os.path.join(figures_dir, f"colormap_{fname_suffix}.png")) 
            plt.close()
            
            console.print(f"[bold green]QC Suite saved to figures/[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error generating QC plots: {e}[/bold red]")

def main():
    print_rich_banner()
    
    # 0. Device selection
    device_choice = Prompt.ask("Select processing device", choices=["cpu", "gpu"], default="gpu")
    console.print(f"Using [bold magenta]{device_choice.upper()}[/bold magenta] for performance heavy operations")
    
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    segy_path = os.path.join(base_dir, "seismicData/segy/1_Original_Seismics.sgy")
    output_dir = os.path.join(base_dir, "seismicFlow2-5D_Plus/output")
    figures_dir = os.path.join(base_dir, "seismicFlow2-5D_Plus/figures")
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
    table = Table(title="Volume Information (Sparse)")
    table.add_column("Dimension", style="cyan"); table.add_column("Size", style="magenta"); table.add_column("Range", style="yellow")
    table.add_row("UNIQUE INLINES", str(len(unique_ilines)), f"{unique_ilines.min()} - {unique_ilines.max()}")
    table.add_row("UNIQUE XLINES", str(len(unique_xlines)), f"{unique_xlines.min()} - {unique_xlines.max()}")
    table.add_row("TIME SAMPLES", str(num_samples), "N/A")
    console.print(table)

    # 3. Mode Selection
    extraction_mode = Prompt.ask("Extract by [bold]inline[/bold], [bold]xline[/bold] or [bold]both[/bold]?", choices=["inline", "xline", "both"], default="both")
    scope = Prompt.ask("Process [bold]FULL[/bold] volume or [bold]SUB-VOLUME[/bold]?", choices=["full", "sub-volume"], default="sub-volume")
    
    modes_to_process = []
    if extraction_mode == "both":
        modes_to_process = ["inline", "xline"]
    else:
        modes_to_process = [extraction_mode]

    # Defaults for initial selection (usually Inline ranges define the volume box)
    slices_all, primary_header, secondary_header = unique_ilines, ilines, xlines
    primary_name, secondary_name = "Inlines", "Xlines"

    if scope == "sub-volume":
        # Always ask for Inline/Xline/Time ranges to define the 3D box
        console.print(f"\n[bold yellow]Select Inline Range ({unique_ilines.min()} - {unique_ilines.max()}):[/bold yellow]")
        while True:
            il_start = IntPrompt.ask("Start Inline", default=int(unique_ilines.min()))
            il_end = IntPrompt.ask("End Inline", default=int(unique_ilines.max()))
            if il_start > il_end: console.print("[bold red]Start must be <= End.[/bold red]"); continue
            break
        
        console.print(f"\n[bold yellow]Select Xline Range ({unique_xlines.min()} - {unique_xlines.max()}):[/bold yellow]")
        while True:
            xl_start = IntPrompt.ask("Start Xline", default=int(unique_xlines.min()))
            xl_end = IntPrompt.ask("End Xline", default=int(unique_xlines.max()))
            if xl_start > xl_end: console.print("[bold red]Start must be <= End.[/bold red]"); continue
            break
            
        console.print(f"\n[bold yellow]Select Time Range (0 - {num_samples-1}):[/bold yellow]")
        while True:
            t_start = IntPrompt.ask("Start Sample Index", default=0)
            t_end_p = IntPrompt.ask("End Sample Index", default=num_samples-1)
            if t_start > t_end_p: console.print("[bold red]Start must be <= End.[/bold red]"); continue
            t_end = t_end_p + 1; break
            
        # Define the box
        slices_il = unique_ilines[(unique_ilines >= il_start) & (unique_ilines <= il_end)]
        slices_xl = unique_xlines[(unique_xlines >= xl_start) & (unique_xlines <= xl_end)]
    else:
        slices_il = unique_ilines
        slices_xl = unique_xlines
        # For full volume, we still need 3D limits for loading
        il_start, il_end = int(unique_ilines.min()), int(unique_ilines.max())
        xl_start, xl_end = int(unique_xlines.min()), int(unique_xlines.max())
        t_start, t_end = 0, num_samples

    num_clusters = IntPrompt.ask("Number of segments/clusters", default=10)
    sample_int = IntPrompt.ask("Trace sample interval for RK4", default=100)
    sigma, rho = 1.0, 2.0
    
    # Storage - Use lists of 2D arrays (slices) to maintain traceability
    mask_slices, st_slices, flow_slices, orient_slices = [], [], [], []
    trace_indices_total = []
    
    start_total = time.time()
    
    # 6. Load FULL Volume to VRAM (2.5D+ Optimization)
    console.print(f"\n[bold yellow]Loading Volume to {device_choice.upper()} Memory...[/bold yellow]")
    
    # Calculate total size to prevent OOM
    # Dimensions: (Inlines, Xlines, Time)
    n_inlines = len(slices_all) if scope == 'full' else len(slices_il)
    n_xlines = len(secondary_header) if scope == 'full' else len(slices_xl)
    # Better: use unique counts
    # But for SEGY reading, we need to be careful.
    
    # Let's read the bulk data first
    # We need to read all traces interacting with our Inlines/Xlines range
    
    start_load = time.time()
    
    # We will read into a 3D numpy array first, then move to GPU
    # Shape: (Primary, Secondary, Time)
    # This requires a structured read.
    
    # Define volume shape
    # If Inline mode: (Inlines, Xlines, Time)
    # If Xline mode: (Xlines, Inlines, Time)
    
    # Indices for the selected range
    # slices is our Primary selection
    # sec_start/sec_end is our Secondary selection
    
    # Load Volume: (Inlines, Xlines, Time)
    n_inlines = len(slices_il)
    n_xlines = len(slices_xl)
    n_samples = t_end - t_start
    
    vol_shape = (n_inlines, n_xlines, n_samples)
    volume_3d = np.zeros(vol_shape, dtype=np.float32)
    
    # Trace Mapping for SEGY Export
    # Key: (Inline, Xline), Value: Original Trace Index
    trace_map = {}
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), console=console) as p:
        task = p.add_task(f"[cyan]Reading {n_inlines} inlines to RAM...", total=n_inlines)
        
        with segyio.open(segy_path, ignore_geometry=True) as s:
            for i, il_val in enumerate(slices_il):
                # We need all traces where INLINE == il_val AND XLINE inside [slices_xl range]
                # Pre-filtering is hard without geometry, so we rely on finding by header.
                # Assuming standard sorting:
                
                # To be robust:
                t_idx = np.where((ilines == il_val) & (xlines >= slices_xl.min()) & (xlines <= slices_xl.max()))[0]
                
                if len(t_idx) > 0:
                    # Sort by Xline to map to our 2nd dimension
                    tr_xlines = xlines[t_idx]
                    sort_ord = np.argsort(tr_xlines)
                    t_idx = t_idx[sort_ord]
                    tr_xlines = tr_xlines[sort_ord]
                    
                    # Store indices for export reference
                    for k, xl_val in enumerate(tr_xlines):
                        trace_map[(il_val, xl_val)] = t_idx[k]
                    
                    # Read data
                    tr_data = np.stack([s.trace[idx][t_start:t_end] for idx in t_idx]).T # (Time, FoundXlines)
                    
                    # Map found Xlines to our volume grid
                    # We have explicit slices_xl.
                    # Use searchsorted to find indices
                    # valid_mask = np.isin(tr_xlines, slices_xl)
                    # relevant_x = tr_xlines[valid_mask]
                    # relevant_data = tr_data[:, valid_mask]
                    
                    # Common case: contiguous.
                    # Let's map directly.
                    # Indices in volume:
                    vol_indices = np.searchsorted(slices_xl, tr_xlines)
                    
                    # Filter out any that didn't match (should fit due to range check, but safety first)
                    valid = (vol_indices >= 0) & (vol_indices < n_xlines)
                    # Also strict check if needed: slices_xl[vol_indices] == tr_xlines
                    
                    # Assign
                    # volume_3d[i, vol_idx, :] = trace[:] 
                    # Volume is (IL, XL, Time). Trace is (Time).
                    # tr_data is (Time, N).
                    
                    for j, v_idx in enumerate(vol_indices):
                        if valid[j]:
                            if slices_xl[v_idx] == tr_xlines[j]:
                                volume_3d[i, v_idx, :] = tr_data[:, j]

                p.advance(task)

    console.print(f"Volume Loaded. Shape: {volume_3d.shape} (Inlines, Xlines, Time)")

    # Move to GPU
    if device_choice == 'gpu' and HAS_CUPY:
        console.print("[bold magenta]Moving Volume to GPU VRAM...[/bold magenta]")
        volume_gpu = cp.asarray(volume_3d)
        console.print(f"GPU Volume Size: {volume_gpu.nbytes / 1e9:.2f} GB")
    else:
        volume_gpu = volume_3d

    # Initialize 3D Output Volumes (Inlines, Xlines, Time) -> Match volume_3d shape
    vol_seg = np.full(vol_shape, -1, dtype=np.float32)
    vol_flow = np.zeros(vol_shape, dtype=np.float32)
    vol_mag = np.zeros(vol_shape, dtype=np.float32)
    vol_orient = np.zeros(vol_shape, dtype=np.float32)

    # 7. Dual-Direction Processing
    flowline_features_il = []
    flowline_features_xl = []
    
    start_total = time.time()
    
    for mode in modes_to_process:
        console.print(f"\n[bold green]Processing Mode: {mode.upper()}[/bold green]")
        
        # Define loop range and data access
        if mode == "inline":
            loop_range = range(n_inlines)
            desc = "Processing Inlines"
        else:
            loop_range = range(n_xlines)
            desc = "Processing Xlines"
            
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), console=console) as p:
            task = p.add_task(f"{desc}...", total=len(loop_range))
            
            for idx in loop_range:
                # Extract Slice from VRAM
                if mode == "inline":
                    # volume_gpu is (IL, XL, Time)
                    # We want (Time, XL) image
                    seismic_2d = volume_gpu[idx, :, :].T # (Time, XL)
                else:
                    # Xline slice: (IL, idx, Time) -> (Time, IL) image
                    seismic_2d = volume_gpu[:, idx, :].T # (Time, IL)
                
                # ST & Vectors
                if device_choice == 'gpu' and HAS_CUPY:
                    S = gpu_structure_tensor_2d(seismic_2d, sigma, rho)
                    _, vectors = eig_special_2d_gpu(S)
                    mag = (S[0,0] + S[1,1]).get()
                    orient = cp.arctan2(vectors[0], vectors[1]).get()
                    vectors = vectors.get()
                    seis_cpu = seismic_2d.get() if hasattr(seismic_2d, 'get') else seismic_2d
                else:
                    S = structure_tensor_2d(seismic_2d, sigma, rho)
                    _, vectors = eig_special_2d(S)
                    mag = S[0,0] + S[1,1]
                    orient = np.arctan2(vectors[0], vectors[1])
                    seis_cpu = seismic_2d
                
                # Store Attributes in 3D Volumes
                new_mag = mag.T 
                new_orient = orient.T
                if mode == "inline":
                    update_mask = (new_mag > vol_mag[idx])
                    vol_mag[idx] = np.maximum(vol_mag[idx], new_mag)
                    vol_orient[idx] = np.where(update_mask, new_orient, vol_orient[idx])
                else:
                    update_mask = (new_mag > vol_mag[:, idx])
                    vol_mag[:, idx] = np.maximum(vol_mag[:, idx], new_mag)
                    vol_orient[:, idx] = np.where(update_mask, new_orient, vol_orient[:, idx])

                # Flowlines
                surfaces, _ = extract_surfaces(seis_cpu, vectors, [sample_int], mode='both', device=device_choice)
                
                # Store surfaces with their 3D coordinates
                for surf in surfaces:
                    surf.slice_index = idx
                    surf.mode = mode 
                    if mode == "inline":
                        flowline_features_il.append(surf)
                    else:
                        flowline_features_xl.append(surf)
                
                p.advance(task)

    # 8. Grid Fusion & Volume Construction
    console.print("\n[bold cyan]Constructing 3D Volumes from 2.5D+ Grid...[/bold cyan]")
    
    # Helper to paint flowlines into 3D volume
    def paint_features(features_list, mode):
        for surf in features_list:
            idx = surf.slice_index
            ys = surf.path[:, 0].astype(int); xs = surf.path[:, 1].astype(int)
            ys = np.clip(ys, 0, n_samples - 1)
            
            if mode == 'inline':
                xs = np.clip(xs, 0, n_xlines - 1)
                vol_flow[idx, xs, ys] = 1.0
            else:
                xs = np.clip(xs, 0, n_inlines - 1)
                vol_flow[xs, idx, ys] = 1.0

    # Clustering
    all_surfaces = flowline_features_il + flowline_features_xl
    if len(all_surfaces) > num_clusters:
        console.print(f"[cyan]Running Global Clustering on {len(all_surfaces)} flowlines...[/cyan]")
        X, _ = surface_to_feature_vector(all_surfaces)
        X_scaled = StandardScaler().fit_transform(X)
        clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(X_scaled)
        
        # Sort labels by depth
        centers = clusterer.cluster_centers_
        sorted_indices = np.argsort(centers[:, 0])
        label_mapping = {old: new for new, old in enumerate(sorted_indices)}
        
        for i, surf in enumerate(all_surfaces):
            surf.kmeans_label = label_mapping[clusterer.labels_[i]]
            
        # Paint Segmentation Seed Points
        for surf in all_surfaces:
            lab = surf.kmeans_label
            idx = surf.slice_index
            ys = surf.path[:, 0].astype(int); xs = surf.path[:, 1].astype(int)
            ys = np.clip(ys, 0, n_samples - 1)
            if surf.mode == 'inline':
                xs = np.clip(xs, 0, n_xlines - 1); vol_seg[idx, xs, ys] = lab
            else:
                xs = np.clip(xs, 0, n_inlines - 1); vol_seg[xs, idx, ys] = lab
                
    else:
        console.print("[red]Not enough surfaces for clustering.[/red]")

    paint_features(flowline_features_il, 'inline')
    paint_features(flowline_features_xl, 'xline')
    
    # Interpolation for Segmentation Volume
    console.print("[bold yellow]Interpolating Sparse Grid to Full 3D Volume...[/bold yellow]")
    from scipy.ndimage import distance_transform_edt
    
    mask_data = (vol_seg != -1)
    if np.any(mask_data):
        try:
            # Nearest Neighbor Interpolation
            indices = distance_transform_edt(~mask_data, return_distances=False, return_indices=True)
            vol_seg = vol_seg[tuple(indices)]
        except Exception as e:
            console.print(f"[red]Interpolation failed: {e}.[/red]")

    export_slices = [vol_seg, vol_mag, vol_flow, vol_orient] 
    # Align formatting for export
    # Standard export expects list of 2D slices? 
    # Or we can write the 3D volume directly.
    # Our save_segy_volume function takes (Samples, Traces) flat.

    # Export
    exports = [
        (vol_seg, "segmentation_volume.sgy", "Segmentation"), 
        (vol_mag, "gradient_tensor_volume.sgy", "Gradient Tensor"), 
        (vol_flow, "flowlines_results.sgy", "Flowlines"),
        (vol_orient, "vector_orientation_volume.sgy", "Vector Orientation")
    ]
    
    # Pre-calculate flattened trace indices for dense export
    # Grid order: Inline (slow), Xline (fast) -> Matches volume_3d[i, j, :] flattened to [i*Nx + j]
    dense_trace_indices = []
    # Also prepare geometry overrides (IL, XL)
    # We can do this on the fly to save memory, or pre-calc.
    # On the fly is fine.
    
    console.print("\n[bold cyan]Exporting Results...[/bold cyan]")
    
    for vol, fname, desc in exports:
        # 1. NPY Export (Always)
        npy_name = fname.replace('.sgy', '.npy')
        np.save(os.path.join(output_dir, npy_name), vol)
        console.print(f"[green]Saved {npy_name}[/green]")
        
        # 2. SEGY Export (Optional)
        if Confirm.ask(f"Export {desc} to SEGY?", default=True):
            sgy_path = os.path.join(output_dir, fname)
            
            # Prepare data: Flatten (IL, XL, Time) -> (Traces, Time) then Transpose to (Samples, Traces)?
            # vol shape: (Ni, Nx, Nt). 
            # Flatten to (Ni*Nx, Nt). 
            # SEGY expects (Traces, Samples) usually for writes if assigning to dst.trace[i]
            # But segyio.create dst.trace can take (Traces, Samples) array? 
            # Let's check save_segy_volume usage: flat_data = data.T.astype(np.float32). 
            # If input equal to (Samples, Traces), then .T is (Traces, Samples).
            
            # Here vol is (Ni, Nx, Nt). Reshape to (Ni*Nx, Nt).
            flat_vol = vol.reshape(-1, n_samples) # (Traces, Samples)
            
            with console.status(f"[bold green]Writing SEGY {fname}..."):
                # Use source SEGY as template for headers
                with segyio.open(segy_path, "r", ignore_geometry=True) as src:
                    spec = segyio.spec()
                    spec.sorting = 1
                    spec.format = src.format
                    spec.samples = src.samples[:n_samples]
                    spec.tracecount = n_inlines * n_xlines
                    
                    with segyio.create(sgy_path, spec) as dst:
                        dst.text[0] = src.text[0]
                        dst.bin = src.bin
                        dst.bin[segyio.BinField.Samples] = n_samples
                        
                        # Copy headers and Data
                        # This loop can be slow in Python.
                        # Optimization: Write data in bulk if possible.
                        dst.trace[:] = flat_vol
                        
                        # Headers need per-trace update
                        # We use trace_map to find original header, else use default (0)
                        
                        # Cache src headers? No, memory heavy.
                        # Accessing src.header[k] is disk seek. Slow.
                        # Optimization: Read all headers from src once?
                        # s.header is a proxy.
                        
                        # Let's iterate.
                        for i in range(n_inlines):
                            il = slices_il[i]
                            for j in range(n_xlines):
                                xl = slices_xl[j]
                                trace_idx = i * n_xlines + j
                                
                                # Source mapping
                                src_idx = trace_map.get((il, xl), 0) # Default to 0 if missing (interpolated)
                                
                                # Copy header
                                dst.header[trace_idx] = src.header[src_idx]
                                
                                # Overwrite Geometry
                                dst.header[trace_idx][segyio.TraceField.INLINE_3D] = int(il)
                                dst.header[trace_idx][segyio.TraceField.CROSSLINE_3D] = int(xl)
                                # Update Sample Count
                                dst.header[trace_idx][segyio.TraceField.TRACE_SAMPLE_COUNT] = n_samples
                                
            console.print(f"[green]Saved {fname}[/green]")
            
            if Confirm.ask(f"Inspect headers for {fname}?"): 
                inspect_segy_headers(sgy_path)

    # 9. QC Visualization
    visualize_qc(volume_3d, vol_seg, vol_mag, vol_orient, flowline_features_il, flowline_features_xl,
                 slices_il, slices_xl, t_start, t_end, extraction_mode, figures_dir, device_choice, sample_int)

if __name__ == "__main__":
    main()
