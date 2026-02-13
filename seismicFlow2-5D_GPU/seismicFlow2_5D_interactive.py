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

                           >>>>>>>>> [ 2.5D CASE ] <<<<<<<<<<<< 
    [/bold cyan]"""
    console.print(Panel.fit(banner_text, border_style="cyan"))

def main():
    print_rich_banner()
    
    # 0. Device selection
    device_choice = Prompt.ask("Select processing device", choices=["cpu", "gpu"], default="gpu")
    console.print(f"Using [bold magenta]{device_choice.upper()}[/bold magenta] for performance heavy operations")
    
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    segy_path = os.path.join(base_dir, "seismicData/segy/1_Original_Seismics.sgy")
    output_dir = os.path.join(base_dir, "seismicFlow2-5D/output")
    figures_dir = os.path.join(base_dir, "seismicFlow2-5D/figures")
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
    extraction_mode = Prompt.ask("Extract by [bold]inline[/bold] or [bold]xline[/bold]?", choices=["inline", "xline"], default="inline")
    scope = Prompt.ask("Process [bold]FULL[/bold] volume or [bold]SUB-VOLUME[/bold]?", choices=["full", "sub-volume"], default="sub-volume")
    
    if extraction_mode == "inline":
        slices_all, primary_header, secondary_header = unique_ilines, ilines, xlines
        primary_name, secondary_name = "Inlines", "Xlines"
    else:
        slices_all, primary_header, secondary_header = unique_xlines, xlines, ilines
        primary_name, secondary_name = "Xlines", "Inlines"

    if scope == "sub-volume":
        console.print(f"\n[bold yellow]Select {primary_name} Range ({slices_all.min()} - {slices_all.max()}):[/bold yellow]")
        while True:
            s_start = IntPrompt.ask(f"Start {extraction_mode}", default=int(slices_all.min()))
            s_end = IntPrompt.ask(f"End {extraction_mode}", default=int(slices_all.max()))
            if s_start > s_end: console.print("[bold red]Start must be <= End.[/bold red]"); continue
            slices = slices_all[(slices_all >= s_start) & (slices_all <= s_end)]
            if len(slices) == 0: console.print("[bold red]No data in range.[/bold red]"); continue
            break
        
        console.print(f"[bold yellow]Select {secondary_name} Range ({secondary_header.min()} - {secondary_header.max()}):[/bold yellow]")
        while True:
            sec_start = IntPrompt.ask(f"Start {secondary_name}", default=int(secondary_header.min()))
            sec_end = IntPrompt.ask(f"End {secondary_name}", default=int(secondary_header.max()))
            if sec_start > sec_end: console.print("[bold red]Start must be <= End.[/bold red]"); continue
            break
            
        console.print(f"[bold yellow]Select Time Range (0 - {num_samples-1}):[/bold yellow]")
        while True:
            t_start = IntPrompt.ask("Start Sample Index", default=0)
            t_end_p = IntPrompt.ask("End Sample Index", default=num_samples-1)
            if t_start > t_end_p: console.print("[bold red]Start must be <= End.[/bold red]"); continue
            t_end = t_end_p + 1; break
    else:
        slices = slices_all
        sec_start, sec_end = int(secondary_header.min()), int(secondary_header.max())
        t_start, t_end = 0, num_samples

    num_clusters = IntPrompt.ask("Number of segments/clusters", default=10)
    sample_int = IntPrompt.ask("Trace sample interval for RK4", default=100)
    sigma, rho = 1.0, 2.0
    
    # Storage - Use lists of 2D arrays (slices) to maintain traceability
    mask_slices, st_slices, flow_slices, orient_slices = [], [], [], []
    trace_indices_total = []
    
    start_total = time.time()
    
    # 7. Bulk Processing
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as p:
        task = p.add_task(f"[bold green]Processing {len(slices)} slices...", total=len(slices))
        
        for i, slice_idx in enumerate(slices):
            p.update(task, description=f"Processing {primary_name} {slice_idx} ({i+1}/{len(slices)})")
            
            t_idx = np.where((primary_header == slice_idx) & (secondary_header >= sec_start) & (secondary_header <= sec_end))[0]
            if len(t_idx) == 0: p.advance(task); continue
            sort_ord = np.argsort(secondary_header[t_idx])
            t_idx = t_idx[sort_ord]
            trace_indices_total.extend(t_idx.tolist())
            
            with segyio.open(segy_path, ignore_geometry=True) as s:
                seismic_2d = np.stack([s.trace[idx][t_start:t_end] for idx in t_idx]).T
            
            # ST & Vector Field
            if device_choice == 'gpu' and HAS_CUPY:
                S = gpu_structure_tensor_2d(seismic_2d, sigma, rho)
                mag_2d = cp.sqrt(S[1]**2 + S[2]**2).get()
                _, vectors = eig_special_2d_gpu(S); vectors = vectors.get()
            else:
                S = structure_tensor_2d(seismic_2d, sigma, rho)
                mag_2d = np.sqrt(S[..., 1]**2 + S[..., 2]**2) if S.ndim == 3 else np.ones_like(seismic_2d)
                _, vectors = eig_special_2d(S)
            
            # Surfaces & Flowlines
            surfaces, _ = extract_surfaces(seismic_2d, vectors, [sample_int], mode='both', device=device_choice)
            flow_slice = np.zeros_like(seismic_2d)
            for surf in surfaces:
                y, x = surf.path[:, 0].astype(int), surf.path[:, 1].astype(int)
                m = (y >= 0) & (y < flow_slice.shape[0]) & (x >= 0) & (x < flow_slice.shape[1]); flow_slice[y[m], x[m]] = 1.0

            # K-Means
            res_mask = np.zeros_like(seismic_2d)
            if len(surfaces) >= num_clusters:
                X, _ = surface_to_feature_vector(surfaces); X_scaled = StandardScaler().fit_transform(X)
                clusterer = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_scaled)
                for ind, surf in enumerate(surfaces): surf.kmeans_label = clusterer.labels_[ind]
                sorted_indices = np.append(np.argsort(clusterer.cluster_centers_[:, 1]), [num_clusters + 1, num_clusters + 2])
                label_mapping = {old: new for new, old in enumerate(sorted_indices)}
                for surf in surfaces: surf.kmeans_label = label_mapping[surf.kmeans_label]
                sorted_tops = sort_tops(seismic_2d, surfaces, clusterer)
                interp_tops = [Surface(interpolate_boundary(top, seismic_2d.shape)) for top in sorted_tops]
                _, YM = np.meshgrid(np.arange(seismic_2d.shape[1]), np.arange(seismic_2d.shape[0]))
                lmap_inv = {v: k for k, v in label_mapping.items()}
                for j in range(len(interp_tops) - 1):
                    y1, y2 = interp_tops[j].path[:, 0], interp_tops[j+1].path[:, 0]
                    if j in lmap_inv: res_mask[np.where((YM > y1) & (YM < y2))] = lmap_inv[j]
            
            # Orientation
            orient = np.arctan2(vectors[0], vectors[1]) # Check 2D script: it was (v[1], v[0])? 
            # 2D script says: orientation = np.arctan2(vector_array[1], vector_array[0])
            # My flow_utils return vectors as (2, H, W). 
            # So vectors[0] is dy (y-comp), vectors[1] is dx (x-comp)? 
            # Wait, structure_tensor returns [Iy2, IyIx, Ix2]. 
            # eig_special returns v1 (smallest eigenvector, flow direction).
            # v1 is [vy, vx]. So v[0] is y-comp, v[1] is x-comp.
            # arctan2(y, x). So arctan2(vectors[0], vectors[1]).
            orient = np.arctan2(vectors[0], vectors[1])

            mask_slices.append(res_mask); st_slices.append(mag_2d); flow_slices.append(flow_slice); orient_slices.append(orient)
            p.advance(task)

    console.print(f"\n[bold green]Bulk processing completed in {time.time() - start_total:.2f}s[/bold green]")

    # 8. QC Figure Suite (Single Pass)
    if Confirm.ask("\nWould you like to visualize a specific slice for QC?"):
        qc_val = IntPrompt.ask(f"Select {primary_name} index", choices=[str(int(s)) for s in slices], default=int(slices[len(slices)//2]))
        idx_qc = [i for i, s in enumerate(slices) if s == qc_val][0]
        
        # Reloading seismic for QC precisely
        t_qc = np.where((primary_header == qc_val) & (secondary_header >= sec_start) & (secondary_header <= sec_end))[0]
        t_qc = t_qc[np.argsort(secondary_header[t_qc])]
        with segyio.open(segy_path, ignore_geometry=True) as s: seis_qc = np.stack([s.trace[idx][t_start:t_end] for idx in t_qc]).T
        
        m_qc, st_qc, f_qc = mask_slices[idx_qc], st_slices[idx_qc], flow_slices[idx_qc]
        console.print(f"[cyan]Generating detailed 7-figure suite for {primary_name} {qc_val}...[/cyan]")
        
        # Vector Field for RK4 parity
        if device_choice == 'gpu': S_qc = gpu_structure_tensor_2d(seis_qc, sigma, rho); _, v_qc = eig_special_2d_gpu(S_qc); v_qc = v_qc.get()
        else: S_qc = structure_tensor_2d(seis_qc, sigma, rho); _, v_qc = eig_special_2d(S_qc)
        surfs_qc, _ = extract_surfaces(seis_qc, v_qc, [sample_int], mode='both', device=device_choice)
        
        # Plotting
        figs = [
            (seis_qc, 'gray', f"Seismic Slice - {qc_val}", f"slice_{extraction_mode}_{qc_val}", 'Amplitude'),
            (st_qc, 'viridis', "Structure Tensor Magnitude", f"vector_mag_{extraction_mode}_{qc_val}", 'Magnitude'),
            (np.arctan2(v_qc[0], v_qc[1]), 'twilight', "Vector Orientation", f"vector_orientation_{extraction_mode}_{qc_val}", 'Radians'),
            (m_qc, 'jet', "Segmentation Map", f"segmentation_{extraction_mode}_{qc_val}", 'Cluster ID')
        ]
        for data, cmap, title, fname, cbar_label in figs:
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap=cmap, aspect='auto')
            plt.colorbar(label=cbar_label)
            plt.title(title)
            plt.savefig(os.path.join(figures_dir, f"{fname}.png"))
            plt.close()
        
        # Specialty Plots
        plt.figure(figsize=(10, 8)); [plt.plot(s.path[:, 1], s.path[:, 0], 'k-', lw=0.5, alpha=0.5) for s in surfs_qc]; plt.title("Flowlines Geometries"); plt.gca().invert_yaxis(); plt.savefig(os.path.join(figures_dir, f"flowlines_{extraction_mode}_{qc_val}.png")); plt.close()
        plt.figure(figsize=(10, 8)); plt.imshow(seis_qc, cmap='gray', aspect='auto'); [plt.plot(s.path[:, 1], s.path[:, 0], 'r-', lw=0.8, alpha=0.7) for s in surfs_qc]; plt.title("Flowlines Overlay"); plt.savefig(os.path.join(figures_dir, f"overlay_{extraction_mode}_{qc_val}.png")); plt.close()
        
        # Interactive Colormap Selection
        cmap_choice = Prompt.ask("Choose colormap for flowlines", choices=["viridis", "jet", "plasma", "magma", "inferno"], default="jet")
        plt.figure(figsize=(10, 8))
        plt.imshow(seis_qc, cmap='gray', aspect='auto', alpha=0.3)
        colors = plt.get_cmap(cmap_choice)(np.linspace(0, 1, len(surfs_qc)))
        for ind, s in enumerate(surfs_qc): plt.plot(s.path[:, 1], s.path[:, 0], color=colors[ind], lw=0.8)
        
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_choice), norm=plt.Normalize(vmin=0, vmax=len(surfs_qc)))
        plt.colorbar(sm, ax=plt.gca(), label='Surface Index')
        plt.title(f"Colormap Flowlines ({cmap_choice})")
        plt.savefig(os.path.join(figures_dir, f"colormap_{extraction_mode}_{qc_val}.png")) 
        plt.close()
        
        console.print(f"[bold green]QC Suite saved to figures/[/bold green]")

    # 9. Reporting and Export
    table = Table(title="Export Summary")
    table.add_column("Property", style="cyan"); table.add_column("Value", style="magenta")
    table.add_row("Total Traces Extracted", str(len(trace_indices_total)))
    table.add_row("Sample Count", str(t_end - t_start))
    console.print(table)

    exports = [
        (mask_slices, "segmentation_volume.sgy", "Segmentation"), 
        (st_slices, "gradient_tensor_volume.sgy", "Gradient Tensor"), 
        (flow_slices, "flowlines_results.sgy", "Flowlines"),
        (orient_slices, "vector_orientation_volume.sgy", "Vector Orientation")
    ]
    for slist, fname, desc in exports:
        if Confirm.ask(f"Export {desc} to SEGY?", default=True):
            path = os.path.join(output_dir, fname)
            data_flat = np.concatenate([s.T for s in slist], axis=0).T # Stack traces side-by-side: (Samples, TotalTraces)
            with console.status(f"[bold green]Saving {desc}..."):
                save_segy_volume(data_flat, path, segy_path, trace_indices_total)
            console.print(f"[green]Exported {fname}[/green]")
            if Confirm.ask(f"Inspect headers for {fname}?"): inspect_segy_headers(path)

if __name__ == "__main__":
    main()
