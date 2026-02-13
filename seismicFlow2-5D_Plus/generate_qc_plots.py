import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = '/home/roderickperez/DataScienceProjects/seismicFlow/seismicFlow2-5D_Plus/output'
figures_dir = '/home/roderickperez/DataScienceProjects/seismicFlow/seismicFlow2-5D_Plus/figures'
os.makedirs(figures_dir, exist_ok=True)

# Load volumes (51, 51, 51)
vol_seg = np.load(os.path.join(output_dir, 'segmentation_volume.npy'))
vol_mag = np.load(os.path.join(output_dir, 'gradient_tensor_volume.npy'))
vol_orient = np.load(os.path.join(output_dir, 'vector_orientation_volume.npy'))
# We don't have the original 3D seismic volume matched to this 51^3 in the output dir, 
# but we can just plot the attributes.
# Actually, I can load the volume_3d from the script if I run it again, 
# or just plot the attributes directly as requested.

# Sub-volume indices: 400-450, 600-650.
# Inline 425 is index 25.
# Xline 625 is index 25.

def save_plots(data, cmap, title, name):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join(figures_dir, name))
    plt.close()

# Inline 425
idx = 25
save_plots(vol_seg[idx].T, 'jet', 'Segmentation Map - Inline 425', 'segmentation_inline_425.png')
save_plots(vol_mag[idx].T, 'viridis', 'Structure Tensor Magnitude - Inline 425', 'vector_mag_inline_425.png')
save_plots(vol_orient[idx].T, 'twilight', 'Vector Orientation - Inline 425', 'vector_orientation_inline_425.png')

# Xline 625
idx = 25
save_plots(vol_seg[:, idx].T, 'jet', 'Segmentation Map - Xline 625', 'segmentation_xline_625.png')
save_plots(vol_mag[:, idx].T, 'viridis', 'Structure Tensor Magnitude - Xline 625', 'vector_mag_xline_625.png')
save_plots(vol_orient[:, idx].T, 'twilight', 'Vector Orientation - Xline 625', 'vector_orientation_xline_625.png')

print("QC plots generated in figures/")
