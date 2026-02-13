import os
import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from flow_utils import *

print_banner()

# Utility functions moved to flow_utils.py

def segment_flowlines(seismic_line, sample_interval_x, mode, num_clusters=None, only_y=False, cmap=plt.cm.jet):

    S = structure_tensor_2d(seismic_line, sigma=1.0, rho=1.0)
    _, vector_array = eig_special_2d(S)

    peak_params = {
        "height": None,
        "distance": None,
        "prominence": None
    }

    surfaces_seismic, _ = extract_surfaces(
        seismic_line,

        vector_array,

        [sample_interval_x],

        mode=mode,

        kwargs=peak_params
    )
    array_mask = np.zeros(seismic_line.shape)
    if num_clusters:
        scaler = StandardScaler()
        X, _ = surface_to_feature_vector(surfaces_seismic, only_y=only_y)
        clusterer = KMeans(n_clusters=num_clusters, random_state=0,
                           n_init="auto").fit(scaler.fit_transform(X))

        #scalar_map = create_scalar_colormap(np.arange(num_clusters),cmap=cmap)
        for ind, surface in enumerate(surfaces_seismic):
            surface.kmeans_label = clusterer.labels_[ind]

        #sorted_labels
        centroids = clusterer.cluster_centers_

        # Step 2: Sort centroids based on the y-coordinate (or your chosen criterion)
        sorted_indices = np.argsort(centroids[:, 1])  # Assuming y-coordinate is at index 1
        sorted_indices = np.append(sorted_indices, num_clusters + 1)
        # Step 3: Create mapping from old to new labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}

        # Step 4: Reassign labels to surfaces
        for surface in surfaces_seismic:
            surface.kmeans_label = label_mapping[surface.kmeans_label]
        
        label_map_back = {value:key for key,value in label_mapping.items()}

        sorted_tops = sort_tops(seismic_line, surfaces_seismic, clusterer)
        sorted_tops_surfaces = [Surface(tops) for tops in sorted_tops]
        interp_tops = [Surface(interpolate_boundary(top.path, seismic_line.shape)) for top in sorted_tops_surfaces]

        _, Y = np.meshgrid(np.arange(seismic_line.shape[1]), np.arange(seismic_line.shape[0]))
        

        for i in range(len(interp_tops) - 1):
            
            y1=interp_tops[i].path[:, 0]
            y2=interp_tops[i+1].path[:, 0]


            array_mask[np.where((Y > y1)&(Y < y2))] = label_map_back[i]

    return array_mask
        
   

# Path to 2D seismic
seispath = 'F3_2Dline.npy'
seismic_line = np.load(seispath)

# the distance between traces to sample peaks/troughs from
sample_interval_x = 100  

# if we are picking on both peak and trough in each trace
mode = 'both'  


#only_y : if we use both the x and y coordinate of the surface paths. Defaults to both. 
only_y = False

#! Clustering parameters
# number of clusters to use for clustering.
num_clusters = 10  


segmentation_map = segment_flowlines(seismic_line,
                                sample_interval_x=sample_interval_x, mode=mode, only_y=only_y, num_clusters=num_clusters)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)
fig,ax = plt.subplots(1,1,dpi=250)
ax.imshow(segmentation_map,cmap='jet')
plt.savefig('output/segmentation_map.png')
print(f"Result saved to output/segmentation_map.png")