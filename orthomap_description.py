#img_shape = (8078, 12717)#zone1
#img_shape = (7366, 9529) #zone3
#img_shape = (5807, 7070) #zone108
#img_shape = (6302, 4627) #zone109
#img_shape = (5757, 7033) #zone108_octubre
#img_shape = (11942, 7897, 3) #"zone102_part1"

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from pathlib import Path

def build_rag_document(tile):

    summary = tile["analysis"]["regional_summary"]

    observations = tile["analysis"]["observations"]

    parts = []

    parts.append(
        f"Region {tile['region_id']}"
    )

    parts.append(
        f"Condition: "
        f"{summary['condition_classification']}"
    )

    parts.append(
        f"Expected Similarity: "
        f"{summary['score_similarity_expected']}"
    )

    parts.append(
        f"Anomaly Similarity: "
        f"{summary['score_anomaly_condition']}"
    )

    for obs in observations:

        parts.append(
            f"Observation: "
            f"{obs['observation_name']}"
        )

        parts.append(
            obs["visual_description"]
        )

        parts.append(
            obs["scientific_reasoning"]
        )

    return "\n".join(parts)

root = 'E:/Experiments/'
folder_maps = root+'Datasets Maps/'

#zone_name="Zone3"
zone_name = "Zone102_part1"
#zone_name = "Zone108_octubre"

map_dir = folder_maps+zone_name+'_full.tif'

print(f"Reading map ...")                  
with rio.open(map_dir) as img :
    metadata = img.meta
    imgnp = img.read() 
print(f"Read map {map_dir}")

folder_jsons = "E:\\Experiments\\agents\\agave\\map_"+zone_name+"\\Descriptions"

#files = sorted(
#    [f for f in os.listdir(folder_jsons)
#     if f.endswith(".json")]
#)
#n_tiles = len(files)

#heatmap = np.zeros_like(imgnp[0,...])
heatmap = np.full((imgnp[0,...].shape[0], imgnp[0,...].shape[1]), np.nan)

img_size = 500

counter=1
for i in range(img_size, imgnp.shape[-2], img_size):
    for j in range(img_size, imgnp.shape[-1], img_size):
        im = imgnp[:,i-img_size:i,j-img_size:j]
        
        if np.count_nonzero(im[-1,:,:]==0) > 0.1*img_size*img_size: #if more than 10% of image is out of bg continue
            continue

        json_file = (Path(folder_jsons) / f"field_report_{format(counter, '04d')}_rgb.json")
        counter+=1

        with open(json_file) as f:
            data = json.load(f)

        region_id = data[0]["region_id"]

        summary = data[0]["analysis"]["regional_summary"]

        expected = summary["score_similarity_expected"]
        anomaly = summary["score_anomaly_condition"]

        score = anomaly - expected

        heatmap[i-img_size:i,j-img_size:j] = score

        #print(build_rag_document(data[0]))
        #break

#"""
plt.figure(figsize=(8,8))

masked = np.ma.masked_invalid(heatmap)

cmap = plt.cm.RdYlGn_r.copy()
cmap.set_bad(color='white', alpha=0)

plt.imshow(masked, cmap=cmap)
plt.colorbar(
    label="Deviation Score (Anomaly - Expected)"
)
plt.title("Orthomap Condition Heatmap")
plt.show()
#"""
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.figure(figsize=(8,8))

masked = np.ma.masked_invalid(heatmap)

cmap = plt.cm.RdYlGn_r.copy()
cmap.set_bad(color='white', alpha=0)

# Center colormap at 0
norm = TwoSlopeNorm(
    vmin=0,#-100,
    vcenter=30,
    vmax=100
)

im = plt.imshow(
    masked,
    cmap=cmap,
    norm=norm
)

cbar = plt.colorbar(im)
cbar.set_label(
    "Deviation Score (Anomaly Similarity - Expected Similarity)"
)

plt.title("Orthomap Condition Heatmap")

plt.show()
"""