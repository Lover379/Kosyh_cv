import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

stars = np.load("PClook/stars/stars.npy")
img = stars > 0

labeled = label(img)
reg = regionprops(labeled)

count = 0

for r in reg:
    area = r.area
    bbox_area = r.bbox_area

    den = area/bbox_area
    if den < 0.6:
        count += 1

print(count)