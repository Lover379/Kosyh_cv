import numpy as np
from skimage import io, color, measure

img = io.imread("PClook/figures_and_colors/balls_and_rects.png")
hsv = color.rgb2hsv(img)
regions = measure.regionprops(measure.label(hsv[:, :, 2] > 0))

circles = {}
rects = {}

for r in regions:
    h = round(hsv[int(r.centroid[0]), int(r.centroid[1]), 0], 2)
    
    if (r.perimeter**2) / r.area < 14.5:
        circles[h] = circles.get(h, 0) + 1
    else:
        rects[h] = rects.get(h, 0) + 1

print(len(regions))
print("Круги:")
for h in sorted(circles):
    print(f"{h:.6f} {circles[h]}")
print("Прямоугольники:")
for h in sorted(rects):
    print(f"{h:.6f} {rects[h]}")