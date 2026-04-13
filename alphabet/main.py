import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

def calculate_voids(reg):
    map_data = reg.image
    canvas = np.pad(map_data, pad_width=1, mode='constant', constant_values=0)
    inverted_view = np.logical_not(canvas)
    areas = label(inverted_view)
    return np.max(areas) - 1

def measure_balance(reg):
    pixels = reg.image.astype(float)
    r, c = pixels.shape
    if c < 3:
        return 0.0
    center = c // 2
    left_side = pixels[:, :center]
    right_side = np.fliplr(pixels[:, c - center:])
    mismatch = np.abs(left_side - right_side).mean()
    return 1.0 - mismatch

def get_descriptor(reg):
    y_rel, x_rel = reg.centroid_local
    y_rel /= reg.image.shape[0]
    x_rel /= reg.image.shape[1]
    
    bound_val = reg.perimeter / reg.image.size
    filling = reg.area / reg.image.size
    
    v_const = (np.all(reg.image, axis=0)).sum() / reg.image.shape[1]
    h_const = (np.all(reg.image, axis=1)).sum() / reg.image.shape[0]
    
    h, w = reg.image.shape
    form_factor = min(h, w) / max(h, w)
    
    return np.array([
        filling, x_rel, y_rel, bound_val, 
        calculate_voids(reg), v_const, h_const, 
        reg.eccentricity, form_factor, measure_balance(reg)
    ])

def predict_class(reg, dataset):
    current_feats = get_descriptor(reg)
    selection = None
    min_err = float('inf')
    
    for label_id, ref_feats in dataset.items():
        err = np.linalg.norm(current_feats - ref_feats)
        if err < min_err:
            min_err = err
            selection = label_id
    return selection

out_path = Path("output_data")
out_path.mkdir(exist_ok=True)

ref_img = imread("PClook/alphabet/alphabet_ext.png")
if ref_img.ndim == 3:
    binary_ref = ref_img[..., :3].sum(axis=2) < (255 * 3 * 0.9)
else:
    binary_ref = ref_img < 128

ref_labels = label(binary_ref)
ref_objects = regionprops(ref_labels)

char_map = ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-"]
brain = {name: get_descriptor(r) for r, name in zip(ref_objects, char_map)}

main_img = imread("PClook/alphabet/symbols.png")
if main_img.ndim == 3:
    binary_main = main_img[..., :3].mean(axis=2) > 0
else:
    binary_main = main_img > 0

target_labels = label(binary_main)
target_objects = regionprops(target_labels)

stats = {}

for idx, item in enumerate(target_objects):
    char_res = predict_class(item, brain)
    stats[char_res] = stats.get(char_res, 0) + 1
    
    if idx < 15:
        plt.figure()
        plt.imshow(item.image)
        plt.title(char_res)
        plt.savefig(out_path / f"res_{idx}.png")
        plt.close()

for key in sorted(stats.keys()):
    print(f"Символ {key}: {stats[key]}")