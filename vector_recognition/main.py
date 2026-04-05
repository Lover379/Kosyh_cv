import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

current_dir = Path(__file__).parent

def read_and_convert(file_path, need_flip=False):
    picture = imread(file_path)
    if picture.ndim == 3:
        picture = picture[..., :3].mean(axis=2)
    return picture < 128 if need_flip else picture > 0

def calculate_empty_spaces(obj):
    padded = np.pad(obj.image, 1)
    labeled = label(~padded)
    return labeled.max() - 1

def get_features(obj):
    shape = obj.image.shape
    height, width = shape
    y_center, x_center = obj.centroid_local
    
    features = np.array([
        obj.area / (height * width),
        y_center / height,
        x_center / width,
        obj.perimeter / obj.area if obj.area else 0,
        calculate_empty_spaces(obj),
        (obj.image.sum(axis=1) == width).sum(),
        (obj.image.sum(axis=0) == height).sum(),
        obj.eccentricity,
        height / width
    ])
    return features

def recognize_symbol(segment, reference_dict):
    current_features = get_features(segment)
    distances = {}
    for key, ref_features in reference_dict.items():
        distance = np.linalg.norm(ref_features - current_features)
        distances[key] = distance
    best_match = min(distances, key=distances.get)
    return best_match, distances[best_match]

alphabet_chars = ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/"]

reference_img = read_and_convert("PClook/vector_recognition/alphabet.png")
reference_objects = regionprops(label(reference_img))

reference_features = {}
for character, obj in zip(alphabet_chars, reference_objects):
    reference_features[character] = get_features(obj)

main_img = read_and_convert("PClook/vector_recognition/alphabet.png")
all_objects = regionprops(label(main_img))

plt.switch_backend("Agg")

statistics = {}
distance_log = []

for idx, obj in enumerate(all_objects, 1):
    recognized_char, min_distance = recognize_symbol(obj, reference_features)
    statistics[recognized_char] = statistics.get(recognized_char, 0) + 1
    distance_log.append((idx, recognized_char, min_distance))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(obj.image, cmap="gray")
    plt.title(f"{recognized_char} (dist: {min_distance:.3f})")
    plt.axis("off")
    plt.savefig(current_dir / f"object_{idx}.png", bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ")
print("="*50)
print(f"{'Символ':<10} {'Количество':<15} {'Процент':<10}")
print("-"*50)
for symbol in sorted(statistics):
    percentage = (statistics[symbol] / len(all_objects)) * 100
    print(f"{symbol:<10} {statistics[symbol]:<15} {percentage:.1f}%")

total_objects = len(all_objects)
not_recognized = statistics.get("?", 0)
accuracy = (1 - not_recognized/total_objects) * 100

print("-"*50)
print(f"\nВсего объектов: {total_objects}")
print(f"Не распознано: {not_recognized}")
print(f"Точность: {accuracy:.2f}%")

print("\n" + "="*50)
print("МИНИМАЛЬНЫЕ РАССТОЯНИЯ ДЛЯ КАЖДОГО ОБЪЕКТА")
print("="*50)
for idx, char, dist in distance_log[:10]:
    print(f"Объект {idx}: {char} (расстояние = {dist:.4f})")
if len(distance_log) > 10:
    print(f"... и еще {len(distance_log) - 10} объектов")

plt.figure(figsize=(12, 8))
plt.imshow(main_img, cmap="gray")
plt.title(f"Всего символов: {total_objects}, Точность: {accuracy:.2f}%")
plt.axis("off")
plt.savefig(current_dir / "total_result.png", bbox_inches='tight')
plt.close()

print(f"\nРезультаты сохранены в: {current_dir}")
print("- Изображения каждого символа: object_X.png")
print("- Общее изображение: total_result.png")