import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_centers(idx):
    img = np.load(f'PClook/trajectory/out/h_{idx}.npy')
    lbl, n = ndimage.label(img)
    return np.array(ndimage.center_of_mass(img, lbl, range(1, n + 1)))

data0 = get_centers(0)
tracks = [[p] for p in data0[data0[:, 0].argsort()]]

for i in range(1, 100):
    points = get_centers(i)
    for t in tracks:
        dists = np.hypot(*(points - t[-1]).T)
        t.append(points[np.argmin(dists)])

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('#f5f5f0')

colors = plt.cm.inferno(np.linspace(0.3, 0.9, len(tracks)))

for idx, path in enumerate(tracks):
    pts = np.array(path)
    ax.plot(pts[:, 1], pts[:, 0], c=colors[idx], lw=1.8, 
            alpha=0.8, marker='o', ms=2, markevery=5)
    ax.scatter(pts[0, 1], pts[0, 0], c=[colors[idx]], s=45, 
               edgecolor='darkblue', lw=0.8, zorder=3)
    ax.scatter(pts[-1, 1], pts[-1, 0], c=[colors[idx]], s=70, 
               marker='^', edgecolor='black', lw=0.5, zorder=3)

ax.set_title('Particle Trajectories', fontsize=15, pad=15, style='italic')
ax.set_xlabel('X position', fontsize=11)
ax.set_ylabel('Y position', fontsize=11)
ax.invert_yaxis()
ax.grid(True, alpha=0.4, linestyle='-.', linewidth=0.7)
ax.tick_params(axis='both', which='major', labelsize=9)

plt.tight_layout()
plt.show()