import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in the current directory.")

# Pad image to handle border pixels
padded = np.pad(img, pad_width=1, mode="edge")
center = padded[1:-1, 1:-1]

# Compute 8-neighbor Local Binary Pattern (clockwise starting top-left)
lbp = np.zeros_like(img, dtype=np.uint8)
neighbors = [
    (padded[0:-2, 0:-2], 128),  # Top-left
    (padded[0:-2, 1:-1], 64),   # Top
    (padded[0:-2, 2:],   32),   # Top-right
    (padded[1:-1, 2:],   16),   # Right
    (padded[2:,   2:],   8),    # Bottom-right
    (padded[2:,   1:-1], 4),    # Bottom
    (padded[2:,   0:-2], 2),    # Bottom-left
    (padded[1:-1, 0:-2], 1),    # Left
]

for neighbor, weight in neighbors:
    lbp += ((neighbor >= center).astype(np.uint8) * weight)

# Save LBP output image
cv2.imwrite("lbp_output.jpg", lbp)

# Compute LBP histogram
hist, bins = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

# Display and save comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original Grayscale Image")
axes[0].axis("off")

axes[1].imshow(lbp, cmap="gray")
axes[1].set_title("LBP Image")
axes[1].axis("off")

axes[2].bar(np.arange(256), hist, width=1.0, color="black")
axes[2].set_title("LBP Histogram")
axes[2].set_xlabel("LBP Value")
axes[2].set_ylabel("Frequency")
axes[2].set_xlim([0, 255])

plt.tight_layout()
plt.savefig("lbp_comparison.png", dpi=300)
plt.show()
