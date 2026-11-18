import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load input image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in the current directory.")

# Apply Otsu's global thresholding
optimal_thresh, segmented_img = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"Optimal Otsu Global Threshold Value: {optimal_thresh}")

# Save segmented output image
cv2.imwrite("otsu_segmented.jpg", segmented_img)

# Display and save comparison with histogram showing threshold line
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original Grayscale Image")
axes[0].axis("off")

axes[1].hist(img.ravel(), bins=256, range=(0, 256), color="gray")
axes[1].axvline(optimal_thresh, color="red", linestyle="--", linewidth=2, label=f"Threshold = {optimal_thresh:.0f}")
axes[1].set_title("Grayscale Histogram & Otsu Threshold")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")
axes[1].set_xlim([0, 255])
axes[1].legend()

axes[2].imshow(segmented_img, cmap="gray")
axes[2].set_title(f"Otsu Segmented Output (T = {optimal_thresh:.0f})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("otsu_comparison.png", dpi=300)
plt.show()
