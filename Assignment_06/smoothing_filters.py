import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_padding(image, size, method):
    r = size // 2

    if method == "replication":
        return np.pad(image, r, mode="edge")
    elif method == "zero":
        return np.pad(image, r, mode="constant", constant_values=0)
    else:
        raise ValueError("Padding must be 'replication' or 'zero'")

def apply_filter(image, kernel, padding):
    size = kernel.shape[0]
    padded = get_padding(image, size, padding)
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+size, j:j+size]
            result[i, j] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)

# Input image
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# User inputs
filter_size = int(input("Enter odd filter size (3, 5, 7): "))
padding_method = input("Enter padding method (replication/zero): ").lower()

if filter_size % 2 == 0:
    raise ValueError("Filter size must be odd.")

# 1. Unweighted averaging filter
average_kernel = np.ones((filter_size, filter_size), dtype=np.float32)
average_kernel /= filter_size * filter_size

averaged_image = apply_filter(image, average_kernel, padding_method)

# 2. Weighted filter: h(x,y) = max(|x|, |y|)
r = filter_size // 2
weighted_kernel = np.zeros((filter_size, filter_size), dtype=np.float32)

for x in range(-r, r + 1):
    for y in range(-r, r + 1):
        weighted_kernel[x + r, y + r] = max(abs(x), abs(y))

# Give the center pixel a positive weight
weighted_kernel[r, r] = 1
weighted_kernel /= np.sum(weighted_kernel)

weighted_image = apply_filter(image, weighted_kernel, padding_method)

# 3. Gaussian filter
sigma = float(input("Enter Gaussian sigma: "))

x = np.arange(-r, r + 1)
y = np.arange(-r, r + 1)
X, Y = np.meshgrid(x, y)

gaussian_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
gaussian_kernel /= np.sum(gaussian_kernel)

gaussian_image = apply_filter(image, gaussian_kernel, padding_method)

# Save results
cv2.imwrite("averaging_filter.jpg", averaged_image)
cv2.imwrite("weighted_filter.jpg", weighted_image)
cv2.imwrite("gaussian_filter.jpg", gaussian_image)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(averaged_image, cmap="gray")
plt.title("Averaging Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(weighted_image, cmap="gray")
plt.title("Weighted Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gaussian_image, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
