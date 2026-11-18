import cv2
import numpy as np
import matplotlib.pyplot as plt

# Put input.jpg in the same Assignment_7 folder
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("input.jpg was not found.")

image_float = image.astype(np.float32)

# 1. Laplacian sharpening
laplacian = cv2.Laplacian(image_float, cv2.CV_32F, ksize=3)
laplacian_sharpened = np.clip(image_float - laplacian, 0, 255).astype(np.uint8)

# 2. Sobel gradient sharpening
sobel_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
sobel_gradient = cv2.magnitude(sobel_x, sobel_y)

sobel_sharpened = np.clip(image_float + sobel_gradient, 0, 255).astype(np.uint8)

# Canny edge detector
canny_edges = cv2.Canny(image, 100, 200)

# 3. Unsharp masking: Gaussian filter size 5x5, sigma = 3
blurred = cv2.GaussianBlur(image, (5, 5), 3)
mask = image_float - blurred.astype(np.float32)
unsharp_image = np.clip(image_float + mask, 0, 255).astype(np.uint8)

# Save output images
cv2.imwrite("laplacian_sharpened.jpg", laplacian_sharpened)
cv2.imwrite("sobel_sharpened.jpg", sobel_sharpened)
cv2.imwrite("canny_edges.jpg", canny_edges)
cv2.imwrite("unsharp_masked.jpg", unsharp_image)

# Display all results
plt.figure(figsize=(12, 8))

images = [
    (image, "Original Image"),
    (laplacian_sharpened, "Laplacian Sharpening"),
    (sobel_sharpened, "Sobel Sharpening"),
    (canny_edges, "Canny Edges"),
    (blurred, "Gaussian Blur (5x5, sigma=3)"),
    (unsharp_image, "Unsharp Masking")
]

for i, (img, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
