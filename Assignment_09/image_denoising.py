import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in the current directory.")

img_float = img.astype(np.float64) / 255.0
ksize = 5  # Window size: 3 or 5 (using 5x5)

# 1. Add Gaussian Noise
rng = np.random.default_rng(42)
gaussian_noisy = np.clip(img_float + rng.normal(0, 0.08, img_float.shape), 0, 1)
gaussian_noisy_u8 = (gaussian_noisy * 255).astype(np.uint8)

# 2. Add Speckle Noise (J = I + n * I)
speckle_noisy = np.clip(img_float + img_float * rng.normal(0, 0.25, img_float.shape), 0, 1)
speckle_noisy_u8 = (speckle_noisy * 255).astype(np.uint8)

# Wiener Filter implementation (local adaptive spatial Wiener filter)
def wiener_filter(image_u8, window_size=5):
    img_f = image_u8.astype(np.float64)
    local_mean = cv2.boxFilter(img_f, -1, (window_size, window_size))
    local_sq_mean = cv2.boxFilter(img_f ** 2, -1, (window_size, window_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    noise_var = np.mean(local_var)  # Estimate noise variance as average local variance
    
    result = local_mean + (np.maximum(local_var - noise_var, 0) / np.maximum(local_var, noise_var + 1e-8)) * (img_f - local_mean)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_all_filters(noisy_u8, window_size=5):
    wiener = wiener_filter(noisy_u8, window_size)
    median = cv2.medianBlur(noisy_u8, window_size)
    gaussian = cv2.GaussianBlur(noisy_u8, (window_size, window_size), 0)
    bilateral = cv2.bilateralFilter(noisy_u8, d=window_size, sigmaColor=75, sigmaSpace=75)
    return wiener, median, gaussian, bilateral

# Denoise Gaussian-corrupted image
g_wiener, g_median, g_gauss, g_bilat = apply_all_filters(gaussian_noisy_u8, ksize)

# Denoise Speckle-corrupted image
s_wiener, s_median, s_gauss, s_bilat = apply_all_filters(speckle_noisy_u8, ksize)

# Save individual outputs
cv2.imwrite("gaussian_noisy.jpg", gaussian_noisy_u8)
cv2.imwrite("gaussian_denoised_wiener.jpg", g_wiener)
cv2.imwrite("gaussian_denoised_median.jpg", g_median)
cv2.imwrite("gaussian_denoised_gaussian.jpg", g_gauss)
cv2.imwrite("gaussian_denoised_bilateral.jpg", g_bilat)

cv2.imwrite("speckle_noisy.jpg", speckle_noisy_u8)
cv2.imwrite("speckle_denoised_wiener.jpg", s_wiener)
cv2.imwrite("speckle_denoised_median.jpg", s_median)
cv2.imwrite("speckle_denoised_gaussian.jpg", s_gauss)
cv2.imwrite("speckle_denoised_bilateral.jpg", s_bilat)

# Display and save full comparison grid
fig, axes = plt.subplots(2, 6, figsize=(18, 6))

row1_imgs = [img, gaussian_noisy_u8, g_wiener, g_median, g_gauss, g_bilat]
row1_titles = ["Original", "Gaussian Noise", "Wiener (5x5)", "Median (5x5)", "Gaussian (5x5)", "Bilateral (5x5)"]

row2_imgs = [img, speckle_noisy_u8, s_wiener, s_median, s_gauss, s_bilat]
row2_titles = ["Original", "Speckle Noise", "Wiener (5x5)", "Median (5x5)", "Gaussian (5x5)", "Bilateral (5x5)"]

for col in range(6):
    axes[0, col].imshow(row1_imgs[col], cmap="gray")
    axes[0, col].set_title(row1_titles[col], fontsize=10)
    axes[0, col].axis("off")

    axes[1, col].imshow(row2_imgs[col], cmap="gray")
    axes[1, col].set_title(row2_titles[col], fontsize=10)
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig("denoising_comparison.png", dpi=300)
plt.show()
