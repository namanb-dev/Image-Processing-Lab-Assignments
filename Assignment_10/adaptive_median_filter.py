import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original grayscale image
original = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError("input.jpg not found in the current directory.")

# Take window size inputs from user
s_init = int(input("Enter initial odd window size (e.g., 3): "))
s_max = int(input("Enter maximum odd window size Smax (e.g., 7): "))

if s_init < 3 or s_init % 2 == 0 or s_max < s_init or s_max % 2 == 0:
    raise ValueError("Window sizes must be odd integers >= 3 with Smax >= initial window size.")

# Add Salt & Pepper noise to create a noisy input image for denoising comparison
rng = np.random.default_rng(42)
noisy = original.copy()
prob = 0.25  # 25% impulse noise density
rand_vals = rng.random(original.shape)
noisy[rand_vals < prob / 2] = 0
noisy[rand_vals > 1 - prob / 2] = 255

def adaptive_median_filter(img, s_init, s_max):
    rows, cols = img.shape
    pad = s_max // 2
    padded = np.pad(img, pad_width=pad, mode="edge")
    output = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            window_size = s_init
            while window_size <= s_max:
                half = window_size // 2
                r_center, c_center = i + pad, j + pad
                window = padded[
                    r_center - half : r_center + half + 1,
                    c_center - half : c_center + half + 1,
                ]

                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = padded[r_center, c_center]

                # Level A
                a1 = int(z_med) - int(z_min)
                a2 = int(z_med) - int(z_max)
                if a1 > 0 and a2 < 0:
                    # Level B
                    b1 = int(z_xy) - int(z_min)
                    b2 = int(z_xy) - int(z_max)
                    if b1 > 0 and b2 < 0:
                        output[i, j] = z_xy
                    else:
                        output[i, j] = z_med
                    break
                else:
                    window_size += 2
                    if window_size > s_max:
                        output[i, j] = z_med
                        break
    return output

# Apply standard median filter (using initial window size) and adaptive median filter
median_filtered = cv2.medianBlur(noisy, s_init)
adaptive_filtered = adaptive_median_filter(noisy, s_init, s_max)

# Compute PSNR values against original clean image
psnr_noisy = cv2.PSNR(original, noisy)
psnr_median = cv2.PSNR(original, median_filtered)
psnr_adaptive = cv2.PSNR(original, adaptive_filtered)

print(f"PSNR (Noisy Image):            {psnr_noisy:.2f} dB")
print(f"PSNR (Median Filter {s_init}x{s_init}):      {psnr_median:.2f} dB")
print(f"PSNR (Adaptive Median Smax={s_max}): {psnr_adaptive:.2f} dB")

# Save output images
cv2.imwrite("noisy_input.jpg", noisy)
cv2.imwrite("standard_median_output.jpg", median_filtered)
cv2.imwrite("adaptive_median_output.jpg", adaptive_filtered)

# Display and save visual comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

titles = [
    "Original Image",
    f"Noisy Image\nPSNR: {psnr_noisy:.2f} dB",
    f"Median Filter ({s_init}x{s_init})\nPSNR: {psnr_median:.2f} dB",
    f"Adaptive Median (Smax={s_max})\nPSNR: {psnr_adaptive:.2f} dB",
]
images = [original, noisy, median_filtered, adaptive_filtered]

for ax, title, image in zip(axes, titles, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("adaptive_median_comparison.png", dpi=300)
plt.show()
