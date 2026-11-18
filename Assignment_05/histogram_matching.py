import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)

input_hist = cv2.calcHist([input_image], [0], None, [256], [0, 256]).ravel()
reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256]).ravel()

input_cdf = np.cumsum(input_hist)
reference_cdf = np.cumsum(reference_hist)

input_cdf /= input_cdf[-1]
reference_cdf /= reference_cdf[-1]

lookup_table = np.zeros(256, dtype=np.uint8)

for intensity in range(256):
    lookup_table[intensity] = np.argmin(
        np.abs(reference_cdf - input_cdf[intensity])
    )

matched_image = lookup_table[input_image]

cv2.imwrite("matched_image.jpg", matched_image)

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(reference_image, cmap="gray")
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(matched_image, cmap="gray")
plt.title("Matched Image")
plt.axis("off")

plt.show()
