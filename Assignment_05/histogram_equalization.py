import cv2
import matplotlib.pyplot as plt

image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

equalized = cv2.equalizeHist(image)

cv2.imwrite("equalized_image.jpg", equalized)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.show()
