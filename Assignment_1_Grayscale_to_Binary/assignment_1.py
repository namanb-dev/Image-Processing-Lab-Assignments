import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Assignment 1: Grayscale to Binary")
    image_path = input("Enter image path (default: /content/image.png): ") or "/content/image.png"
    
    # 1. Load a grayscale image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image could not be loaded. Check the file path.")
        return

    # 3. Calculate the mean intensity
    mean_intensity = np.mean(img)
    print(f"Mean intensity of the image: {mean_intensity:.2f}")

    # 4. CASE 1: Convert using mean intensity as threshold
    threshold_case1 = mean_intensity
    # Manual thresholding
    binary_case1 = np.zeros_like(img)
    binary_case1[img >= threshold_case1] = 255
    
    # 6. CASE 2: Input threshold from user
    while True:
        try:
            threshold_case2 = float(input("Enter a threshold value (0-255): "))
            if 0 <= threshold_case2 <= 255:
                break
            else:
                print("Threshold must be between 0 and 255.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # 8. Convert using user threshold
    binary_case2 = np.zeros_like(img)
    binary_case2[img >= threshold_case2] = 255

    # Displaying results
    plt.figure(figsize=(15, 5))
    
    # 2. Display original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Grayscale Image")
    plt.axis("off")
    
    # 5. Display CASE 1
    plt.subplot(1, 3, 2)
    plt.imshow(binary_case1, cmap="gray")
    plt.title(f"Case 1: Mean Threshold ({threshold_case1:.2f})")
    plt.axis("off")
    
    # 9. Display CASE 2
    plt.subplot(1, 3, 3)
    plt.imshow(binary_case2, cmap="gray")
    plt.title(f"Case 2: User Threshold ({threshold_case2})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
