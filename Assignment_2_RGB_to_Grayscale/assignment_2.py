import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Assignment 2: RGB to Grayscale")
    image_path = input("Enter image path (default: /content/image.png): ") or "/content/image.png"
    
    # 1. Loading RGB color image (OpenCV reads in BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Image could not be loaded. Check the file path.")
        return
        
    # 3. Converting BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Separating planes
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    
    # 5. CASE 1: Mean average
    gray_case1 = (R + G + B) / 3.0
    gray_case1 = np.clip(gray_case1, 0, 255).astype(np.uint8)
    
    # 7. CASE 2: Input weights
    while True:
        try:
            print("\nEnter weights for R, G, B (must be 0-1 and sum to 1):")
            wR = float(input("Weight for R (e.g., 0.7): "))
            wG = float(input("Weight for G (e.g., 0.2): "))
            wB = float(input("Weight for B (e.g., 0.1): "))
            
            # 8. Validating
            if 0 <= wR <= 1 and 0 <= wG <= 1 and 0 <= wB <= 1:
                if abs((wR + wG + wB) - 1.0) < 1e-5:
                    break
                else:
                    print(f"Weights sum to {wR+wG+wB}, but must sum exactly to 1.")
            else:
                print("Weights must be between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
            
    print(f"\nEntered Weights -> R: {wR}, G: {wG}, B: {wB}")
    
    # 9. Using weighted formula
    gray_case2 = wR * R + wG * G + wB * B
    gray_case2 = np.clip(gray_case2, 0, 255).astype(np.uint8)
    
    # 6. Displaying original, planes, and grayscale result (Case 1)
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB Image")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    # Displaying R plane as grayscale to show intensity
    plt.imshow(R.astype(np.uint8), cmap="gray")
    plt.title("R Plane")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(G.astype(np.uint8), cmap="gray")
    plt.title("G Plane")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    plt.imshow(B.astype(np.uint8), cmap="gray")
    plt.title("B Plane")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(gray_case1, cmap="gray")
    plt.title("Case 1: Mean Average Grayscale")
    plt.axis("off")
    
    # 10. Displaying weighted grayscale result (Case 2)
    plt.subplot(2, 3, 6)
    plt.imshow(gray_case2, cmap="gray")
    plt.title(f"Case 2: Weighted (R={wR}, G={wG}, B={wB})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
