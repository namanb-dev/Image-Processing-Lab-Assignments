import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_border(img, border_width, border_val):
    h, w = img.shape
    
    # Creating left and right border arrays
    left_border = np.full((h, border_width), border_val, dtype=np.uint8)
    right_border = np.full((h, border_width), border_val, dtype=np.uint8)
    
    # Combining horizontally: left + image + right
    middle = np.hstack((left_border, img, right_border))
    
    # Creating top and bottom border arrays for the new width
    new_w = w + 2 * border_width
    top_border = np.full((border_width, new_w), border_val, dtype=np.uint8)
    bottom_border = np.full((border_width, new_w), border_val, dtype=np.uint8)
    
    # Combining vertically: top + middle + bottom
    padded_img = np.vstack((top_border, middle, bottom_border))
    
    return padded_img

def main():
    print("Assignment 4: Image Border/Padding")
    image_path = input("Enter image path (default: /content/image.png): ") or "/content/image.png"
    
    # Load as grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Image could not be loaded. Check the file path.")
        return
        
    orig_h, orig_w = img_gray.shape
    
    # Input border width
    while True:
        try:
            border_width = int(input("Enter border width in pixels: "))
            if border_width >= 0:
                break
            print("Border width cannot be negative.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
            
    # Input border color (grayscale)
    while True:
        try:
            border_val_gray = int(input("Enter desired border intensity (0-255): "))
            if 0 <= border_val_gray <= 255:
                break
            print("Intensity must be between 0 and 255.")
        except ValueError:
            print("Invalid input.")
            
    # Process Grayscale
    padded_gray = add_border(img_gray, border_width, border_val_gray)
    
    # Converting original to binary (using simple threshold for demonstration)
    # Using 127 as default threshold
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Input border color (binary)
    while True:
        try:
            border_val_bin = int(input("Enter border value for binary image (0 or 255): "))
            if border_val_bin in [0, 255]:
                break
            print("Binary border value must be 0 or 255.")
        except ValueError:
            print("Invalid input.")
            
    # Process Binary
    padded_binary = add_border(img_binary, border_width, border_val_bin)
    
    print(f"\nOriginal dimensions: {orig_h}x{orig_w}")
    print(f"Border width: {border_width}")
    print(f"Final dimensions: {padded_gray.shape[0]}x{padded_gray.shape[1]}")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.title(f"Original Grayscale ({orig_h}x{orig_w})")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(padded_gray, cmap="gray")
    plt.title(f"Padded Grayscale (Val={border_val_gray})")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_binary, cmap="gray")
    plt.title("Original Binary Image")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(padded_binary, cmap="gray")
    plt.title(f"Padded Binary (Val={border_val_bin})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
