import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_neighbor_interpolation(img, new_h, new_w):
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3
    
    if is_color:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
    else:
        out = np.zeros((new_h, new_w), dtype=np.uint8)
        
    scale_y = h / new_h
    scale_x = w / new_w
    
    for i in range(new_h):
        for j in range(new_w):
            orig_y = min(int(i * scale_y), h - 1)
            orig_x = min(int(j * scale_x), w - 1)
            
            if is_color:
                out[i, j, :] = img[orig_y, orig_x, :]
            else:
                out[i, j] = img[orig_y, orig_x]
                
    return out

def bilinear_interpolation(img, new_h, new_w):
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3
    
    if is_color:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
    else:
        out = np.zeros((new_h, new_w), dtype=np.uint8)
        
    scale_y = h / new_h
    scale_x = w / new_w
    
    for i in range(new_h):
        for j in range(new_w):
            y = i * scale_y
            x = j * scale_x
            
            y_int = int(y)
            x_int = int(x)
            
            a = x - x_int
            b = y - y_int
            
            y_next = min(y_int + 1, h - 1)
            x_next = min(x_int + 1, w - 1)
            
            if is_color:
                for c in range(img.shape[2]):
                    P00 = img[y_int, x_int, c]
                    P10 = img[y_int, x_next, c]
                    P01 = img[y_next, x_int, c]
                    P11 = img[y_next, x_next, c]
                    
                    val = (1-a)*(1-b)*P00 + a*(1-b)*P10 + (1-a)*b*P01 + a*b*P11
                    out[i, j, c] = int(val)
            else:
                P00 = img[y_int, x_int]
                P10 = img[y_int, x_next]
                P01 = img[y_next, x_int]
                P11 = img[y_next, x_next]
                
                val = (1-a)*(1-b)*P00 + a*(1-b)*P10 + (1-a)*b*P01 + a*b*P11
                out[i, j] = int(val)
                
    return out

def main():
    print("Assignment 3: Image Scaling")
    image_path = input("Enter image path (default: /content/image.png): ") or "/content/image.png"
    
    img = cv2.imread(image_path)
    if img is None:
        print("Image could not be loaded. Check the file path.")
        return
        
    # Converting BGR to RGB for correct plotting if color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    orig_h, orig_w = img.shape[:2]
    
    print("\nScaling Options:")
    print("1: Scale Factor")
    print("2: Desired Dimensions")
    
    choice = input("Enter 1 or 2: ")
    if choice == '1':
        while True:
            try:
                scale = float(input("Enter positive scale factor (e.g., 0.5 for shrink, 2.0 for zoom): "))
                if scale > 0:
                    break
                print("Scale must be > 0")
            except ValueError:
                print("Invalid input.")
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
    elif choice == '2':
        while True:
            try:
                new_w = int(input("Enter desired width (positive integer): "))
                new_h = int(input("Enter desired height (positive integer): "))
                if new_w > 0 and new_h > 0:
                    break
                print("Dimensions must be > 0")
            except ValueError:
                print("Invalid input.")
    else:
        print("Invalid choice. Exiting.")
        return
        
    print(f"\nOriginal dimensions: {orig_h}x{orig_w}")
    print(f"New dimensions: {new_h}x{new_w}")
    
    print("Processing Nearest Neighbor...")
    nn_img = nearest_neighbor_interpolation(img, new_h, new_w)
    
    print("Processing Bilinear...")
    bl_img = bilinear_interpolation(img, new_h, new_w)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(f"Original ({orig_h}x{orig_w})")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    if len(img.shape) == 3:
        plt.imshow(nn_img)
    else:
        plt.imshow(nn_img, cmap="gray")
    plt.title(f"Nearest Neighbor ({new_h}x{new_w})")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    if len(img.shape) == 3:
        plt.imshow(bl_img)
    else:
        plt.imshow(bl_img, cmap="gray")
    plt.title(f"Bilinear ({new_h}x{new_w})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
