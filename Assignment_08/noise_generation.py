import cv2
import numpy as np
import matplotlib.pyplot as plt

# Put input.jpg in the same folder
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("input.jpg was not found.")

image = image.astype(np.float32)
rows, cols = image.shape
n = rows * cols

np.random.seed(42)

# 1. Uniform noise: z = a + (b - a)U
a, b = -30, 30
u = np.random.rand(n)
uniform_manual = a + (b - a) * u
uniform_builtin = np.random.uniform(a, b, n)

# 2. Gaussian noise using Box-Muller equation
mean, sigma = 0, 20
u1 = np.random.rand(n)
u2 = np.random.rand(n)
gaussian_manual = mean + sigma * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
gaussian_builtin = np.random.normal(mean, sigma, n)

# 3. Erlang noise: z = -b * ln(U1 × U2 × ... × Uk)
k, scale = 3, 10
u = np.random.rand(k, n)
erlang_manual = -scale * np.log(np.prod(u, axis=0))
erlang_builtin = np.random.gamma(k, scale, n)

# 4. Exponential noise: z = -(1/lambda) ln(U)
lam = 0.05
u = np.random.rand(n)
exponential_manual = -np.log(u) / lam
exponential_builtin = np.random.exponential(1 / lam, n)

# 5. Rayleigh noise: z = sigma * sqrt(-2 ln(U))
rayleigh_sigma = 15
u = np.random.rand(n)
rayleigh_manual = rayleigh_sigma * np.sqrt(-2 * np.log(u))
rayleigh_builtin = np.random.rayleigh(rayleigh_sigma, n)

noises = [
    ("Uniform Noise", uniform_manual, uniform_builtin),
    ("Gaussian Noise", gaussian_manual, gaussian_builtin),
    ("Erlang Noise", erlang_manual, erlang_builtin),
    ("Exponential Noise", exponential_manual, exponential_builtin),
    ("Rayleigh Noise", rayleigh_manual, rayleigh_builtin)
]

# Plot manual and built-in noise histograms
plt.figure(figsize=(12, 18))

for i, (name, manual, builtin) in enumerate(noises):
    plt.subplot(5, 2, 2 * i + 1)
    plt.hist(manual, bins=100, color="blue", alpha=0.7)
    plt.title(name + " - Equation Method")
    plt.xlabel("Noise value")
    plt.ylabel("Frequency")

    plt.subplot(5, 2, 2 * i + 2)
    plt.hist(builtin, bins=100, color="red", alpha=0.7)
    plt.title(name + " - Built-in Method")
    plt.xlabel("Noise value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("noise_histograms.png")
plt.show()

# Add manually generated noise to the input image and save outputs
for name, manual, builtin in noises:
    noisy_image = image + manual.reshape(rows, cols)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    filename = name.lower().replace(" ", "_") + ".jpg"
    cv2.imwrite(filename, noisy_image)

print("All noise images and histograms have been saved.")
