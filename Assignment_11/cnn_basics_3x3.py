import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Initialize a random 3x3 matrix
matrix = np.random.randint(1, 10, size=(3, 3)).astype(np.float32)
print("Original 3x3 Matrix:\n", matrix)

# 2. Zero Padding (P = 1) -> turns 3x3 into 5x5
padded = np.pad(matrix, pad_width=1, mode="constant", constant_values=0)
print("\nPadded Matrix (P=1, 5x5):\n", padded)

# 3. 2D Convolution (3x3 kernel, stride=1 on padded 5x5 -> 3x3 output)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)

conv_out = np.zeros((3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        conv_out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
print("\nConvolution Output (3x3):\n", conv_out)

# 4. Activation Functions (ReLU & Sigmoid)
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

relu_out = relu(conv_out)
sigmoid_out = sigmoid(conv_out)
print("\nReLU Output:\n", relu_out)
print("\nSigmoid Output:\n", np.round(sigmoid_out, 4))

# 5. Pooling Operations (2x2 window, stride=1 on 3x3 ReLU output -> 2x2)
max_pool = np.zeros((2, 2), dtype=np.float32)
min_pool = np.zeros((2, 2), dtype=np.float32)
for i in range(2):
    for j in range(2):
        patch = relu_out[i:i+2, j:j+2]
        max_pool[i, j] = np.max(patch)
        min_pool[i, j] = np.min(patch)

global_avg_pool = np.mean(relu_out)

print("\nMax Pooling (2x2):\n", max_pool)
print("\nMin Pooling (2x2):\n", min_pool)
print(f"\nGlobal Average Pooling (Scalar): {global_avg_pool:.4f}")

# 6. Flattening
flattened = max_pool.flatten()
print("\nFlattened Vector:", flattened)

# 7. Batch Normalization (1D)
eps = 1e-5
gamma, beta = 1.0, 0.0
mean = np.mean(flattened)
var = np.var(flattened)
batch_norm_out = gamma * ((flattened - mean) / np.sqrt(var + eps)) + beta
print("Batch Normalized Output:", np.round(batch_norm_out, 4))

# 8. Dropout (rate = 0.25)
dropout_rate = 0.25
keep_prob = 1.0 - dropout_rate
mask = (np.random.rand(*batch_norm_out.shape) < keep_prob).astype(np.float32)
dropout_out = (batch_norm_out * mask) / keep_prob
print("Dropout Output (rate=0.25):", np.round(dropout_out, 4))

# 9. Fully Connected (Dense) Layer (4 inputs -> 3 output classes)
weights = np.random.randn(flattened.shape[0], 3).astype(np.float32) * 0.5
bias = np.zeros(3, dtype=np.float32)
dense_logits = np.dot(dropout_out, weights) + bias
exp_scores = np.exp(dense_logits - np.max(dense_logits))
softmax_probs = exp_scores / np.sum(exp_scores)
print("Dense Layer Softmax Probabilities:", np.round(softmax_probs, 4))

# 10. Early Stopping Demonstration (monitoring simulated validation loss)
val_losses = [0.92, 0.78, 0.65, 0.61, 0.62, 0.63, 0.64]
patience = 2
best_loss = float("inf")
wait = 0

print("\nEarly Stopping Simulation (patience=2):")
for epoch, v_loss in enumerate(val_losses, start=1):
    if v_loss < best_loss:
        best_loss = v_loss
        wait = 0
        status = "Improved"
    else:
        wait += 1
        status = f"No improvement ({wait}/{patience})"
    print(f"Epoch {epoch}: val_loss = {v_loss:.2f} -> {status}")
    if wait >= patience:
        print(f"Early stopping triggered at epoch {epoch}! Best val_loss = {best_loss:.2f}")
        break

# Save and display visual summary of 3x3 operations
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
items = [
    (matrix, "Original 3x3"),
    (padded, "Padded 5x5 (P=1)"),
    (conv_out, "Conv Output 3x3"),
    (relu_out, "ReLU Output 3x3"),
    (max_pool, "Max Pool 2x2"),
]
for ax, (data, title) in zip(axes, items):
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(title, fontsize=10)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:.1f}", ha="center", va="center", color="white", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("cnn_basics_3x3_output.png", dpi=300)
plt.show()
