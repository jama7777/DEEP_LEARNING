import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(a): return a * (1 - a)

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize
np.random.seed(42)
W1 = np.random.randn(2, 2) * 1.5
W2 = np.random.randn(2, 1) * 1.5
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 2001

# --- TRAINING LOOP ---
for epoch in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    dz2 = (a2 - y) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

# --- VISUALIZATION ---
def plot_boundary(ax, w, b, title):
    # Solve w1*x1 + w2*x2 + b = 0  => x2 = (-w1*x1 - b) / w2
    x_vals = np.linspace(-0.5, 1.5, 10)
    if abs(w[1]) > 0.001:
        y_vals = (-w[0] * x_vals - b) / w[1]
        ax.plot(x_vals, y_vals, '--', label="Decision Boundary")
    
    # Fill the 'Active' side
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = sigmoid(np.dot(grid, w.reshape(2,1)) + b).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, cmap='RdBu')
    
    # Plot XOR points
    for i in range(4):
        ax.scatter(X[i,0], X[i,1], c='red' if y[i,0]==0 else 'green', edgecolors='k', s=100)
    ax.set_title(title)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_boundary(axes[0], W1[:, 0], b1[0, 0], "Neuron 1: The 'OR' Specialist")
plot_boundary(axes[1], W1[:, 1], b1[0, 1], "Neuron 2: The 'AND' Specialist")

# Final Output Visualization
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
grid = np.c_[xx.ravel(), yy.ravel()]
h_act = sigmoid(np.dot(grid, W1) + b1)
out_act = sigmoid(np.dot(h_act, W2) + b2).reshape(xx.shape)
axes[2].contourf(xx, yy, out_act, alpha=0.3, cmap='RdBu')
for i in range(4):
    axes[2].scatter(X[i,0], X[i,1], c='red' if y[i,0]==0 else 'green', edgecolors='k', s=100)
axes[2].set_title("Combined Result: XOR Solved!")

plt.tight_layout()
plt.savefig("negotiation_visualized.png")
print("\n[SYSTEM] Visualization saved as 'negotiation_visualized.png'")
