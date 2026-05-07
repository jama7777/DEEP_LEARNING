import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return (x > 0).astype(float)

# Create data
x = np.linspace(-5, 5, 400)

plt.figure(figsize=(12, 8))

# --- Plot 1: The Functions ---
plt.subplot(2, 1, 1)
plt.plot(x, x, label="Linear (f(x)=x)", color='gray', linestyle='--')
plt.plot(x, sigmoid(x), label="Sigmoid (Smooth Curve)", color='blue', linewidth=2)
plt.plot(x, relu(x), label="ReLU (Bent Line)", color='red', linewidth=2)
plt.title("Activation Functions: How they bend space")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-2, 5)

# --- Plot 2: The Derivatives (Learning Signal) ---
plt.subplot(2, 1, 2)
plt.plot(x, np.ones_like(x), label="Linear Derivative (Always 1)", color='gray', linestyle='--')
plt.plot(x, sigmoid_derivative(x), label="Sigmoid Derivative (Vanishing!)", color='blue', linewidth=2)
plt.plot(x, relu_derivative(x), label="ReLU Derivative (Binary 0 or 1)", color='red', linewidth=2)
plt.title("Derivatives: Why ReLU stays 'Alive' while Sigmoid 'Vanishes'")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-0.1, 1.2)

plt.tight_layout()
plt.savefig("/Users/indra/Desktop/DEEP_LEARNING/activation_comparison.png")
print("Graph saved to: /Users/indra/Desktop/DEEP_LEARNING/activation_comparison.png")
