import numpy as np
from xray_utils import show_detailed_math

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():
    # 🥊 THE CONTRADICTION TEST
    # Input:  2, 1, 2, 1
    # Target: 0, 1, 1, 0
    X = np.array([[3], [1], [2], [1]])
    y_true = np.array([[0], [1], [1], [0]])

    # ARCHITECTURE: 1 -> 2 -> 2 -> 2 -> 1 (4 Layers!)
    W1 = np.random.randn(1, 2); b1 = np.zeros((1, 2))
    W2 = np.random.randn(2, 2); b2 = np.zeros((1, 2))
    W3 = np.random.randn(2, 2); b3 = np.zeros((1, 2))
    W4 = np.random.randn(2, 1); b4 = np.zeros((1, 1))
    
    lr = 0.1
    epochs = 100000

    print("--- 🧠 4-LAYER DEEP CONTRADICTION TEST ---")
    
    for epoch in range(epochs):
        # 1. Forward
        a0 = X
        z1 = np.dot(a0, W1) + b1; a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2; a2 = sigmoid(z2)
        z3 = np.dot(a2, W3) + b3; a3 = sigmoid(z3)
        z4 = np.dot(a3, W4) + b4; predictions = sigmoid(z4)
        
        # 2. Backprop
        gap = predictions - y_true
        
        # Layer 4
        d_z4 = 2 * gap * sigmoid_derivative(z4)
        d_W4 = np.dot(a3.T, d_z4); d_b4 = np.sum(d_z4, axis=0, keepdims=True)
        
        # Layer 3
        d_a3 = np.dot(d_z4, W4.T); d_z3 = d_a3 * sigmoid_derivative(z3)
        d_W3 = np.dot(a2.T, d_z3); d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        # Layer 2
        d_a2 = np.dot(d_z3, W3.T); d_z2 = d_a2 * sigmoid_derivative(z2)
        d_W2 = np.dot(a1.T, d_z2); d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        # Layer 1
        d_a1 = np.dot(d_z2, W2.T); d_z1 = d_a1 * sigmoid_derivative(z1)
        d_W1 = np.dot(a0.T, d_z1); d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # 3. Update
        W4 -= lr * d_W4; b4 -= lr * d_b4
        W3 -= lr * d_W3; b3 -= lr * d_b3
        W2 -= lr * d_W2; b2 -= lr * d_b2
        W1 -= lr * d_W1; b1 -= lr * d_b1

    print("\n" + "="*50)
    print("FINAL 4-LAYER PREDICTIONS")
    print("="*50)
    print(np.round(predictions, 2))

if __name__ == "__main__":
    main()
