import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(a): return a * (1 - a)

# XOR Data
X_raw = np.array([[1,2], [2,3], [4,5], [6,7]])
y = np.array([[0], [1], [1], [0]])

# Normalization
X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

# Initialize
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.5
W2 = np.random.randn(2, 1) * 0.5
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 5001

for epoch in range(epochs):
    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Backward
    dz2 = (a2 - y) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print("🕵️ THE DEEP CANCELLATION CHECK")
print("=" * 50)
for i in range(len(X)):
    h1 = a1[i, 0] * W2[0, 0] # Contribution of Neuron 1
    h2 = a1[i, 1] * W2[1, 0] # Contribution of Neuron 2
    total = h1 + h2 + b2[0,0]
    pred = sigmoid(total)
    
    print(f"Input: {X_raw[i]}")
    print(f"  Neuron 1 Push: {h1:+.3f}")
    print(f"  Neuron 2 Push: {h2:+.3f}")
    print(f"  Final Sum (before sigmoid): {total:+.3f} -> Prediction: {pred:.3f}")
    print("-" * 30)

print("\n[DEEP POINT]: Notice how for [6,7], one neuron pushes POSITIVE and the other pushes NEGATIVE.")
print("They 'Negotiated' to cancel each other out to reach 0!")
