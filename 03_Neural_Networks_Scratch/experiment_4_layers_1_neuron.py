import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)

# Inputs
X = np.array([
    [11, 23],
    [11, 23],
    [-2500, -4300],
    [-2500, -4300]
])

# Targets
y_true = np.array([
    [0],
    [0],
    [1],
    [1]
])

# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

print("--- BUILDING DEEP NARROW NETWORK ---")
print("4 Hidden Layers, 1 Neuron Each (WITH BIAS)")

# --- 4 HIDDEN LAYERS, 1 NEURON EACH (IDENTICAL WEIGHTS) ---
# Layer 1: 2 inputs -> 1 neuron (Must be 2x1)
W1 = np.array([[0.5], [0.5]])
b1 = np.zeros((1, 1))

# Layer 2, 3, 4, 5: 1 input -> 1 neuron (Must be 1x1)
W2 = np.array([[0.5]])
b2 = np.zeros((1, 1))

W3 = np.array([[0.5]])
b3 = np.zeros((1, 1))

W4 = np.array([[0.5]])
b4 = np.zeros((1, 1))

W5 = np.array([[0.5]])
b5 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 50000

print("\nTraining...")

for epoch in range(epochs):
    # --- FORWARD PASS ---
    z1 = np.dot(X_normalized, W1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)
    
    z4 = np.dot(a3, W4) + b4
    a4 = sigmoid(z4)
    
    z5 = np.dot(a4, W5) + b5
    predictions = sigmoid(z5)
    
    # --- BACKPROPAGATION ---
    gap = predictions - y_true
    
    # Layer 5 (Output)
    d_z5 = 2 * gap * sigmoid_derivative(z5)
    d_W5 = np.dot(a4.T, d_z5)
    d_b5 = np.sum(d_z5, axis=0, keepdims=True)
    
    # Layer 4
    d_a4 = np.dot(d_z5, W5.T)
    d_z4 = d_a4 * sigmoid_derivative(z4)
    d_W4 = np.dot(a3.T, d_z4)
    d_b4 = np.sum(d_z4, axis=0, keepdims=True)
    
    # Layer 3
    d_a3 = np.dot(d_z4, W4.T)
    d_z3 = d_a3 * sigmoid_derivative(z3)
    d_W3 = np.dot(a2.T, d_z3)
    d_b3 = np.sum(d_z3, axis=0, keepdims=True)
    
    # Layer 2
    d_a2 = np.dot(d_z3, W3.T)
    d_z2 = d_a2 * sigmoid_derivative(z2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    
    # Layer 1
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)
    d_W1 = np.dot(X_normalized.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    # --- UPDATE WEIGHTS & BIASES ---
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
    W3 -= learning_rate * d_W3
    b3 -= learning_rate * d_b3
    
    W4 -= learning_rate * d_W4
    b4 -= learning_rate * d_b4
    
    W5 -= learning_rate * d_W5
    b5 -= learning_rate * d_b5

print("\n--- TRAINING COMPLETE ---")
print("Final Predictions:")
for i in range(len(X)):
    print(f"Input {X[i]} -> Prediction: {predictions[i][0]:.4f} (Target: {y_true[i][0]})")
