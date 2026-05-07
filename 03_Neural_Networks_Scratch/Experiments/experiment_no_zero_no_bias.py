import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)

# Inputs: Like a truth table, but using 1 and 2 instead of 0 and 1 (No Zeros!)
X = np.array([
    [11, 23],
    [11, 23],
    [-2500, -4300],
    [-2500, -4300]
])

# Target: Like an AND gate
y_true = np.array([
    [0],
    [0],
    [1],
    [1]
])

# --- DATA NORMALIZATION ---
# We use Z-score Standardization: (X - Mean) / Standard Deviation
# This shrinks the massive numbers down to a small range (usually between -3 and +3)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

print("--- NORMALIZED INPUTS ---")
print(X_normalized)
print("-------------------------\n")

# --- SYMMETRIC WEIGHTS (Cloned) & NO BIAS ---
# Layer 1: 2 inputs -> 4 hidden neurons
# We create one set of weights and duplicate it perfectly 4 times
n1 = np.array([[0.5], [0.5]]) 
W1 = np.hstack([n1, n1+0.5, n1, n1])      

# Layer 2: 4 hidden -> 1 output
# We create one output weight and duplicate it perfectly 4 times
n2 = np.array([[0.07]])
W2 = np.vstack([n2, n2+0.5, n2, n2])      

learning_rate = 0.1
epochs = 50000

print("--- INITIAL SYMMETRIC WEIGHTS (NO BIAS) ---")
print("W1:\n", W1)
print("W2:\n", W2)
print("\nTraining...")

for epoch in range(epochs):
    # 1. Forward Pass (No Biases!)
    z1 = np.dot(X_normalized, W1)
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, W2)
    predictions = sigmoid(z2)
    
    # 2. Backpropagation
    gap = predictions - y_true
    
    # --- THE LOUDSPEAKER EFFECT ---
    # We turn the loudspeaker back to 1 to show normal learning
    loudspeaker = np.array([
        [1], 
        [1], 
        [1], 
        [1]  
    ])
    gap = gap * loudspeaker
    
    d_z2 = 2 * gap * sigmoid_derivative(z2)
    d_W2 = np.dot(a1.T, d_z2)
    
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)
    d_W1 = np.dot(X_normalized.T, d_z1)
    
    # 3. Update Weights (No biases to update!)
    W1 -= learning_rate * d_W1
    W2 -= learning_rate * d_W2

print("\n--- TRAINING COMPLETE ---")
print("Final W1:\n", W1)
print("Final W2:\n", W2)
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"Input {X[i]} -> Prediction: {predictions[i][0]:.4f} (Target: {y_true[i][0]})")

print("\nNotice how W1's columns are still identical, and W2's rows are identical!")
print("They learned (the weights changed), but they learned as identical clones.")
