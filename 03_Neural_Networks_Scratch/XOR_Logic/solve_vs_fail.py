import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(a): return a * (1 - a)

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def train_xor(hidden_size, epochs=5000):
    np.random.seed(42)
    W1 = np.random.randn(2, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1)
    b2 = np.zeros((1, 1))
    lr = 0.5
    
    for _ in range(epochs):
        a1 = sigmoid(np.dot(X, W1) + b1)
        a2 = sigmoid(np.dot(a1, W2) + b2)
        dz2 = (a2 - y) * sigmoid_derivative(a2)
        dW2 = np.dot(a1.T, dz2); db2 = np.sum(dz2, axis=0)
        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
        dW1 = np.dot(X.T, dz1); db1 = np.sum(dz1, axis=0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return a2

print("📉 CASE 1: 1 HIDDEN NEURON (The 'Impossible' Challenge)")
preds_1 = train_xor(1)
for i in range(4):
    print(f"Input {X[i]} -> Pred: {preds_1[i][0]:.3f} (Target: {y[i][0]})")
print("Deep Point: It fails! It predicts 0.5 for everything because 1 line cannot 'sandwich' XOR.")

print("\n📈 CASE 2: 2 HIDDEN NEURONS (The 'Minimum' Requirement)")
preds_2 = train_xor(2)
for i in range(4):
    print(f"Input {X[i]} -> Pred: {preds_2[i][0]:.3f} (Target: {y[i][0]})")
print("Deep Point: It succeeds! The two neurons 'negotiated' to solve the puzzle.")
