import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(a): return a * (1 - a)

def train_network(initial_W1, label):
    # XOR Data
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 2-2-1 Architecture
    W1 = initial_W1.copy()
    W2 = np.random.randn(2, 1)
    lr = 0.5
    losses = []

    for epoch in range(2000):
        # Forward
        z1 = np.dot(X, W1)
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2)
        a2 = sigmoid(z2)
        
        # Loss
        loss = np.mean((a2 - y)**2)
        losses.append(loss)
        
        # Backprop
        dz2 = (a2 - y) * sigmoid_derivative(a2)
        dW2 = np.dot(a1.T, dz2)
        
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * sigmoid_derivative(a1)
        dW1 = np.dot(X.T, dz1)
        
        # Update
        W2 -= lr * dW2
        W1 -= lr * dW1
        
    return losses, W1

# --- THE EXPERIMENT ---
np.random.seed(42)

# 1. The Clone Setup: All hidden weights are EXACTLY the same
clone_W1 = np.ones((2, 2)) * 0.5

# 2. The Random Setup: Tiny differences to "break symmetry"
random_W1 = np.random.randn(2, 2) * 0.1

print("🚀 Running Experiment: Clones vs. Specialists...")
losses_clone, final_W1_clone = train_network(clone_W1, "Clone")
losses_random, final_W1_random = train_network(random_W1, "Random")

# --- RESULTS ---
print("\n--- CLONE NETWORK RESULTS ---")
print("Final Weights (W1):\n", final_W1_clone)
print("Deep Point: Notice how the weights for Neuron 1 and Neuron 2 are still IDENTICAL!")

print("\n--- RANDOM NETWORK RESULTS ---")
print("Final Weights (W1):\n", final_W1_random)
print("Deep Point: The weights have 'diverged' into different values to solve different parts of XOR.")

plt.figure(figsize=(10, 5))
plt.plot(losses_clone, label="Identical Initialization (Symmetry Trap)")
plt.plot(losses_random, label="Random Initialization (Symmetry Broken)")
plt.title("The 'Push': Why Randomness is Required")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("symmetry_loss_comparison.png")
print("\n[SYSTEM] Comparison plot saved as 'symmetry_loss_comparison.png'")
