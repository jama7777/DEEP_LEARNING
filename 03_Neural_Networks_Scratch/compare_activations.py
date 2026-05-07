import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): 
    s = sigmoid(z)
    return s * (1 - s)

def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

# DATA
X = np.array([[0.9, 0.8, 0.2], [0.7, 0.7, 0.3], [0.3, 0.3, 0.9]])
y_true = np.array([[1.0], [1.0], [0.0]])

def train_network(activation_type='sigmoid', epochs=1000, lr=0.1):
    np.random.seed(42)
    # 3 -> 5 -> 1
    W1 = np.random.randn(3, 5) * 0.5
    b1 = np.zeros((1, 5))
    W2 = np.random.randn(5, 1) * 0.5
    b2 = np.zeros((1, 1))
    
    losses = []
    
    for epoch in range(epochs):
        # Forward
        z1 = np.dot(X, W1) + b1
        h1 = relu(z1) if activation_type == 'relu' else sigmoid(z1)
        
        z2 = np.dot(h1, W2) + b2
        pred = sigmoid(z2) # Output always sigmoid for 0-1 range
        
        # Loss
        loss = np.mean((pred - y_true)**2)
        losses.append(loss)
        
        # Backprop
        d_out = (pred - y_true) * (pred * (1 - pred))
        
        if activation_type == 'relu':
            d_h1 = np.dot(d_out, W2.T) * relu_deriv(z1)
        else:
            d_h1 = np.dot(d_out, W2.T) * sigmoid_deriv(z1)
            
        # Update
        W2 -= lr * np.dot(h1.T, d_out)
        b2 -= lr * np.sum(d_out, axis=0, keepdims=True)
        W1 -= lr * np.dot(X.T, d_h1)
        b1 -= lr * np.sum(d_h1, axis=0, keepdims=True)
        
    return losses

# Run training
print("Training Sigmoid...")
sigmoid_losses = train_network('sigmoid', epochs=1500000, lr=0.2)
print("Training ReLU...")
relu_losses = train_network('relu', epochs=1500000, lr=0.2)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sigmoid_losses, label='Sigmoid (The Classic)', color='blue', linewidth=2)
plt.plot(relu_losses, label='ReLU (The Modern Standard)', color='red', linewidth=2, linestyle='--')
plt.title('Battle of Activations: Sigmoid vs ReLU', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (Loss)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_comparison.png')
print("\nGraph saved as 'loss_comparison.png'")

# Final Loss Comparison
print(f"Final Sigmoid Loss: {sigmoid_losses[-1]:.6f}")
print(f"Final ReLU Loss: {relu_losses[-1]:.6f}")
