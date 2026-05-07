import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():
    # 1. THE XOR DATASET
    # Non-linearly separable!
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_true = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    
    # 2. ARCHITECTURE: 2 -> 4 -> 4 -> 1
    # We need enough "brain power" to fold the logic
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    
    W2 = np.random.randn(4, 4)
    b2 = np.zeros((1, 4))
    
    W3 = np.random.randn(4, 1)
    b3 = np.zeros((1, 1))
    
    learning_rate = 0.5
    
    print("--- 🧩 SOLVING THE 'IMPOSSIBLE' XOR PROBLEM ---")
    
    for epoch in range(10001):
        # Forward Prop
        h1 = sigmoid(np.dot(X, W1) + b1)
        h2 = sigmoid(np.dot(h1, W2) + b2)
        predictions = sigmoid(np.dot(h2, W3) + b3)
        
        # Loss
        loss = np.mean((predictions - y_true)**2)
        
        # Backprop (The "Chain of Blame" you mastered)
        d_out = (predictions - y_true) * sigmoid_derivative(np.dot(h2, W3) + b3)
        d_h2 = np.dot(d_out, W3.T) * sigmoid_derivative(np.dot(h1, W2) + b2)
        d_h1 = np.dot(d_h2, W2.T) * sigmoid_derivative(np.dot(X, W1) + b1)
        
        # Update
        W3 -= learning_rate * np.dot(h2.T, d_out)
        b3 -= learning_rate * np.sum(d_out, axis=0, keepdims=True)
        
        W2 -= learning_rate * np.dot(h1.T, d_h2)
        b2 -= learning_rate * np.sum(d_h2, axis=0, keepdims=True)
        
        W1 -= learning_rate * np.dot(X.T, d_h1)
        b1 -= learning_rate * np.sum(d_h1, axis=0, keepdims=True)
        
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5} | Loss: {loss:.8f}")

    print("-" * 50)
    print("✅ XOR LOGIC LEARNED")
    print("\nFinal XOR Truth Table Check:")
    for i in range(len(X)):
        pred_binary = 1 if predictions[i] > 0.5 else 0
        print(f"Input: {X[i]} | Target: {y_true[i][0]} | Predicted: {predictions[i][0]:.4f} ({pred_binary})")

if __name__ == "__main__":
    main()
