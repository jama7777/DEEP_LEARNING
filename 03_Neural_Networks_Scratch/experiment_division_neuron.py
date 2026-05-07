import numpy as np

def sigmoid(z):
    # Clip z to avoid overflow in exp if numbers get too crazy
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():
    print("--- 🧠 THE DIVISION NEURON EXPERIMENT ---")
    
    # Original Data
    X_raw = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_true = np.array([[0], [1], [1], [0]])

    # Engineered Features (x1, x2, x1*x2, bias)
    X = np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ])

    np.random.seed(42)
    # Start weights away from 0 to avoid immediate Division by Zero
    W1 = np.random.randn(4, 1) + 2.0 
    
    learning_rate = 0.05 # Smaller learning rate because Division gradients explode!
    epochs = 5000
    
    print("\nTraining Division Neuron silently...")
    for epoch in range(epochs):
        # --- 1. NEW FORWARD PROP: DIVISION! ---
        # Instead of np.dot(X, W1), we do: z = (x1/w1) + (x2/w2) + (x3/w3) + (x4/w4)
        
        # Add a tiny epsilon so Python doesn't crash if a weight hits exactly 0.0
        epsilon = 1e-8
        W_safe = W1.T + epsilon 
        
        # Divide each input by its weight, then sum them together
        z1 = np.sum(X / W_safe, axis=1, keepdims=True)
        predictions = sigmoid(z1)
        
        # --- 2. NEW BACKPROPAGATION: THE CALCULUS OF DIVISION! ---
        gap = predictions - y_true
        d_z1 = 2 * gap * sigmoid_derivative(z1)
        
        # The Derivative of (x * w) is just (x).
        # But the Derivative of (x / w) is (-x / w^2). 
        # We MUST completely rewrite the Backpropagation equation!
        
        derivative_of_division = -X / (W_safe ** 2)
        d_W1_matrix = d_z1 * derivative_of_division
        
        # Sum the gradients across all 4 samples to get the final Weight Updates
        d_W1 = np.sum(d_W1_matrix, axis=0, keepdims=True).T
        
        # Update weights
        W1 -= learning_rate * d_W1

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE ({epochs} epochs)")
    print("="*50)
    
    print("\n--- 🏁 FINAL PREDICTION ---")
    for i in range(len(X)):
        print(f"Input: {X_raw[i]} -> Prediction: {predictions[i][0]:.4f} (Target: {y_true[i][0]})")
        
    print("\n--- FINAL 4 WEIGHTS ---")
    print(f"w1: {W1[0][0]:.4f}")
    print(f"w2: {W1[1][0]:.4f}")
    print(f"w3: {W1[2][0]:.4f}")
    print(f"w4: {W1[3][0]:.4f}")

if __name__ == "__main__":
    main()
