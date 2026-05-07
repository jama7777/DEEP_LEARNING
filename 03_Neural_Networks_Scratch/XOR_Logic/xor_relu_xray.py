import numpy as np
from xray_utils import show_detailed_math

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def main():
    # --- ENGINEERED XOR DATA ---
    X = np.array([
        [0, 0, 0, 1],  # Target 0
        [0, 1, 0, 1],  # Target 1
        [1, 0, 0, 1],  # Target 1
        [1, 1, 1, 1]   # Target 0
    ])
    y_true = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(4, 1) * 0.5
    
    learning_rate = 0.1
    epochs = 500

    print("--- 🧠 THE ReLU NEURON: DEEP MATH X-RAY ---")
    
    for epoch in range(epochs):
        # --- FORWARD PASS ---
        z = np.dot(X, W1)
        predictions = relu(z)
        
        # --- BACKWARD PASS ---
        gap = predictions - y_true
        # The ReLU Derivative Switch: 0 or 1
        d_z = 2 * gap * relu_derivative(z)
        d_W1 = np.dot(X.T, d_z)

        # --- MATH WALKTHROUGH (EPOCH 1 ONLY) ---
        if epoch < 5:  # Showing first 5 epochs for deep clarity
            print("\n" + "="*50)
            print(f"🚀 EPOCH {epoch+1}: THE RELU CALCULUS")
            print("="*50)
            
            # Forward Pass Math
            show_detailed_math("Step 1: Z = X @ W1", [X, W1], z, operation="*")
            show_detailed_math("Step 2: Predictions = relu(Z)", [z], predictions, label="relu(z)")
            
            print("\n--- 2. THE BACKWARD PASS (The Chain Rule) ---")
            show_detailed_math("Step 3: The Gap = Preds - y_true", [predictions, y_true], gap, operation="-")
            
            # Show the ReLU Switch (The Derivative)
            deriv = relu_derivative(z)
            show_detailed_math("Step 4: Error Signal d_z = 2*gap .* relu'(z)", [2 * gap, deriv], d_z, operation=".*")
            
            # Show the final weight gradient
            show_detailed_math("Step 5: Gradient d_W1 = X.T @ d_z", [X.T, d_z], d_W1, operation="*")
            
            print("\n--- TRAINING SILENTLY... ---")

        # Updates
        W1 -= learning_rate * d_W1

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE ({epochs} epochs)")
    print("="*50)
    
    # FINAL PREDICTION
    print("\n--- 🏁 FINAL PREDICTION (ReLU Neuron) ---")
    for i in range(len(X)):
        print(f"Input: {X[i][:2]} -> Prediction: {predictions[i][0]:.4f} (Target: {y_true[i][0]})")
        
    print("\n--- FINAL 4 WEIGHTS ---")
    for i, w in enumerate(W1):
        print(f"w{i+1}: {w[0]:.4f}")

if __name__ == "__main__":
    main()
