import numpy as np
from xray_utils import (
    show_detailed_math, show_dot_logic, show_activation_logic
)

def main():
    # --- PROPER 2-INPUT XOR DATA ---
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_true = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    # Exactly 2 weights for x1 and x2
    W1 = np.random.randn(2, 1) * 3.0
    # A dedicated bias variable
    b1 = np.zeros((1, 1))
    
    learning_rate = 0.5
    epochs = 10000

    print("--- 🌊 THE SINE MIRACLE: Proper 2-Input Neuron ---")
    
    for epoch in range(epochs):
        # --- FORWARD PASS ---
        # z = x1*w1 + x2*w2 + b
        z = np.dot(X, W1) + b1
        predictions = np.sin(z)
        
        # --- BACKWARD PASS ---
        gap = predictions - y_true
        d_z = 2 * gap * np.cos(z)
        
        d_W1 = np.dot(X.T, d_z)
        d_b1 = np.sum(d_z, axis=0, keepdims=True)

        # --- MATH WALKTHROUGH (EPOCH 1 ONLY) ---
        if epoch == 0:
            print("\n" + "="*50)
            print("🚀 EPOCH 1: THE 2-INPUT CALCULUS")
            print("="*50)
            show_detailed_math("Step 1: Z = X @ W + b", [X, W1, b1], z, operation="+")
            show_detailed_math("Step 2: Predictions = sin(Z)", [z], predictions, label="sin(z)")
            
            print("\n--- 2. THE BACKWARD PASS ---")
            deriv = np.cos(z)
            show_detailed_math("Step 3: d_z = 2 * gap * cos(Z)", [2 * gap, deriv], d_z, operation=".*")
            
            print("\n--- TRAINING SILENTLY... ---")

        # Updates
        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE ({epochs} epochs)")
    print("="*50)
    
    # FINAL PREDICTION
    print("\n--- 🏁 FINAL PREDICTION (Proper 2-Input Neuron) ---")
    for i in range(len(X)):
        raw_z = (np.dot(X[i], W1) + b1)[0][0]
        print(f"Input: {X[i]} -> Raw Sum(z): {raw_z:.4f} -> Sin(z): {predictions[i][0]:.4f} (Target: {y_true[i][0]})")
        
    print("\n--- FINAL WEIGHTS & BIAS ---")
    print(f"w1: {W1[0][0]:.4f}")
    print(f"w2: {W1[1][0]:.4f}")
    print(f"Bias: {b1[0][0]:.4f}")

if __name__ == "__main__":
    main()
