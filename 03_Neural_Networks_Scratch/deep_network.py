import numpy as np
from xray_utils import show_detailed_math, show_activation_logic

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    X = np.array([[0.9, 0.8, 0.2], [0.7, 0.7, 0.3], [0.3, 0.3, 0.9]])
    y_true = np.array([[1.0], [1.0], [0.0]])

    np.random.seed(42)
    W1 = np.random.randn(3, 5); b1 = np.zeros((1, 5))
    W2 = np.random.randn(5, 4); b2 = np.zeros((1, 4))
    W3 = np.random.randn(4, 1); b3 = np.zeros((1, 1))
    
    print("--- 🧠 DEEP NETWORK: X-RAY EDITION ---")
    
    for epoch in range(1):
        # Forward Propagation
        z1 = np.dot(X, W1) + b1
        h1_out = sigmoid(z1)
        show_detailed_math("1. LAYER 1 (3->5)", [X, W1, b1], z1)
        
        z2 = np.dot(h1_out, W2) + b2
        h2_out = sigmoid(z2)
        show_detailed_math("2. LAYER 2 (5->4)", [h1_out, W2, b2], z2)
        
        z3 = np.dot(h2_out, W3) + b3
        predictions = sigmoid(z3)
        show_detailed_math("3. OUTPUT LAYER (4->1)", [h2_out, W3, b3], z3)
        show_activation_logic("FINAL PREDICTIONS", z3, predictions, "sigmoid")

    print("\nDeep architecture visualized! You can see how the information is compressed as it flows to the output.")

if __name__ == "__main__":
    main()
