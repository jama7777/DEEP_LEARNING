import numpy as np
from xray_utils import show_detailed_math, visualize_backprop

# --- MATH HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def run_double_network():
    # DATA: 2 Inputs, 2 Targets
    X = np.array([[0.5, 0.1]])
    y_true = np.array([[0.8, 0.2]])
    
    # WEIGHTS: 2nd layer connects 2 hidden to 2 output (2x2)
    np.random.seed(42)
    W1 = np.random.randn(2, 2)
    b1 = np.zeros((1, 2))
    W2 = np.random.randn(2, 2)
    b2 = np.zeros((1, 2))
    
    lr = 1.0 
    
    print("--- 🔬 THE MULTI-TRACK BRAIN (2-2-2) ---")
    print(f"Goal: Input {X[0]} must result in Target {y_true[0]}")
    
    for epoch in range(1):
        print(f"\n{'='*30}")
        print(f"   FORWARD PROPAGATION")
        print(f"{'='*30}")
        
        # 1. Input -> Hidden Score
        hidden_score = np.dot(X, W1) + b1
        show_detailed_math("1. HIDDEN SCORE (X * W1 + b1)", [X, W1, b1], hidden_score)
        
        # 2. Hidden Activation
        hidden_activation = sigmoid(hidden_score)
        show_detailed_math("2. HIDDEN ACTIVATION", [hidden_activation], hidden_activation, label="sigmoid")
        
        # 3. Hidden -> Output Score
        output_score = np.dot(hidden_activation, W2) + b2
        show_detailed_math("3. OUTPUT SCORE", [hidden_activation, W2, b2], output_score)
        
        # 4. Final Prediction (Two outputs!)
        output_prediction = sigmoid(output_score)
        show_detailed_math("4. FINAL PREDICTION", [output_prediction], output_prediction, label="prediction")

        print(f"\n{'='*30}")
        print(f"   BACKWARD PROPAGATION")
        print(f"{'='*30}")
        
        # 5. The Gap (Two Mistakes at once)
        gap = output_prediction - y_true
        show_detailed_math("5. GAP (Pred - Target)", [output_prediction, y_true], gap, operation="-")
        
        # 6. Output Signal
        deriv_output = sigmoid_derivative(output_prediction)
        output_signal = 2 * gap * deriv_output
        show_detailed_math("6. OUTPUT SIGNAL", [2 * gap, deriv_output], output_signal, operation=".*")
        
        # 7. Update for W2 (2x2 matrix)
        dW2 = np.dot(hidden_activation.T, output_signal)
        show_detailed_math("7. dW2 (Update Matrix)", [hidden_activation.T, output_signal], dW2)
        
        # 8. Passing blame back to Hidden Layer
        # Each hidden neuron gets blame from BOTH outputs
        deriv_hidden = sigmoid_derivative(hidden_activation)
        hidden_error = np.dot(output_signal, W2.T) 
        hidden_signal = hidden_error * deriv_hidden
        show_detailed_math("8. HIDDEN SIGNAL", [hidden_error, deriv_hidden], hidden_signal, operation=".*")
        
        # 9. Update for W1 (2x2 matrix)
        dW1 = np.dot(X.T, hidden_signal)
        show_detailed_math("9. dW1 (Update Matrix)", [X.T, hidden_signal], dW1)
        
        # VISUALIZE BACKPROP
        visualize_backprop(X, W1, W2, hidden_signal, output_signal, dW1, dW2)
        
        # APPLY CHANGES
        W1 -= lr * dW1
        b1 -= lr * hidden_signal
        W2 -= lr * dW2
        b2 -= lr * output_signal
        
        print(f"\n[SYSTEM] 2x2 Matrices updated. Multi-output step complete.")

if __name__ == "__main__":
    run_double_network()
