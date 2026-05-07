import numpy as np
from xray_utils import show_detailed_math, visualize_backprop

# --- MATH HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is the activated value (sigmoid(z))
    return x * (1 - x)

def run_simplest_network():
    # DATA: One input (0.5), One target (0.8)
    X = np.array([[0.5]])
    y_true = np.array([[0.8]])
    
    # WEIGHTS: 1 Input -> 2 Hidden Neurons -> 1 Output (1-2-1)
    np.random.seed(42)
    W1 = np.random.randn(1, 2) # Now 2 neurons in hidden layer
    b1 = np.zeros((1, 2))
    W2 = np.random.randn(2, 1) # Must accept 2 inputs from hidden
    b2 = np.zeros((1, 1))
    
    lr = 1.0 
    
    print("--- 🔬 THE DUAL NEURON (1-2-1) ---")
    print(f"Goal: Input {X[0,0]} must result in Target {y_true[0,0]}")
    
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
        
        # 4. Final Prediction
        output_prediction = sigmoid(output_score)
        show_detailed_math("4. FINAL PREDICTION", [output_prediction], output_prediction, label="prediction")

        print(f"\n{'='*30}")
        print(f"   BACKWARD PROPAGATION")
        print(f"{'='*30}")
        
        # 5. The Gap (Mistake)
        gap = output_prediction - y_true
        show_detailed_math("5. GAP (Pred - Target)", [output_prediction, y_true], gap, operation="-")
        
        # 6. Output Signal (The Blame on the Output Neuron)
        deriv_output = sigmoid_derivative(output_prediction)
        output_signal = 2 * gap * deriv_output
        show_detailed_math("6. OUTPUT SIGNAL (2 * Gap * Deriv)", [2 * gap, deriv_output], output_signal, operation=".*")
        
        # 7. Update for W2 (Output Weights)
        dW2 = np.dot(hidden_activation.T, output_signal)
        show_detailed_math("7. dW2 (Hidden_Act.T * Output_Signal)", [hidden_activation.T, output_signal], dW2)
        
        # 8. Passing blame back to Hidden Layer
        deriv_hidden = sigmoid_derivative(hidden_activation)
        hidden_error = np.dot(output_signal, W2.T) # Signal traveling back through the W2 bridge
        hidden_signal = hidden_error * deriv_hidden
        show_detailed_math("8. HIDDEN SIGNAL (Output_Sig * W2.T * Deriv)", [hidden_error, deriv_hidden], hidden_signal, operation=".*")
        
        # 9. Update for W1 (Hidden Weights)
        dW1 = np.dot(X.T, hidden_signal)
        show_detailed_math("9. dW1 (X.T * Hidden_Signal)", [X.T, hidden_signal], dW1)
        
        # VISUALIZE BACKPROP
        # Passing renamed variables to the visualizer
        visualize_backprop(X, W1, W2, hidden_signal, output_signal, dW1, dW2)
        
        # APPLY CHANGES
        W1 -= lr * dW1
        b1 -= lr * hidden_signal
        W2 -= lr * dW2
        b2 -= lr * output_signal
        
        print(f"\n[SYSTEM] Weights updated. Calculation complete.")

if __name__ == "__main__":
    run_simplest_network()
