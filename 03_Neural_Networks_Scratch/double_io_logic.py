import numpy as np
from xray_utils import show_detailed_math

# --- MATH HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def run_double_io_one_hidden():
    # DATA: 4 samples, 2 inputs each (Full XOR set)
    X = np.array([[0, 1], 
                  [1, 0],
                  [0, 0],
                  [1, 1]])
    
    # TARGETS: 4 samples, 1 output each
    y_true = np.array([[1], 
                       [1],
                       [0],
                       [0]]) 
    
    np.random.seed(42)
    
    # --- ARCHITECTURE: 2 Inputs -> 1 Hidden Neuron -> 2 Outputs ---
    
    # W1: 2 inputs -> 1 hidden neuron
    W1 = np.random.randn(2, 1); b1 = np.zeros((1, 1))
    
    # W2: 1 hidden neuron -> 1 output neuron
    W2 = np.random.randn(1, 1); b2 = np.zeros((1, 1))
    
    lr = 0.0001
    epochs = 100000 # Reduced to 100k for faster testing
    
    print("--- ⚖️ THE SINGLE NEURON BALANCER (2-1-2) ---")
    print(f"Training on {len(X)} samples over {epochs} epochs.")
    
    for epoch in range(epochs):
        # Only show math for the first epoch and the last epoch
        show_math = (epoch == 0 or epoch == epochs - 1)
        
        if show_math:
            print(f"\n{'#'*40}")
            print(f"   🚀 EPOCH {epoch + 1}")
            print(f"{'#'*40}")
        
        # 1. FORWARD PROPAGATION
        # Hidden Layer
        h_score = np.dot(X, W1) + b1
        h_act = sigmoid(h_score)
        if show_math: show_detailed_math("1. HIDDEN CALC (X * W1)", [X, W1], h_score)
        
        # Output Layer
        o_score = np.dot(h_act, W2) + b2
        prediction = sigmoid(o_score)
        if show_math:
            show_detailed_math("3. OUTPUT CALC (H_Act * W2)", [h_act, W2], o_score)
            show_detailed_math("4. PREDICTION (Sigmoid)", [prediction], prediction, label="prediction")
    
        # 2. BACKWARD PROPAGATION
        gap = prediction - y_true
        if show_math:
            print(f"\n{'='*30}\n   BACKWARD PROPAGATION\n{'='*30}")
            show_detailed_math("5. GAP (Pred - Target)", [prediction, y_true], gap, operation="-")
        
        out_signal = 2 * gap * sigmoid_derivative(prediction)
        
        # Step 7: Update W2
        dW2 = np.dot(h_act.T, out_signal)
        if show_math: show_detailed_math("7. dW2 (H_Act.T * Out_Signal)", [h_act.T, out_signal], dW2)
        
        # Step 8: THE MERGED BLAME
        hidden_error = np.dot(out_signal, W2.T) 
        deriv_h = sigmoid_derivative(h_act)
        hidden_signal = hidden_error * deriv_h
        
        if show_math:
            show_detailed_math("8a. HIDDEN ERROR (Out_Signal * W2.T)", [out_signal, W2.T], hidden_error)
            print("\n--- 8b. HIDDEN SIGNAL (Error * Deriv) ---")
            print(f"First Sample Signal: {hidden_signal[0]}")
        
        # Step 9: Update W1
        dW1 = np.dot(X.T, hidden_signal)
        if show_math: show_detailed_math("9. dW1 (X.T * Hidden_Signal)", [X.T, hidden_signal], dW1)
    
        # APPLY UPDATES
        W1 -= lr * dW1
        W2 -= lr * dW2
        b1 -= lr * np.sum(hidden_signal, axis=0, keepdims=True)
        b2 -= lr * np.sum(out_signal, axis=0, keepdims=True)
        
        print(f"\n[SYSTEM] Epoch {epoch+1} complete.")

if __name__ == "__main__":
    run_double_io_one_hidden()
