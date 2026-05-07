import numpy as np
from xray_utils import show_detailed_math, visualize_backprop

# --- MATH HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def run_multi_neuron_network():
    # DATA: 1 sample, 2 inputs, 2 targets
    X = np.array([[0.5, 0.1]])
    y_true = np.array([[0.8, 0.2]])
    
    np.random.seed(42)
    
    # ARCHITECTURE: 2 Inputs -> 2 Hidden Neurons -> 2 Outputs
    
    # W1: (2 inputs -> 2 hidden)
    W1 = np.random.randn(2, 2); b1 = np.zeros((1, 2))
    
    # W2: (2 hidden -> 2 outputs)
    W2 = np.random.randn(2, 2); b2 = np.zeros((1, 2))
    
    lr = 0.5
    epochs = 2
    
    print("--- 🎡 THE MULTI-NEURON NETWORK (2-2-2) ---")
    print("Watch how the 2x2 Weight matrices handle multiple signals at once.")
    
    for epoch in range(epochs):
        print(f"\n{'#'*40}")
        print(f"   🚀 EPOCH {epoch + 1}")
        print(f"{'#'*40}")
        
        # 1. FORWARD PROPAGATION
        h_score = np.dot(X, W1) + b1
        h_act = sigmoid(h_score)
        show_detailed_math("1. HIDDEN CALC (X * W1)", [X, W1], h_score)
        show_detailed_math("2. HIDDEN ACT (Sigmoid)", [h_act], h_act, label="sigmoid")
        
        o_score = np.dot(h_act, W2) + b2
        prediction = sigmoid(o_score)
        show_detailed_math("3. OUTPUT CALC (H_Act * W2)", [h_act, W2], o_score)
        show_detailed_math("4. FINAL PREDICTION", [prediction], prediction, label="prediction")
    
        print(f"\n{'='*30}\n   BACKWARD PROPAGATION\n{'='*30}")
        
        # 2. BACKWARD PROPAGATION
        # Step 6: Output Signal (2 values)
        gap = prediction - y_true
        out_signal = 2 * gap * sigmoid_derivative(prediction)
        show_detailed_math("6. OUTPUT SIGNAL (2 errors)", [out_signal], out_signal)
        
        # Step 7: Update W2 (2x2 matrix)
        dW2 = np.dot(h_act.T, out_signal)
        show_detailed_math("7. dW2 (H_Act.T * Out_Signal)", [h_act.T, out_signal], dW2)
        
        # Step 8: THE MULTI-BLAME MERGE (Passing back through 2x2 W2)
        hidden_error = np.dot(out_signal, W2.T) 
        show_detailed_math("8a. HIDDEN ERROR (Out_Signal * W2.T)", [out_signal, W2.T], hidden_error)
        
        deriv_h = sigmoid_derivative(h_act)
        hidden_signal = hidden_error * deriv_h
        
        print("\n--- 8b. HIDDEN SIGNAL (Error * Deriv) ---")
        print(f"Error {hidden_error[0]} * Deriv {deriv_h[0]} = {hidden_signal[0]}")
        
        # Step 9: Update W1 (2x2 matrix)
        dW1 = np.dot(X.T, hidden_signal)
        show_detailed_math("9. dW1 (X.T * Hidden_Signal)", [X.T, hidden_signal], dW1)
    
        # APPLY UPDATES
        W1 -= lr * dW1
        W2 -= lr * dW2
        b1 -= lr * np.sum(hidden_signal, axis=0, keepdims=True)
        b2 -= lr * np.sum(out_signal, axis=0, keepdims=True)

        # GENERATE VISUALIZATION for the first epoch
        if epoch == 0:
            print("\n[VISUAL] Generating Backprop Diagram...")
            # Using the correct signature: (X, W1, W2, d_hidden, d_out, dW1, dW2)
            visualize_backprop(X, W1, W2, hidden_signal, out_signal, dW1, dW2)
        
        print(f"\n[SYSTEM] Epoch {epoch+1} complete. 2x2 weights adjusted.")

if __name__ == "__main__":
    run_multi_neuron_network()
