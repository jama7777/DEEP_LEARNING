import numpy as np
from xray_utils import show_detailed_math, show_dot_logic, show_activation_logic, show_elementwise_logic, visualize_network

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():
    # XOR DATA
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, 4); b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1); b2 = np.zeros((1, 1))
    
    learning_rate = 0.1
    epochs = 500 # Back to the deep-dive single epoch

    print("--- 🧠 THE TOTAL X-RAY: Deep Dive Reverted ---")
    
    for epoch in range(epochs):
        # 1. Forward Prop
        hidden_scores = np.dot(X, W1) + b1
        show_detailed_math("1. HIDDEN SCORES (Z1)", [X, W1, b1], hidden_scores)
        show_dot_logic("Z1", X, W1, row=1, col=2)
        
        hidden_activations = sigmoid(hidden_scores)
        show_detailed_math("2. HIDDEN ACTIVATIONS (A1)", [hidden_activations], hidden_activations, label="sigmoid(Z1)")
        show_activation_logic("A1", hidden_scores, hidden_activations, "sigmoid")
        
        final_scores = np.dot(hidden_activations, W2) + b2
        show_detailed_math("3. FINAL SCORES (Z2)", [hidden_activations, W2, b2], final_scores)
        show_dot_logic("Z2", hidden_activations, W2, row=1, col=0)
        
        predictions = sigmoid(final_scores)
        show_detailed_math("4. PREDICTIONS (A2)", [predictions], predictions, label="sigmoid(Z2)")
        show_activation_logic("A2", final_scores, predictions, "sigmoid")
        
        # --- NEW: VISUALIZE THE NETWORK STATE ---
        # Let's see the diagram for the very first sample [0,0]
        visualize_network(X, W1, W2, hidden_activations, predictions, sample_idx=0)
        
        # --- BACKPROP ---
        print("\n" + "!"*20 + " BACKPROPAGATION START " + "!"*20)
        
        gap = predictions - y_true
        show_detailed_math("5. OUTPUT GAP (Pred - Target)", [predictions, y_true], gap, operation="-")
        show_elementwise_logic("Gap", predictions, y_true, gap, "-", row=1, col=0)
        
        sensitivity_out = sigmoid_derivative(final_scores)
        show_detailed_math("6. OUTPUT SENSITIVITY (Sig_Deriv)", [sensitivity_out], sensitivity_out, label="deriv(sigmoid(Z2))")
        show_activation_logic("Sens_Out", final_scores, sensitivity_out, "sigmoid_derivative")
        
        d_out = 2 * gap * sensitivity_out
        show_detailed_math("7. d_out (2 * Gap * Sensitivity)", [d_out], d_out, label="2 * Gap * Sensitivity")
        show_elementwise_logic("d_out", (2 * gap), sensitivity_out, d_out, "*", row=1, col=0)
        
        d_W2 = np.dot(hidden_activations.T, d_out)
        show_detailed_math("8. GRAD_W2 (A1.T * d_out)", [hidden_activations.T, d_out], d_W2)
        show_dot_logic("Grad_W2", hidden_activations.T, d_out, row=0, col=0)
        
        error_hidden = np.dot(d_out, W2.T)
        show_detailed_math("9. HIDDEN ERROR (d_out * W2.T)", [d_out, W2.T], error_hidden)
        show_dot_logic("HiddenError", d_out, W2.T, row=1, col=2)
        
        sensitivity_hidden = sigmoid_derivative(hidden_scores)
        show_detailed_math("10. HIDDEN SENSITIVITY (Sig_Deriv)", [sensitivity_hidden], sensitivity_hidden, label="deriv(sigmoid(Z1))")
        show_activation_logic("Sens_Hidden", hidden_scores, sensitivity_hidden, "sigmoid_derivative")
        
        d_hidden = error_hidden * sensitivity_hidden
        show_detailed_math("11. d_hidden (Error * Sens)", [d_hidden], d_hidden, label="Error * Sensitivity")
        show_elementwise_logic("d_hidden", error_hidden, sensitivity_hidden, d_hidden, "*", row=1, col=2)
        
        d_W1 = np.dot(X.T, d_hidden)
        show_detailed_math("12. GRAD_W1 (X.T * d_hidden)", [X.T, d_hidden], d_W1)
        show_dot_logic("Grad_W1", X.T, d_hidden, row=1, col=2)

        # --- THE LEARNING STEP (Updating the weights) ---
        W2 -= learning_rate * d_W2
        b2 -= learning_rate * np.sum(d_out, axis=0, keepdims=True)
        W1 -= learning_rate * d_W1
        b1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

        print("\n" + "="*50)
        print(f"FINISH EPOCH {epoch}: Weights have been updated for the next round!")
    
    # After the loop finishes
    print("\n" + "="*50)
    print("✅ ALL EPOCHS COMPLETED!")
    print("Final Weights:")
    print("W1:\n", W1)
    print("b1:\n", b1)
    print("W2:\n", W2)
    print("b2:\n", b2)

    # Let's run one final forward pass to see the final result
    print("\n--- Final Predictions ---")
    final_hidden_scores = np.dot(X, W1) + b1
    final_hidden_activations = sigmoid(final_hidden_scores)
    final_scores = np.dot(final_hidden_activations, W2) + b2
    final_predictions = sigmoid(final_scores)
    
    print("Inputs X:")
    print(X)
    print("\nTarget y_true:")
    print(y_true)
    print("\nFinal Predictions (A2):")
    print(final_predictions)
    
    loss = np.mean((final_predictions - y_true)**2)
    print(f"\nFinal Mean Squared Error: {loss:.4f}")
    
    # Check accuracy (threshold at 0.5)
    accuracy = np.mean((final_predictions > 0.5).astype(int) == y_true) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
