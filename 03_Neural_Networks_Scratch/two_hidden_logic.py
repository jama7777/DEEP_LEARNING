import numpy as np
from xray_utils import show_detailed_math

# --- MATH HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is the activated value
    return x * (1 - x)

def run_two_layer_network():
    # DATA: One input (0.5), One target (0.8)
    X = np.array([[0.5]])
    y_true = np.array([[0.8]])
    
    # WEIGHTS: 1 Input -> 1 Hidden (H1) -> 1 Hidden (H2) -> 1 Output
    np.random.seed(42)
    W1 = np.random.randn(1, 1); b1 = np.zeros((1, 1))
    W2 = np.random.randn(1, 1); b2 = np.zeros((1, 1))
    W3 = np.random.randn(1, 1); b3 = np.zeros((1, 1))
    
    lr = 1.0 
    
    print("--- 🔬 THE DEEP NEURON (1-1-1-1) ---")
    
    # --- FORWARD PROPAGATION ---
    # 1. Hidden Layer 1
    h1_score = np.dot(X, W1) + b1
    h1_act = sigmoid(h1_score)
    show_detailed_math("1. HIDDEN 1 ACTIVATION", [h1_act], h1_act, label="sigmoid")
    
    # 2. Hidden Layer 2
    h2_score = np.dot(h1_act, W2) + b2
    h2_act = sigmoid(h2_score)
    show_detailed_math("2. HIDDEN 2 ACTIVATION", [h2_act], h2_act, label="sigmoid")
    
    # 3. Output Layer
    out_score = np.dot(h2_act, W3) + b3
    out_pred = sigmoid(out_score)
    show_detailed_math("3. FINAL PREDICTION", [out_pred], out_pred, label="prediction")

    print(f"\n{'='*30}")
    print(f"   BACKWARD PROPAGATION")
    print(f"{'='*30}")
    
    # 4. Output Signal (The Master Blame)
    gap = out_pred - y_true
    out_signal = 2 * gap * sigmoid_derivative(out_pred)
    show_detailed_math("4. OUTPUT SIGNAL (Total Error)", [out_signal], out_signal)
    
    # 5. Update W3 (Last Bridge)
    dW3 = np.dot(h2_act.T, out_signal)
    show_detailed_math("5. dW3 (H2_Act * Out_Sig)", [h2_act.T, out_signal], dW3)
    
    # 6. Pass Blame to Hidden 2
    h2_error = np.dot(out_signal, W3.T) # Traveling back through W3
    h2_signal = h2_error * sigmoid_derivative(h2_act)
    show_detailed_math("6. HIDDEN 2 SIGNAL (Via W3)", [h2_error, h2_act], h2_signal)
    
    # 7. Update W2 (Middle Bridge)
    dW2 = np.dot(h1_act.T, h2_signal)
    show_detailed_math("7. dW2 (H1_Act * H2_Sig)", [h1_act.T, h2_signal], dW2)
    
    # 8. Pass Blame to Hidden 1
    h1_error = np.dot(h2_signal, W2.T) # Traveling back through W2
    h1_signal = h1_error * sigmoid_derivative(h1_act)
    show_detailed_math("8. HIDDEN 1 SIGNAL (Via W2)", [h1_error, h1_act], h1_signal)
    
    # 9. Update W1 (First Bridge)
    dW1 = np.dot(X.T, h1_signal)
    show_detailed_math("9. dW1 (X * H1_Sig)", [X.T, h1_signal], dW1)

    # APPLY UPDATES
    W3 -= lr * dW3
    W2 -= lr * dW2
    W1 -= lr * dW1
    
    print("\n[SYSTEM] Deep backprop complete. Signals cascaded through all layers.")

if __name__ == "__main__":
    run_two_layer_network()
