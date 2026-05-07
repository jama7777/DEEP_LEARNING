import numpy as np
from xray_utils import show_detailed_math

# --- MATH HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def run_ten_layer_xor():
    # XOR DATA
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])
    
    np.random.seed(42)
    
    # ARCHITECTURE: 2 Inputs -> 10 Hidden Layers (1 neuron each) -> 1 Output
    # This means we need 11 Weight matrices
    weights = []
    biases = []
    
    # Layer 1: (2 inputs -> 1 neuron)
    weights.append(np.random.randn(2, 1))
    biases.append(np.zeros((1, 1)))
    
    # Layers 2 to 11: (1 neuron -> 1 neuron)
    for _ in range(10):
        weights.append(np.random.randn(1, 1))
        biases.append(np.zeros((1, 1)))
        
    lr = 0.5
    
    print("--- 🚀 THE 10-LAYER WATERFALL EXPERIMENT ---")
    print("Goal: Watch the 'Blame' travel through 11 separate bridges (Weights).")
    
    # 1. FORWARD PROPAGATION
    # Store all activations to use them in backprop
    activations = [X]
    for i in range(11):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = sigmoid(z)
        activations.append(a)
        
    prediction = activations[-1]
    
    print(f"\n{'='*30}\n   BACKWARD PROPAGATION (The Waterfall)\n{'='*30}")
    
    # 2. BACKWARD PROPAGATION
    # A. Initial Blame (Output Signal)
    gap = prediction - y_true
    # current_signal is our "active error" that moves backwards
    current_signal = 2 * gap * sigmoid_derivative(prediction)
    
    # B. The Cascade Loop
    # We go from Weight 11 down to Weight 1
    for i in reversed(range(11)):
        # Step 7: Update Weights for this bridge
        # dW = Activation_of_previous_layer * current_signal
        dW = np.dot(activations[i].T, current_signal)
        
        # Step 8: Pass Blame to the layer behind (This is what you're learning!)
        if i > 0:
            # 1. Crossing the bridge (Weight)
            error_back = np.dot(current_signal, weights[i].T)
            
            # 2. Filtering through the sensitivity (Derivative)
            # This becomes the 'current_signal' for the NEXT iteration (the layer behind)
            current_signal = error_back * sigmoid_derivative(activations[i])
            
            # Let's show the math for a few layers
            if i in [10, 5, 1]:
                print(f"\n--- Layer {i} Blame Calculation (W{i+1} -> H{i}) ---")
                show_detailed_math(f"H{i} SIGNAL (Signal * W{i+1}.T * Deriv)", [error_back], current_signal)
        
        # Apply the update
        weights[i] -= lr * dW
        
    print("\n[SUCCESS] The error signal has cascaded through all 11 bridges.")
    print("Each time it crossed a Weight, it was scaled by that weight's strength.")

if __name__ == "__main__":
    run_ten_layer_xor()
