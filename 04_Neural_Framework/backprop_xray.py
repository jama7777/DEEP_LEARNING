import numpy as np
from xray_utils import show_detailed_math, show_activation_logic, show_dot_logic

# --- THE BACKPROP X-RAY ---

class DenseLayerXRay:
    def __init__(self, input_size, output_size):
        # Initializing with simple numbers for readable math
        self.weights = np.array([
            [0.2, 0.8, -0.5],
            [0.5, -0.1, 0.4]
        ])
        self.biases = np.array([[0.1, 0.0, -0.1]])
        
    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        
        print("\n" + "="*60)
        print("🏗️ FORWARD PASS: LINEAR TRANSFORMATION (X @ W + b)")
        print("="*60)
        show_detailed_math("Z = Input @ Weights + Bias", [self.input, self.weights, self.biases], self.z, operation="*")
        return self.z

    def backward(self, grad_output):
        """
        grad_output is dL/dZ (The Error Signal arriving at this layer)
        """
        print("\n" + "="*60)
        print("🕵️ BACKWARD PASS: THE CHAIN RULE INVESTIGATION")
        print("="*60)
        
        # 1. dL/dW = X.T @ dL/dZ
        # How much did each weight contribute to the error?
        self.grad_weights = np.dot(self.input.T, grad_output)
        
        print("\n🔍 STEP 1: Gradient for Weights (dL/dW)")
        print("Logic: [Input Samples] @ [Error Signals]")
        show_detailed_math("dL/dW = Input.T @ dL/dZ", [self.input.T, grad_output], self.grad_weights, operation="*")
        
        # 2. dL/db = sum(dL/dZ)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        print("\n🔍 STEP 2: Gradient for Biases (dL/db)")
        show_detailed_math("dL/db = Sum of Error Signals", [grad_output], self.grad_biases, label="sum")

        # 3. dL/dX = dL/dZ @ W.T
        # Passing the blame back to the PREVIOUS layer
        grad_input = np.dot(grad_output, self.weights.T)
        print("\n🔍 STEP 3: Passing Blame Backward (dL/dX)")
        print("Logic: [Error Signal] @ [Weights.T]")
        show_detailed_math("dL/dX = dL/dZ @ Weights.T", [grad_output, self.weights.T], grad_input, operation="*")
        
        return grad_input

def main():
    # 1. SETUP DATA
    # Input: 1 sample with 2 features (e.g., embedding of a word)
    x = np.array([[1.0, -0.5]])
    target = np.array([[0.0, 1.0, 0.0]]) # One-hot target for 3 categories
    
    layer = DenseLayerXRay(2, 3)
    
    # 2. FORWARD PASS
    z = layer.forward(x)
    
    # Simple Softmax-like Error for visualization
    # Let's say dL/dZ is just (Prediction - Target) for simplicity
    prediction = z # Mocking raw scores
    error_signal = prediction - target
    
    print("\n" + "!"*60)
    print("🎯 THE ERROR HAS OCCURRED!")
    print(f"Target was {target[0]}, but we got {prediction[0]}.")
    print(f"Error Signal (dL/dZ): {error_signal[0]}")
    print("!"*60)

    # 3. BACKWARD PASS (The X-Ray)
    layer.backward(error_signal)

    print("\n💡 DEEP POINT:")
    print("Notice how dL/dW uses the INPUT. If a feature was 0, its weight update would be 0.")
    print("Notice how dL/dX uses the WEIGHTS. Stronger weights pass more blame backward.")

if __name__ == "__main__":
    main()
