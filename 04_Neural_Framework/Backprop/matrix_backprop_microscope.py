import numpy as np

def deep_matrix_microscope():
    print("🔬 DEEP DIVE: THE TWO BACKPROP DOT PRODUCTS")
    print("=" * 60)

    # Simple Setup: 2 Inputs, 3 Outputs
    # Input X: [A, B]
    # Weights W: [W11, W12, W13]
    #            [W21, W22, W23]
    
    x = np.array([[10.0, 1.0]])      # Input A is very active, B is quiet
    error = np.array([[0.1, -0.2, 0.0]]) # Errors at the 3 outputs
    
    # Current Weights (for Step 2)
    weights = np.array([
        [0.5, 0.1, 0.8],
        [0.2, 0.9, 0.3]
    ])

    print("\n--- 🧱 PART 1: weights_error (dL/dW) ---")
    print("Formula: Input.T @ Error")
    print("Goal: Find out how much each connection is 'guilty'.")
    
    # Manually calculate one element to show the logic
    # guilt_w11 = input[0] * error[0]
    w_grad = np.dot(x.T, error)
    
    print(f"\nInput (X): {x[0]}")
    print(f"Error (dZ): {error[0]}")
    print(f"\nResulting Gradient Matrix (dL/dW):")
    print(w_grad)
    
    print("\n💡 VERBAL LOGIC for dL/dW11 (Weight from Input A to Output 1):")
    print(f"   Input A was {x[0,0]} and Output 1 had error {error[0,0]}.")
    print(f"   Guilt: {x[0,0]} * {error[0,0]} = {w_grad[0,0]}")
    print("   [DEEP POINT]: Because Input A was huge (10.0), any error it touched gets magnified.")
    print("   The weight from A takes 10x more blame than the weight from B!")

    print("\n" + "-"*60)
    print("\n--- 📢 PART 2: input_error (dL/dX) ---")
    print("Formula: Error @ Weights.T")
    print("Goal: Tell the previous layer how much they messed up.")
    
    x_grad = np.dot(error, weights.T)
    
    print(f"\nError (dZ): {error[0]}")
    print(f"Weights (W):\n{weights}")
    print(f"\nResulting Input Blame (dL/dX): {x_grad[0]}")
    
    print("\n💡 VERBAL LOGIC for dL/dA (Blame sent back to Input A):")
    print(f"   Output 1 had error {error[0,0]} and Weight A->1 was {weights[0,0]}.")
    print(f"   Output 2 had error {error[0,1]} and Weight A->2 was {weights[0,1]}.")
    print(f"   Output 3 had error {error[0,2]} and Weight A->3 was {weights[0,2]}.")
    print(f"   Total Blame: ({error[0,0]}*{weights[0,0]}) + ({error[0,1]}*{weights[0,1]}) + ({error[0,2]}*{weights[0,2]})")
    print(f"   Total Blame: {x_grad[0,0]:.4f}")
    print("   [DEEP POINT]: We use Weights.T because we are tracing the signal BACKWARDS")
    print("   along the same wires it came in on.")

if __name__ == "__main__":
    deep_matrix_microscope()
