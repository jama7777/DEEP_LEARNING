import numpy as np

def visualize_improvement():
    print("🚀 THE ERROR REDUCTION EXPERIMENT")
    print("=" * 60)

    # 1. SETUP
    # Input: Embedding of 'i love' (8 dimensions)
    # Output: Score for 'ai' (1 dimension for simplicity)
    x = np.array([[1.0, 0.5, -0.2, 0.1, 0.8, -0.1, 0.4, 0.2]])
    target = 1.0 # We want the score to be 1.0
    
    # Random initial weights
    weights = np.random.randn(8, 1) * 0.1
    lr = 0.5

    # --- PASS 1: BEFORE LEARNING ---
    prediction_1 = np.dot(x, weights)[0, 0]
    error_1 = target - prediction_1
    mse_1 = 0.5 * (error_1**2)
    
    print("\n--- 📉 PASS 1: THE FAILURE ---")
    print(f"Prediction: {prediction_1:.4f}")
    print(f"Goal:       {target:.4f}")
    print(f"Error Gap:  {error_1:.4f}")
    print(f"Loss (MSE): {mse_1:.4f}")

    # --- THE MATH (Backprop) ---
    # output_error = -(target - prediction) = -error_1
    # weights_error = x.T @ output_error
    output_error = -(target - prediction_1)
    weights_error = x.T * output_error
    
    print("\n--- 🛠️ THE NUDGE (Weights -= LR * Gradient) ---")
    print(f"Output Error Signal: {output_error:.4f}")
    print("First 3 Weight Updates:")
    for i in range(3):
        grad = weights_error[i, 0]
        old_w = weights[i, 0]
        new_w = old_w - (lr * grad)
        print(f"  W[{i}]: {old_w:+.4f} -> {new_w:+.4f} (Grad: {grad:+.4f})")

    # APPLY THE UPDATE
    weights -= lr * weights_error

    # --- PASS 2: AFTER LEARNING ---
    prediction_2 = np.dot(x, weights)[0, 0]
    error_2 = target - prediction_2
    mse_2 = 0.5 * (error_2**2)

    print("\n--- 📈 PASS 2: THE IMPROVEMENT ---")
    print(f"New Prediction: {prediction_2:.4f}")
    print(f"New Error Gap:  {error_2:.4f}")
    print(f"New Loss (MSE): {mse_2:.4f}")
    
    improvement = ((mse_1 - mse_2) / mse_1) * 100
    print(f"\n🏆 SUCCESS: Loss reduced by {improvement:.2f}% in a single step!")
    print("\n💡 DEEP REASON:")
    print("Because we subtracted the Gradient, we literally 'walked' down the slope")
    print("toward the target. The weights are now 'tuned' specifically for this input.")

if __name__ == "__main__":
    visualize_improvement()
