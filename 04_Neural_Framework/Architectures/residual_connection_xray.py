import numpy as np

def residual_connection_xray():
    print("🛣️ RESIDUAL CONNECTIONS: THE HIGHWAY vs. THE SIDE ROAD")
    print("=" * 70)

    # 1. THE INPUT (The word 'River')
    x = np.array([1.0, 0.0, 0.5, 0.0])
    print(f"Original Word (Identity): {x}")

    # 2. THE LAYER (The 'Side Road')
    # Let's say this layer is messy and changes the word a lot
    layer_output = np.random.randn(4) * 0.2
    print(f"Layer Transformation:    {layer_output}")

    # 3. THE ADDITION (The Magic)
    # Output = Layer(x) + x
    final_output = layer_output + x

    print("\n" + "-" * 70)
    print(f"Final Output with Residual: {final_output}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. Notice that the final output still LOOKS like the original word.")
    print("   The 1.0 is now 1.1, the 0.5 is now 0.4. It's 'River' with context.")
    print("2. During Backprop, the gradient can 'Skip' the messy layer and")
    print("   travel back along the highway. This is why we can train 100 layers!")
    print("3. Without this, the model would 'forget' what word it was reading.")

if __name__ == "__main__":
    residual_connection_xray()
