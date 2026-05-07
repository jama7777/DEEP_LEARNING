import numpy as np

def relu_origami_xray():
    print("✂️ THE RELU ORIGAMI: FOLDING FLAT LOGIC")
    print("=" * 60)

    # Simple Inputs
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # 🏗️ SCENARIO 1: THE LINEAR TRAP (No ReLU)
    # Layer 1: y = x * 2
    # Layer 2: z = y * 0.5
    # Result: z = (x * 2) * 0.5 = x (No change!)
    y_linear = x * 2.0
    z_linear = y_linear * 0.5
    
    print("--- 🔴 SCENARIO 1: NO RELU (Stacked Linear) ---")
    print(f"Input:    {x}")
    print(f"Output:   {z_linear}")
    print("Logic: The two layers 'collapsed' into one. Still a straight line.")

    # 🏗️ SCENARIO 2: THE RELU FOLD
    # Layer 1: y = x * 2
    # ReLU:    r = max(0, y)
    # Layer 2: z = r * 0.5
    y_hidden = x * 2.0
    r_hidden = np.maximum(0, y_hidden)
    z_relu = r_hidden * 0.5
    
    print("\n--- 🟢 SCENARIO 2: WITH RELU (The Fold) ---")
    print(f"Input:    {x}")
    print(f"Hidden:   {y_hidden} (Before Fold)")
    print(f"ReLU:     {r_hidden} (The Fold)")
    print(f"Output:   {z_relu}")
    
    print("\n" + "=" * 60)
    print("💡 THE DEEP TRUTH:")
    print("1. In Scenario 1, the output is exactly the same as the input.")
    print("2. In Scenario 2, the logic 'BENT' at zero.")
    print("3. This 'Bend' is what allows the network to say:")
    print("   'IF the word is happy THEN predict ai, ELSE predict nothing.'")
    print("\n[CONCLUSION]: ReLU is the 'Switch' that enables logic.")

if __name__ == "__main__":
    relu_origami_xray()
